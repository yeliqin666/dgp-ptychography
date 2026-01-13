"""
DGP Ptychography Reconstructor

Implements the three-stage reconstruction pipeline:
1. Conventional pixelated reconstruction (initialization)
2. DGP pre-training (autoencoders)
3. Joint DGP optimization

Based on McCray et al., "Deep generative priors for robust and efficient
electron ptychography" (2025).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
import time

from .forward_model import MultisliceForwardModel, electron_wavelength
from .models import create_dgp, count_parameters
from .losses import CombinedLoss
from .conventional import ConventionalReconstructor


class DGPPtychographyReconstructor:
    """
    Deep Generative Prior Ptychography Reconstructor.
    
    Implements the complete DGP-enabled reconstruction pipeline with
    pre-training and joint optimization.
    
    Args:
        measured_intensities: Experimental diffraction data (N, H, W)
        scan_positions: Probe positions in pixels (N, 2)
        pixel_size: Real-space pixel size (Angstroms)
        energy: Electron energy (eV)
        wavelength: Electron wavelength (Angstroms), computed if not provided
        num_slices: Number of object slices (1 for single-slice)
        device: PyTorch device
        verbose: Print progress
    """
    
    def __init__(
        self,
        measured_intensities: np.ndarray,
        scan_positions: np.ndarray,
        pixel_size: float,
        energy: float = 300e3,
        wavelength: Optional[float] = None,
        num_slices: int = 1,
        slice_thickness: Optional[float] = None,
        device: str = 'cuda',
        verbose: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Convert data to torch
        self.measured_intensities = torch.from_numpy(
            measured_intensities
        ).float().to(self.device)
        
        self.scan_positions = torch.from_numpy(
            scan_positions
        ).float().to(self.device)
        
        self.pixel_size = pixel_size
        self.energy = energy
        self.wavelength = wavelength if wavelength else electron_wavelength(energy)
        self.num_slices = num_slices
        
        # Default slice thickness (if multislice)
        if slice_thickness is None and num_slices > 1:
            self.slice_thickness = 1.0  # Angstroms
        else:
            self.slice_thickness = slice_thickness or 1.0
        
        # Dataset info
        self.num_positions = measured_intensities.shape[0]
        self.probe_shape = measured_intensities.shape[1:]
        
        # Estimate object size from scan positions
        max_y = scan_positions[:, 0].max()
        max_x = scan_positions[:, 1].max()
        self.object_shape = (
            num_slices,
            int(max_y) + self.probe_shape[0] + 10,
            int(max_x) + self.probe_shape[1] + 10
        )
        
        if self.verbose:
            print("=" * 60)
            print("DGP Ptychography Reconstructor Initialized")
            print("=" * 60)
            print(f"Device: {self.device}")
            print(f"Energy: {energy/1e3:.1f} keV")
            print(f"Wavelength: {self.wavelength:.4f} Å")
            print(f"Pixel size: {pixel_size:.4f} Å")
            print(f"Probe shape: {self.probe_shape}")
            print(f"Object shape: {self.object_shape}")
            print(f"Scan positions: {self.num_positions}")
            print(f"Number of slices: {num_slices}")
            print("=" * 60)
        
        # Storage for results
        self.estimated_obj = None
        self.estimated_probe = None
        self.obj_dgp = None
        self.probe_dgp = None
        self.history = {'stage1': [], 'stage2': [], 'stage3': []}
    
    def reconstruct(
        self,
        # Stage 1: Conventional reconstruction
        stage1_iterations: int = 50,
        stage1_method: str = 'gradient_descent',
        
        # Stage 2: DGP pre-training
        stage2_iterations: int = 50,
        stage2_lr: float = 1e-3,
        
        # Stage 3: Joint optimization
        stage3_iterations: int = 100,
        stage3_lr_obj: float = 1e-4,
        stage3_lr_probe: float = 1e-4,
        
        # DGP architecture
        num_layers: int = 3,
        start_filters: int = 16,
        obj_final_activation: str = 'identity',
        probe_final_activation: str = 'identity',
        
        # Loss weights
        tv_weight_xy: float = 0.0,
        tv_weight_z: float = 0.0,
        surface_zero_weight: float = 0.0,
        
        # Optimization
        use_adam: bool = True,
        noise_sigma: float = 0.025,
        batch_size: Optional[int] = None,
        
    ) -> Dict:
        """
        Run complete three-stage reconstruction.
        
        Returns:
            Dictionary with reconstructed object, probe, and history
        """
        
        # ============================================
        # STAGE 1: Conventional Pixelated Reconstruction
        # ============================================
        if self.verbose:
            print("\n" + "=" * 60)
            print("STAGE 1: Conventional Reconstruction")
            print("=" * 60)
        
        self._stage1_conventional_reconstruction(
            iterations=stage1_iterations,
            method=stage1_method
        )
        
        # ============================================
        # STAGE 2: DGP Pre-training
        # ============================================
        if self.verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: DGP Pre-training")
            print("=" * 60)
        
        self._stage2_dgp_pretraining(
            iterations=stage2_iterations,
            lr=stage2_lr,
            num_layers=num_layers,
            start_filters=start_filters,
            obj_final_activation=obj_final_activation,
            probe_final_activation=probe_final_activation
        )
        
        # ============================================
        # STAGE 3: Joint DGP Optimization
        # ============================================
        if self.verbose:
            print("\n" + "=" * 60)
            print("STAGE 3: Joint DGP Optimization")
            print("=" * 60)
        
        self._stage3_joint_optimization(
            iterations=stage3_iterations,
            lr_obj=stage3_lr_obj,
            lr_probe=stage3_lr_probe,
            tv_weight_xy=tv_weight_xy,
            tv_weight_z=tv_weight_z,
            surface_zero_weight=surface_zero_weight,
            use_adam=use_adam,
            noise_sigma=noise_sigma,
            batch_size=batch_size
        )
        
        # Return final results
        return self.get_results()
    
    def _stage1_conventional_reconstruction(
        self,
        iterations: int,
        method: str
    ):
        """Stage 1: Conventional pixelated reconstruction for initialization."""
        
        reconstructor = ConventionalReconstructor(
            measured_intensities=self.measured_intensities.cpu().numpy(),
            scan_positions=self.scan_positions.cpu().numpy(),
            pixel_size=self.pixel_size,
            wavelength=self.wavelength,
            object_shape=self.object_shape,
            probe_shape=self.probe_shape,
            device=str(self.device),
            verbose=self.verbose
        )
        
        result = reconstructor.reconstruct(
            iterations=iterations,
            method=method
        )
        
        # Store results
        self.estimated_obj = torch.from_numpy(result['object']).to(self.device)
        self.estimated_probe = torch.from_numpy(result['probe']).to(self.device)
        self.history['stage1'] = result['loss_history']
        
        if self.verbose:
            print(f"✓ Conventional reconstruction complete")
            print(f"  Final loss: {self.history['stage1'][-1]:.6e}")
    
    def _stage2_dgp_pretraining(
        self,
        iterations: int,
        lr: float,
        num_layers: int,
        start_filters: int,
        obj_final_activation: str,
        probe_final_activation: str
    ):
        """Stage 2: Pre-train DGPs as autoencoders."""
        
        # Create DGP models
        # Object DGP
        obj_channels = self.num_slices * 2  # Complex = 2 channels per slice
        self.obj_dgp = create_dgp(
            in_channels=obj_channels,
            out_channels=obj_channels,
            num_layers=num_layers,
            start_filters=start_filters,
            final_activation=obj_final_activation,
            output_complex=False
        ).to(self.device)
        
        # Probe DGP (assume single mode for simplicity, can extend to mixed-state)
        probe_channels = 2  # Complex
        self.probe_dgp = create_dgp(
            in_channels=probe_channels,
            out_channels=probe_channels,
            num_layers=num_layers,
            start_filters=start_filters,
            final_activation=probe_final_activation,
            output_complex=False
        ).to(self.device)
        
        if self.verbose:
            print(f"Object DGP: {count_parameters(self.obj_dgp):,} parameters")
            print(f"Probe DGP: {count_parameters(self.probe_dgp):,} parameters")
        
        # Pre-train object DGP
        if self.verbose:
            print("\nPre-training object DGP...")
        
        self._pretrain_autoencoder(
            dgp=self.obj_dgp,
            target=self.estimated_obj,
            iterations=iterations,
            lr=lr,
            name="Object"
        )
        
        # Pre-train probe DGP
        if self.verbose:
            print("\nPre-training probe DGP...")
        
        self._pretrain_autoencoder(
            dgp=self.probe_dgp,
            target=self.estimated_probe,
            iterations=iterations,
            lr=lr,
            name="Probe"
        )
        
        if self.verbose:
            print(f"\n✓ DGP pre-training complete")
    
    def _pretrain_autoencoder(
        self,
        dgp: nn.Module,
        target: torch.Tensor,
        iterations: int,
        lr: float,
        name: str
    ):
        """Pre-train a DGP as an autoencoder."""
        
        optimizer = optim.Adam(dgp.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # Prepare input (complex to real-imag channels)
        if torch.is_complex(target):
            target_real = torch.stack([target.real, target.imag], dim=0)
        else:
            target_real = target
        
        # Ensure target has batch dimension
        if target_real.ndim == 3:
            target_real = target_real.unsqueeze(0)
        
        losses = []
        pbar = tqdm(range(iterations), desc=f"{name} DGP", disable=not self.verbose)
        
        for it in pbar:
            optimizer.zero_grad()
            
            # Forward through DGP
            output = dgp.dgp(target_real)
            
            # Reconstruction loss
            loss = loss_fn(output, target_real)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if it % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.6e}'})
        
        self.history['stage2'].append({name: losses})
    
    def _stage3_joint_optimization(
        self,
        iterations: int,
        lr_obj: float,
        lr_probe: float,
        tv_weight_xy: float,
        tv_weight_z: float,
        surface_zero_weight: float,
        use_adam: bool,
        noise_sigma: float,
        batch_size: Optional[int]
    ):
        """Stage 3: Joint optimization with DGPs in forward model."""
        
        # Create forward model
        forward_model = MultisliceForwardModel(
            probe_shape=self.probe_shape,
            object_shape=self.object_shape,
            pixel_size=self.pixel_size,
            wavelength=self.wavelength,
            slice_thickness=self.slice_thickness,
            device=str(self.device)
        ).to(self.device)
        
        # Create loss function
        loss_fn = CombinedLoss(
            fidelity_weight=1.0,
            tv_weight_xy=tv_weight_xy,
            tv_weight_z=tv_weight_z,
            surface_zero_weight=surface_zero_weight,
            fidelity_type='mse'
        ).to(self.device)
        
        # Optimizers
        if use_adam:
            opt_obj = optim.Adam(self.obj_dgp.parameters(), lr=lr_obj)
            opt_probe = optim.Adam(self.probe_dgp.parameters(), lr=lr_probe)
        else:
            opt_obj = optim.SGD(self.obj_dgp.parameters(), lr=lr_obj)
            opt_probe = optim.SGD(self.probe_dgp.parameters(), lr=lr_probe)
        
        # Prepare inputs for DGPs
        obj_input = self._prepare_dgp_input(self.estimated_obj)
        probe_input = self._prepare_dgp_input(self.estimated_probe)
        
        # Training loop
        if batch_size is None:
            batch_size = min(32, self.num_positions)
        
        num_batches = (self.num_positions + batch_size - 1) // batch_size
        
        losses_total = []
        pbar = tqdm(range(iterations), desc="Joint optimization", disable=not self.verbose)
        
        for it in pbar:
            epoch_loss = 0.0
            
            # Shuffle positions
            indices = torch.randperm(self.num_positions, device=self.device)
            
            for batch_idx in range(num_batches):
                # Get batch
                start = batch_idx * batch_size
                end = min(start + batch_size, self.num_positions)
                batch_indices = indices[start:end]
                
                # Zero gradients
                opt_obj.zero_grad()
                opt_probe.zero_grad()
                
                # Add noise to DGP inputs (helps avoid local minima)
                obj_noisy = obj_input + noise_sigma * torch.randn_like(obj_input)
                probe_noisy = probe_input + noise_sigma * torch.randn_like(probe_input)
                
                # Generate object and probe from DGPs
                obj_generated = self.obj_dgp.dgp(obj_noisy)
                probe_generated = self.probe_dgp.dgp(probe_noisy)
                
                # Convert to complex
                obj_complex = self._to_complex(obj_generated)
                probe_complex = self._to_complex(probe_generated)
                
                # Forward model
                predicted = forward_model(
                    probe=probe_complex.unsqueeze(0),  # Add mode dimension
                    obj=obj_complex.squeeze(0),  # Remove batch dimension
                    positions=self.scan_positions,
                    batch_indices=batch_indices
                )
                
                # Measured intensities for this batch
                measured_batch = self.measured_intensities[batch_indices]
                
                # Compute losses
                loss_dict = loss_fn(
                    predicted_intensity=predicted,
                    measured_intensity=measured_batch,
                    obj=obj_complex,
                    probe=probe_complex
                )
                
                loss = loss_dict['total']
                
                # Backward
                loss.backward()
                opt_obj.step()
                opt_probe.step()
                
                epoch_loss += loss.item()
            
            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            losses_total.append(avg_loss)
            
            if it % 10 == 0:
                pbar.set_postfix({'loss': f'{avg_loss:.6e}'})
        
        self.history['stage3'] = losses_total
        
        # Store final reconstructions
        with torch.no_grad():
            obj_final = self.obj_dgp.dgp(obj_input)
            probe_final = self.probe_dgp.dgp(probe_input)
            
            self.estimated_obj = self._to_complex(obj_final).squeeze(0)
            self.estimated_probe = self._to_complex(probe_final).squeeze(0)
        
        if self.verbose:
            print(f"\n✓ Joint optimization complete")
            print(f"  Final loss: {losses_total[-1]:.6e}")
    
    def _prepare_dgp_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare complex tensor as DGP input (real-imag channels)."""
        if torch.is_complex(x):
            x_real = torch.stack([x.real, x.imag], dim=0)
        else:
            x_real = x
        
        if x_real.ndim == 3:
            x_real = x_real.unsqueeze(0)  # Add batch dimension
        
        return x_real
    
    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Convert real-imag channels to complex."""
        if x.shape[1] == 2:
            # (B, 2, H, W) -> (B, H, W) complex
            return torch.complex(x[:, 0], x[:, 1])
        else:
            # (B, C, H, W) with C = num_slices * 2
            # Reshape to (B, num_slices, 2, H, W) then combine
            B, C, H, W = x.shape
            num_slices = C // 2
            x = x.reshape(B, num_slices, 2, H, W)
            return torch.complex(x[:, :, 0], x[:, :, 1])
    
    def get_results(self) -> Dict:
        """Get reconstruction results."""
        return {
            'object': self.estimated_obj.cpu().numpy(),
            'probe': self.estimated_probe.cpu().numpy(),
            'history': self.history,
            'obj_dgp': self.obj_dgp,
            'probe_dgp': self.probe_dgp
        }
