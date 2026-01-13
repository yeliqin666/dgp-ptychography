"""
Conventional Pixelated Ptychography Reconstructor

Implements standard iterative ptychography algorithms for initialization:
- Gradient descent optimization
- ePIE algorithm
- Other standard methods

Used in Stage 1 of the DGP reconstruction pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict
from tqdm import tqdm

from .forward_model import MultisliceForwardModel


class ConventionalReconstructor:
    """
    Conventional pixelated ptychography reconstructor.
    
    Uses gradient descent or iterative algorithms to reconstruct
    object and probe on pixelated grids.
    """
    
    def __init__(
        self,
        measured_intensities: np.ndarray,
        scan_positions: np.ndarray,
        pixel_size: float,
        wavelength: float,
        object_shape: tuple,
        probe_shape: tuple,
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.device = torch.device(device)
        self.verbose = verbose
        
        # Convert to torch
        self.measured_intensities = torch.from_numpy(
            measured_intensities
        ).float().to(self.device)
        
        self.scan_positions = torch.from_numpy(
            scan_positions
        ).float().to(self.device)
        
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.object_shape = object_shape
        self.probe_shape = probe_shape
        self.num_positions = measured_intensities.shape[0]
        
        # Initialize object and probe
        self.obj = None
        self.probe = None
        self._initialize_reconstruction()
    
    def _initialize_reconstruction(self):
        """Initialize object and probe."""
        # Initialize object as complex zeros (transmission = 1 = exp(i*0))
        self.obj = nn.Parameter(
            torch.zeros(
                self.object_shape,
                dtype=torch.complex64,
                device=self.device
            )
        )
        
        # Initialize probe with circular aperture
        probe = self._make_circular_probe()
        self.probe = nn.Parameter(probe)
        
        if self.verbose:
            print(f"Initialized object: {self.obj.shape}")
            print(f"Initialized probe: {self.probe.shape}")
    
    def _make_circular_probe(self) -> torch.Tensor:
        """Create initial probe as circular aperture."""
        h, w = self.probe_shape
        
        # Frequency coordinates
        y = torch.arange(h, device=self.device) - h // 2
        x = torch.arange(w, device=self.device) - w // 2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # Circular aperture
        radius = min(h, w) // 4
        aperture = ((X**2 + Y**2) <= radius**2).float()
        
        # Add some phase curvature (defocus)
        phase = 0.01 * (X**2 + Y**2)
        
        # Complex probe
        probe = aperture * torch.exp(1j * phase)
        
        # Normalize
        probe = probe / torch.sqrt(torch.sum(torch.abs(probe)**2))
        
        return probe.unsqueeze(0)  # Add mode dimension
    
    def reconstruct(
        self,
        iterations: int = 50,
        method: str = 'gradient_descent',
        lr_obj: float = 0.1,
        lr_probe: float = 0.01,
        batch_size: Optional[int] = None
    ) -> Dict:
        """
        Run conventional reconstruction.
        
        Args:
            iterations: Number of iterations
            method: 'gradient_descent' or 'epie'
            lr_obj: Learning rate for object
            lr_probe: Learning rate for probe
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with results
        """
        
        if method == 'gradient_descent':
            return self._reconstruct_gradient_descent(
                iterations, lr_obj, lr_probe, batch_size
            )
        elif method == 'epie':
            return self._reconstruct_epie(iterations, batch_size)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _reconstruct_gradient_descent(
        self,
        iterations: int,
        lr_obj: float,
        lr_probe: float,
        batch_size: Optional[int]
    ) -> Dict:
        """Gradient descent reconstruction."""
        
        # Create forward model
        forward_model = MultisliceForwardModel(
            probe_shape=self.probe_shape,
            object_shape=self.object_shape,
            pixel_size=self.pixel_size,
            wavelength=self.wavelength,
            slice_thickness=1.0,  # Not used for single slice
            device=str(self.device)
        ).to(self.device)
        
        # Optimizers
        opt_obj = optim.SGD([self.obj], lr=lr_obj)
        opt_probe = optim.SGD([self.probe], lr=lr_probe)
        
        # Loss function
        loss_fn = nn.MSELoss()
        
        if batch_size is None:
            batch_size = min(32, self.num_positions)
        
        num_batches = (self.num_positions + batch_size - 1) // batch_size
        
        loss_history = []
        
        pbar = tqdm(range(iterations), desc="Reconstruction", disable=not self.verbose)
        
        for it in pbar:
            epoch_loss = 0.0
            
            # Shuffle positions
            indices = torch.randperm(self.num_positions, device=self.device)
            
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, self.num_positions)
                batch_indices = indices[start:end]
                
                opt_obj.zero_grad()
                opt_probe.zero_grad()
                
                # Forward pass
                # Convert transmission to complex object
                obj_complex = torch.exp(1j * self.obj)
                
                predicted = forward_model(
                    probe=self.probe,
                    obj=obj_complex.squeeze(0),  # Remove batch dim
                    positions=self.scan_positions,
                    batch_indices=batch_indices
                )
                
                measured_batch = self.measured_intensities[batch_indices]
                
                # Loss
                loss = loss_fn(predicted, measured_batch)
                
                # Backward
                loss.backward()
                opt_obj.step()
                opt_probe.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            
            if it % 10 == 0:
                pbar.set_postfix({'loss': f'{avg_loss:.6e}'})
        
        # Return results
        with torch.no_grad():
            obj_complex = torch.exp(1j * self.obj)
        
        return {
            'object': obj_complex.cpu().numpy(),
            'probe': self.probe.cpu().numpy(),
            'loss_history': loss_history
        }
    
    def _reconstruct_epie(
        self,
        iterations: int,
        batch_size: Optional[int]
    ) -> Dict:
        """Extended Ptychographic Iterative Engine (ePIE) algorithm."""
        
        # ePIE parameters
        alpha = 0.5  # Object update weight
        beta = 0.5   # Probe update weight
        
        if batch_size is None:
            batch_size = 1  # ePIE traditionally processes one position at a time
        
        loss_history = []
        
        pbar = tqdm(range(iterations), desc="ePIE", disable=not self.verbose)
        
        for it in pbar:
            epoch_loss = 0.0
            
            # Process each scan position
            indices = torch.randperm(self.num_positions, device=self.device)
            
            for idx in indices[:batch_size]:
                pos = self.scan_positions[idx]
                measured = self.measured_intensities[idx]
                
                # Extract object patch
                py, px = pos.round().long()
                py, px = py.item(), px.item()
                
                h, w = self.probe_shape
                obj_patch = torch.exp(
                    1j * self.obj[0, py:py+h, px:px+w]
                )
                
                # Exit wave
                exit_wave = self.probe[0] * obj_patch
                
                # FFT to detector
                exit_wave_fft = torch.fft.fft2(exit_wave)
                exit_wave_fft = torch.fft.fftshift(exit_wave_fft)
                
                # Apply amplitude constraint
                intensity_pred = torch.abs(exit_wave_fft)**2
                amplitude_meas = torch.sqrt(measured)
                phase_pred = torch.angle(exit_wave_fft)
                
                exit_wave_fft_new = amplitude_meas * torch.exp(1j * phase_pred)
                
                # IFFT back
                exit_wave_fft_new = torch.fft.ifftshift(exit_wave_fft_new)
                exit_wave_new = torch.fft.ifft2(exit_wave_fft_new)
                
                # Update object
                probe_conj = torch.conj(self.probe[0])
                probe_max = torch.max(torch.abs(self.probe[0]))**2
                
                obj_update = alpha * probe_conj / (probe_max + 1e-8) * (
                    exit_wave_new - exit_wave
                )
                
                with torch.no_grad():
                    obj_patch_new = obj_patch + obj_update
                    # Convert back to phase
                    self.obj.data[0, py:py+h, px:px+w] = torch.angle(obj_patch_new)
                
                # Update probe
                obj_conj = torch.conj(obj_patch)
                obj_max = torch.max(torch.abs(obj_patch))**2
                
                probe_update = beta * obj_conj / (obj_max + 1e-8) * (
                    exit_wave_new - exit_wave
                )
                
                with torch.no_grad():
                    self.probe.data[0] = self.probe.data[0] + probe_update
                
                # Compute loss
                loss = torch.mean((intensity_pred - measured)**2)
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(indices[:batch_size])
            loss_history.append(avg_loss)
            
            if it % 10 == 0:
                pbar.set_postfix({'loss': f'{avg_loss:.6e}'})
        
        # Return results
        with torch.no_grad():
            obj_complex = torch.exp(1j * self.obj)
        
        return {
            'object': obj_complex.cpu().numpy(),
            'probe': self.probe.cpu().numpy(),
            'loss_history': loss_history
        }
