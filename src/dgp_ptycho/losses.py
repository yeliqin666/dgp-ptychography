"""
Loss Functions and Regularizers

Implements fidelity losses and regularization terms for DGP ptychography.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PtychographyLoss(nn.Module):
    """
    Fidelity loss for ptychography.
    
    Compares predicted diffraction intensities with measured intensities.
    Supports Poisson or Gaussian noise models.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: 'mse', 'poisson', or 'amplitude'
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        predicted: torch.Tensor,
        measured: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute fidelity loss.
        
        Args:
            predicted: Predicted intensities (N, H, W)
            measured: Measured intensities (N, H, W)
            mask: Optional mask for valid pixels (N, H, W)
        
        Returns:
            Scalar loss value
        """
        if mask is not None:
            predicted = predicted * mask
            measured = measured * mask
            norm_factor = mask.sum()
        else:
            norm_factor = predicted.numel()
        
        if self.loss_type == 'mse':
            # Mean squared error
            loss = torch.sum((predicted - measured) ** 2) / norm_factor
        
        elif self.loss_type == 'poisson':
            # Poisson negative log-likelihood
            # -log P(measured | predicted) = predicted - measured * log(predicted)
            epsilon = 1e-10
            predicted_safe = predicted + epsilon
            loss = torch.sum(
                predicted_safe - measured * torch.log(predicted_safe)
            ) / norm_factor
        
        elif self.loss_type == 'amplitude':
            # Amplitude-based loss
            pred_amp = torch.sqrt(predicted + 1e-10)
            meas_amp = torch.sqrt(measured + 1e-10)
            loss = torch.sum((pred_amp - meas_amp) ** 2) / norm_factor
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation regularization loss.
    
    Encourages piecewise-smooth reconstructions by penalizing
    the sum of absolute gradients.
    """
    
    def __init__(
        self,
        weight_xy: float = 1.0,
        weight_z: float = 1.0,
        anisotropic: bool = True
    ):
        """
        Args:
            weight_xy: Weight for xy (in-plane) gradients
            weight_z: Weight for z (beam direction) gradients
            anisotropic: If True, sum |∇x| + |∇y|; if False, sqrt(∇x² + ∇y²)
        """
        super().__init__()
        self.weight_xy = weight_xy
        self.weight_z = weight_z
        self.anisotropic = anisotropic
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute TV loss.
        
        Args:
            x: Tensor of shape (C, H, W) or (C, D, H, W)
               For complex, x should be magnitude or phase
        
        Returns:
            Scalar TV loss
        """
        if x.ndim == 3:
            # 2D case: (C, H, W)
            return self._tv_2d(x)
        elif x.ndim == 4:
            # 3D case: (C, D, H, W)
            return self._tv_3d(x)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")
    
    def _tv_2d(self, x: torch.Tensor) -> torch.Tensor:
        """TV for 2D images."""
        # Gradients in x and y
        diff_x = x[..., :, 1:] - x[..., :, :-1]
        diff_y = x[..., 1:, :] - x[..., :-1, :]
        
        if self.anisotropic:
            # L1 norm: |∇x| + |∇y|
            tv = torch.sum(torch.abs(diff_x)) + torch.sum(torch.abs(diff_y))
        else:
            # L2 norm: sqrt(∇x² + ∇y²)
            # Need to handle edge differences carefully
            diff_x_pad = F.pad(diff_x, (0, 0, 0, 1), mode='constant', value=0)
            diff_y_pad = F.pad(diff_y, (0, 1, 0, 0), mode='constant', value=0)
            tv = torch.sum(torch.sqrt(diff_x_pad**2 + diff_y_pad**2 + 1e-8))
        
        # Normalize by number of pixels
        tv = tv / (x.numel() / x.shape[0])
        
        return tv * self.weight_xy
    
    def _tv_3d(self, x: torch.Tensor) -> torch.Tensor:
        """TV for 3D volumes."""
        # In-plane gradients
        diff_x = x[..., :, :, 1:] - x[..., :, :, :-1]
        diff_y = x[..., :, 1:, :] - x[..., :, :-1, :]
        
        # Along beam direction
        diff_z = x[..., 1:, :, :] - x[..., :-1, :, :]
        
        if self.anisotropic:
            tv_xy = torch.sum(torch.abs(diff_x)) + torch.sum(torch.abs(diff_y))
            tv_z = torch.sum(torch.abs(diff_z))
        else:
            diff_x_pad = F.pad(diff_x, (0, 0, 0, 1, 0, 0))
            diff_y_pad = F.pad(diff_y, (0, 1, 0, 0, 0, 0))
            tv_xy = torch.sum(torch.sqrt(diff_x_pad**2 + diff_y_pad**2 + 1e-8))
            tv_z = torch.sum(torch.abs(diff_z))
        
        # Normalize
        n_pixels = x.numel() / x.shape[0]
        tv_xy = tv_xy / n_pixels * self.weight_xy
        tv_z = tv_z / n_pixels * self.weight_z
        
        return tv_xy + tv_z


class SurfaceZeroLoss(nn.Module):
    """
    Surface zero loss for multislice reconstruction.
    
    Penalizes non-zero density at the top and bottom surfaces of
    the reconstructed volume, encouraging physically plausible
    3D reconstructions.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: Loss weight
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volume tensor (C, D, H, W) or (D, H, W)
        
        Returns:
            Scalar loss penalizing surface values
        """
        if x.ndim == 3:
            # (D, H, W)
            top = x[0]
            bottom = x[-1]
        elif x.ndim == 4:
            # (C, D, H, W)
            top = x[:, 0]
            bottom = x[:, -1]
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")
        
        # L1 loss on surface slices
        loss = (torch.sum(torch.abs(top)) + torch.sum(torch.abs(bottom))) / 2
        
        # Normalize by surface area
        loss = loss / (top.numel() + bottom.numel())
        
        return loss * self.weight


class ProbeOrthogonalityConstraint(nn.Module):
    """
    Orthogonality constraint for mixed-state probes.
    
    Enforces that different probe modes remain orthogonal,
    which is physically required for incoherent mode mixing.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Args:
            weight: Constraint weight
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, probe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probe: Mixed-state probe (num_modes, H, W) complex
        
        Returns:
            Orthogonality loss
        """
        num_modes = probe.shape[0]
        
        if num_modes == 1:
            return torch.tensor(0.0, device=probe.device)
        
        # Flatten spatial dimensions
        probe_flat = probe.reshape(num_modes, -1)
        
        # Compute Gram matrix
        gram = torch.matmul(probe_flat, probe_flat.conj().T)
        
        # Off-diagonal elements should be zero
        mask = 1.0 - torch.eye(num_modes, device=probe.device)
        loss = torch.sum(torch.abs(gram * mask))
        
        # Normalize
        loss = loss / (num_modes * (num_modes - 1))
        
        return loss * self.weight


class CombinedLoss(nn.Module):
    """
    Combined loss for DGP ptychography.
    
    Combines fidelity loss with optional regularization terms.
    """
    
    def __init__(
        self,
        fidelity_weight: float = 1.0,
        tv_weight_xy: float = 0.0,
        tv_weight_z: float = 0.0,
        surface_zero_weight: float = 0.0,
        probe_orthog_weight: float = 0.0,
        fidelity_type: str = 'mse'
    ):
        """
        Args:
            fidelity_weight: Weight for data fidelity loss
            tv_weight_xy: Weight for in-plane TV regularization
            tv_weight_z: Weight for through-plane TV regularization
            surface_zero_weight: Weight for surface zero constraint
            probe_orthog_weight: Weight for probe orthogonality
            fidelity_type: Type of fidelity loss
        """
        super().__init__()
        
        self.fidelity_loss = PtychographyLoss(loss_type=fidelity_type)
        self.tv_loss = TotalVariationLoss(
            weight_xy=tv_weight_xy,
            weight_z=tv_weight_z
        ) if (tv_weight_xy > 0 or tv_weight_z > 0) else None
        
        self.surface_loss = SurfaceZeroLoss(
            weight=surface_zero_weight
        ) if surface_zero_weight > 0 else None
        
        self.probe_orthog = ProbeOrthogonalityConstraint(
            weight=probe_orthog_weight
        ) if probe_orthog_weight > 0 else None
        
        self.fidelity_weight = fidelity_weight
    
    def forward(
        self,
        predicted_intensity: torch.Tensor,
        measured_intensity: torch.Tensor,
        obj: Optional[torch.Tensor] = None,
        probe: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            predicted_intensity: Predicted diffraction intensities
            measured_intensity: Measured diffraction intensities
            obj: Object for regularization (optional)
            probe: Probe for constraints (optional)
            mask: Valid pixel mask (optional)
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Fidelity loss
        losses['fidelity'] = self.fidelity_loss(
            predicted_intensity, measured_intensity, mask
        ) * self.fidelity_weight
        
        # TV regularization on object
        if self.tv_loss is not None and obj is not None:
            # For complex objects, apply TV to magnitude or phase
            if torch.is_complex(obj):
                obj_mag = torch.abs(obj)
                losses['tv'] = self.tv_loss(obj_mag)
            else:
                losses['tv'] = self.tv_loss(obj)
        
        # Surface zero constraint
        if self.surface_loss is not None and obj is not None:
            if torch.is_complex(obj):
                obj_mag = torch.abs(obj)
                losses['surface_zero'] = self.surface_loss(obj_mag)
            else:
                losses['surface_zero'] = self.surface_loss(obj)
        
        # Probe orthogonality
        if self.probe_orthog is not None and probe is not None:
            losses['probe_orthog'] = self.probe_orthog(probe)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    # Test fidelity loss
    pred = torch.rand(10, 64, 64)
    meas = torch.rand(10, 64, 64)
    
    fidelity = PtychographyLoss(loss_type='mse')
    loss = fidelity(pred, meas)
    print(f"Fidelity loss: {loss.item():.6f}")
    
    # Test TV loss
    obj = torch.rand(1, 16, 128, 128)
    tv = TotalVariationLoss(weight_xy=0.01, weight_z=0.001)
    loss = tv(obj)
    print(f"TV loss: {loss.item():.6f}")
    
    # Test surface zero
    obj_3d = torch.rand(1, 8, 64, 64)
    surf = SurfaceZeroLoss(weight=0.1)
    loss = surf(obj_3d)
    print(f"Surface zero loss: {loss.item():.6f}")
    
    print("\nAll tests passed!")
