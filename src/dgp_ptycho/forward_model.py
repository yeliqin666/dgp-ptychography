"""
Multislice Ptychography Forward Model

Implements the mixed-state multislice electron ptychography forward model
with automatic differentiation support.
"""

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from typing import Optional, Tuple, List


class MultislicePropagator(nn.Module):
    """
    Multislice propagation operator for electron waves.
    
    Uses the angular spectrum method for free-space propagation between
    sample slices.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
        wavelength: float,
        slice_thickness: float,
        device: str = 'cpu'
    ):
        """
        Args:
            shape: (height, width) of the array
            pixel_size: Pixel size in Angstroms
            wavelength: Electron wavelength in Angstroms
            slice_thickness: Thickness of each slice in Angstroms
            device: PyTorch device
        """
        super().__init__()
        
        self.shape = shape
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.slice_thickness = slice_thickness
        self.device = device
        
        # Precompute propagator in Fourier space
        self.register_buffer('propagator', self._make_propagator())
    
    def _make_propagator(self) -> torch.Tensor:
        """Create the Fresnel propagator in Fourier space."""
        ny, nx = self.shape
        
        # Frequency coordinates
        ky = torch.fft.fftfreq(ny, d=self.pixel_size, device=self.device)
        kx = torch.fft.fftfreq(nx, d=self.pixel_size, device=self.device)
        
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        k_squared = KX**2 + KY**2
        
        # Fresnel propagator: exp(-i * pi * lambda * z * (kx^2 + ky^2))
        phase = -np.pi * self.wavelength * self.slice_thickness * k_squared
        propagator = torch.exp(1j * phase)
        
        return propagator
    
    def forward(self, wavefield: torch.Tensor) -> torch.Tensor:
        """
        Propagate wavefield by slice_thickness.
        
        Args:
            wavefield: Complex wavefield (..., H, W)
            
        Returns:
            Propagated wavefield
        """
        # FFT
        wavefield_fft = fft.fft2(wavefield)
        
        # Apply propagator
        wavefield_fft = wavefield_fft * self.propagator
        
        # IFFT
        wavefield = fft.ifft2(wavefield_fft)
        
        return wavefield


class MultisliceForwardModel(nn.Module):
    """
    Mixed-state multislice ptychography forward model.
    
    Takes a 3D object (slices along beam direction) and mixed-state probe,
    computes diffraction patterns at each scan position.
    """
    
    def __init__(
        self,
        probe_shape: Tuple[int, int],
        object_shape: Tuple[int, int, int],  # (num_slices, height, width)
        pixel_size: float,
        wavelength: float,
        slice_thickness: float,
        device: str = 'cpu'
    ):
        """
        Args:
            probe_shape: Shape of probe (H, W)
            object_shape: Shape of object (num_slices, H, W)
            pixel_size: Real-space pixel size (Angstroms)
            wavelength: Electron wavelength (Angstroms)
            slice_thickness: Thickness of each object slice (Angstroms)
            device: PyTorch device
        """
        super().__init__()
        
        self.probe_shape = probe_shape
        self.object_shape = object_shape
        self.num_slices = object_shape[0]
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.slice_thickness = slice_thickness
        self.device = device
        
        # Create propagator if multislice
        if self.num_slices > 1:
            self.propagator = MultislicePropagator(
                shape=probe_shape,
                pixel_size=pixel_size,
                wavelength=wavelength,
                slice_thickness=slice_thickness,
                device=device
            )
        else:
            self.propagator = None
    
    def forward(
        self,
        probe: torch.Tensor,  # (num_modes, H, W) complex
        obj: torch.Tensor,    # (num_slices, H, W) complex
        positions: torch.Tensor,  # (num_positions, 2) in pixels
        batch_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward model: compute diffraction intensities.
        
        Args:
            probe: Mixed-state probe (num_modes, H_probe, W_probe)
            obj: Multislice object (num_slices, H_obj, W_obj)
            positions: Scan positions in pixels (N, 2)
            batch_indices: Optional indices for batching (N,)
            
        Returns:
            Predicted intensities (N, H_probe, W_probe)
        """
        num_positions = positions.shape[0]
        num_modes = probe.shape[0]
        
        if batch_indices is None:
            batch_indices = torch.arange(num_positions, device=self.device)
        
        # Initialize output
        intensities = torch.zeros(
            len(batch_indices), *self.probe_shape,
            dtype=torch.float32, device=self.device
        )
        
        # Process each position
        for i, pos_idx in enumerate(batch_indices):
            pos = positions[pos_idx]
            
            # Extract object patch at this position
            exit_wave = self._compute_exit_wave(probe, obj, pos)
            
            # Sum over probe modes (incoherent sum)
            intensity = torch.sum(torch.abs(exit_wave)**2, dim=0)
            intensities[i] = intensity
        
        return intensities
    
    def _compute_exit_wave(
        self,
        probe: torch.Tensor,
        obj: torch.Tensor,
        position: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute exit wave for a single scan position.
        
        Args:
            probe: (num_modes, H, W) complex
            obj: (num_slices, H_obj, W_obj) complex
            position: (2,) scan position in pixels
            
        Returns:
            Exit wave (num_modes, H, W) after propagation through all slices
        """
        num_modes = probe.shape[0]
        
        # Extract object ROI at this position
        # Convert position to integer pixel coordinates
        py, px = position.round().long()
        py, px = py.item(), px.item()
        
        probe_h, probe_w = self.probe_shape
        
        # For each slice
        exit_waves = probe.clone()  # Start with incident probe
        
        for slice_idx in range(self.num_slices):
            # Extract object transmission function for this slice
            obj_slice = obj[slice_idx, py:py+probe_h, px:px+probe_w]
            
            # Pad if necessary (near edges)
            if obj_slice.shape != (probe_h, probe_w):
                obj_slice = self._pad_to_shape(obj_slice, (probe_h, probe_w))
            
            # Apply transmission (multiply each mode)
            exit_waves = exit_waves * obj_slice.unsqueeze(0)
            
            # Propagate to next slice (if not last slice)
            if slice_idx < self.num_slices - 1 and self.propagator is not None:
                exit_waves = self.propagator(exit_waves)
        
        # FFT to get far-field diffraction pattern
        exit_waves = fft.fft2(exit_waves)
        exit_waves = fft.fftshift(exit_waves, dim=(-2, -1))
        
        return exit_waves
    
    def _pad_to_shape(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Pad tensor to target shape with zeros."""
        h, w = tensor.shape
        th, tw = target_shape
        
        if h >= th and w >= tw:
            return tensor[:th, :tw]
        
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        
        # Pad with ones (neutral transmission)
        padded = torch.nn.functional.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            mode='constant',
            value=1.0
        )
        
        return padded


def compute_probe_positions(
    scan_shape: Tuple[int, int],
    step_size: Tuple[float, float],
    pixel_size: float
) -> np.ndarray:
    """
    Generate raster scan positions.
    
    Args:
        scan_shape: (ny, nx) number of scan positions
        step_size: (dy, dx) step size in Angstroms
        pixel_size: Pixel size in Angstroms
        
    Returns:
        positions: (N, 2) array of positions in pixels
    """
    ny, nx = scan_shape
    dy, dx = step_size
    
    # Position in Angstroms
    y_positions = np.arange(ny) * dy
    x_positions = np.arange(nx) * dx
    
    # Convert to pixels
    y_positions = y_positions / pixel_size
    x_positions = x_positions / pixel_size
    
    # Create 2D grid
    Y, X = np.meshgrid(y_positions, x_positions, indexing='ij')
    
    # Flatten to list of positions
    positions = np.stack([Y.ravel(), X.ravel()], axis=1)
    
    return positions


def electron_wavelength(energy_eV: float) -> float:
    """
    Calculate relativistic electron wavelength.
    
    Args:
        energy_eV: Electron energy in eV
        
    Returns:
        Wavelength in Angstroms
    """
    # Constants
    c = 299792458  # m/s
    m0 = 9.10938356e-31  # kg
    e = 1.602176634e-19  # C
    h = 6.62607015e-34  # J⋅s
    
    # Relativistic wavelength
    E = energy_eV * e  # Joules
    lambda_m = h / np.sqrt(2 * m0 * E * (1 + E / (2 * m0 * c**2)))
    
    # Convert to Angstroms
    lambda_A = lambda_m * 1e10
    
    return lambda_A
