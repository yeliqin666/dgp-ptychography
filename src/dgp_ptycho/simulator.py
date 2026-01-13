"""
Data Simulator

Simulate ptychography data for testing and validation.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def make_test_object(
    shape: Tuple[int, int],
    object_type: str = 'phase_gradient',
    feature_size: float = 10.0
) -> np.ndarray:
    """
    Create a test object.
    
    Args:
        shape: Object shape (H, W)
        object_type: Type of object ('phase_gradient', 'gaussian', 'circles', 'atoms')
        feature_size: Characteristic feature size in pixels
    
    Returns:
        Complex-valued object (transmission function)
    """
    h, w = shape
    
    if object_type == 'phase_gradient':
        # Simple phase gradient
        y, x = np.ogrid[:h, :w]
        phase = 2 * np.pi * (x / w + y / h) / 2
        obj = np.exp(1j * phase)
    
    elif object_type == 'gaussian':
        # Gaussian phase object
        y, x = np.ogrid[:h, :w]
        y = y - h // 2
        x = x - w // 2
        
        phase = 0.5 * np.exp(-(x**2 + y**2) / (2 * feature_size**2))
        obj = np.exp(1j * phase)
    
    elif object_type == 'circles':
        # Multiple circular features
        obj = np.ones((h, w), dtype=complex)
        
        n_circles = 5
        for i in range(n_circles):
            cy = np.random.randint(h // 4, 3 * h // 4)
            cx = np.random.randint(w // 4, 3 * w // 4)
            radius = feature_size + np.random.rand() * feature_size
            
            y, x = np.ogrid[:h, :w]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            phase_shift = 0.5 + 0.3 * np.random.rand()
            obj[mask] = np.exp(1j * phase_shift)
    
    elif object_type == 'atoms':
        # Simulate atomic columns
        obj = np.ones((h, w), dtype=complex)
        
        # Lattice spacing
        spacing = int(feature_size)
        
        for y in range(spacing, h - spacing, spacing):
            for x in range(spacing, w - spacing, spacing):
                # Add some randomness
                yy = y + np.random.randint(-spacing//4, spacing//4)
                xx = x + np.random.randint(-spacing//4, spacing//4)
                
                # Gaussian peak
                Y, X = np.ogrid[:h, :w]
                gaussian = np.exp(-((X - xx)**2 + (Y - yy)**2) / (2 * (spacing/4)**2))
                
                phase_shift = 0.8 * gaussian
                obj *= np.exp(1j * phase_shift)
    
    else:
        raise ValueError(f"Unknown object type: {object_type}")
    
    return obj


def make_probe(
    shape: Tuple[int, int],
    semiangle_cutoff: float = 20e-3,
    wavelength: float = 0.0197,  # 300 keV
    pixel_size: float = 0.1,
    defocus: float = 0.0,
    astigmatism: float = 0.0
) -> np.ndarray:
    """
    Create a probe function.
    
    Args:
        shape: Probe shape (H, W)
        semiangle_cutoff: Aperture semi-angle (rad)
        wavelength: Electron wavelength (Angstroms)
        pixel_size: Pixel size (Angstroms)
        defocus: Defocus (Angstroms)
        astigmatism: Astigmatism (Angstroms)
    
    Returns:
        Complex-valued probe
    """
    h, w = shape
    
    # Frequency coordinates
    fy = np.fft.fftfreq(h, d=pixel_size)
    fx = np.fft.fftfreq(w, d=pixel_size)
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    
    # Angular coordinates
    theta = wavelength * np.sqrt(FX**2 + FY**2)
    
    # Aperture
    aperture = (theta <= semiangle_cutoff).astype(float)
    
    # Aberrations
    chi = (
        np.pi / wavelength * defocus * (FX**2 + FY**2)
        + np.pi / wavelength * astigmatism * (FX**2 - FY**2)
    )
    
    # Probe in Fourier space
    probe_fft = aperture * np.exp(-1j * chi)
    
    # Transform to real space
    probe = np.fft.ifft2(probe_fft)
    probe = np.fft.fftshift(probe)
    
    # Normalize
    probe = probe / np.sqrt(np.sum(np.abs(probe)**2))
    
    return probe


def simulate_diffraction(
    obj: np.ndarray,
    probe: np.ndarray,
    scan_positions: np.ndarray,
    add_noise: bool = True,
    poisson_noise: bool = True,
    dose_per_position: float = 1e5,
    gaussian_noise_std: float = 10.0
) -> np.ndarray:
    """
    Simulate diffraction patterns.
    
    Args:
        obj: Object transmission function (H_obj, W_obj)
        probe: Probe function (H_probe, W_probe)
        scan_positions: Scan positions in pixels (N, 2)
        add_noise: Whether to add noise
        poisson_noise: Add Poisson noise
        dose_per_position: Total counts per position (for Poisson noise)
        gaussian_noise_std: Gaussian noise standard deviation
    
    Returns:
        Diffraction intensities (N, H_probe, W_probe)
    """
    num_positions = len(scan_positions)
    probe_h, probe_w = probe.shape
    
    intensities = np.zeros((num_positions, probe_h, probe_w))
    
    for i, pos in enumerate(scan_positions):
        py, px = int(pos[0]), int(pos[1])
        
        # Extract object patch
        obj_patch = obj[py:py+probe_h, px:px+probe_w]
        
        # Pad if necessary
        if obj_patch.shape != probe.shape:
            obj_patch = np.pad(
                obj_patch,
                [(0, max(0, probe_h - obj_patch.shape[0])),
                 (0, max(0, probe_w - obj_patch.shape[1]))],
                mode='constant',
                constant_values=1+0j
            )
        
        # Exit wave
        exit_wave = probe * obj_patch
        
        # FFT to detector
        exit_wave_fft = np.fft.fft2(exit_wave)
        exit_wave_fft = np.fft.fftshift(exit_wave_fft)
        
        # Intensity
        intensity = np.abs(exit_wave_fft)**2
        
        # Normalize to dose
        if add_noise:
            intensity = intensity / intensity.sum() * dose_per_position
        
        intensities[i] = intensity
    
    # Add noise
    if add_noise:
        if poisson_noise:
            intensities = np.random.poisson(intensities).astype(float)
        
        if gaussian_noise_std > 0:
            noise = np.random.normal(0, gaussian_noise_std, intensities.shape)
            intensities = intensities + noise
            intensities = np.maximum(intensities, 0)
    
    return intensities


def generate_scan_positions(
    scan_shape: Tuple[int, int],
    step_size: float,
    pixel_size: float,
    randomize: bool = False
) -> np.ndarray:
    """
    Generate raster scan positions.
    
    Args:
        scan_shape: (ny, nx) scan grid
        step_size: Step size (Angstroms)
        pixel_size: Pixel size (Angstroms)
        randomize: Add random jitter
    
    Returns:
        positions: (N, 2) array in pixels
    """
    ny, nx = scan_shape
    
    # Grid in Angstroms
    y_ang = np.arange(ny) * step_size
    x_ang = np.arange(nx) * step_size
    
    # Convert to pixels
    y_pix = y_ang / pixel_size
    x_pix = x_ang / pixel_size
    
    Y, X = np.meshgrid(y_pix, x_pix, indexing='ij')
    
    positions = np.stack([Y.ravel(), X.ravel()], axis=1)
    
    # Add jitter
    if randomize:
        jitter = np.random.randn(*positions.shape) * (step_size / pixel_size) * 0.1
        positions = positions + jitter
    
    return positions


def create_test_dataset(
    object_type: str = 'atoms',
    scan_shape: Tuple[int, int] = (10, 10),
    probe_shape: Tuple[int, int] = (128, 128),
    object_size: Tuple[int, int] = (256, 256),
    pixel_size: float = 0.1,
    energy: float = 300e3,
    step_size: float = 10.0,
    dose_per_position: float = 1e5,
    add_noise: bool = True,
    seed: Optional[int] = None
) -> dict:
    """
    Create a complete test dataset for ptychography.
    
    Args:
        object_type: Type of test object
        scan_shape: Scan grid shape
        probe_shape: Probe/detector shape
        object_size: Object size
        pixel_size: Pixel size (Angstroms)
        energy: Electron energy (eV)
        step_size: Scan step size (Angstroms)
        dose_per_position: Electron dose per position
        add_noise: Add Poisson and Gaussian noise
        seed: Random seed
    
    Returns:
        Dictionary with object, probe, intensities, positions, and parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Wavelength
    from .forward_model import electron_wavelength
    wavelength = electron_wavelength(energy)
    
    # Create object
    obj = make_test_object(object_size, object_type=object_type)
    
    # Create probe
    probe = make_probe(
        probe_shape,
        wavelength=wavelength,
        pixel_size=pixel_size
    )
    
    # Generate scan positions
    positions = generate_scan_positions(
        scan_shape,
        step_size=step_size,
        pixel_size=pixel_size
    )
    
    # Simulate diffraction
    intensities = simulate_diffraction(
        obj, probe, positions,
        add_noise=add_noise,
        dose_per_position=dose_per_position
    )
    
    return {
        'object': obj,
        'probe': probe,
        'intensities': intensities,
        'positions': positions,
        'pixel_size': pixel_size,
        'wavelength': wavelength,
        'energy': energy,
        'scan_shape': scan_shape,
        'probe_shape': probe_shape,
        'object_size': object_size
    }
