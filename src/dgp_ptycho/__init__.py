"""
DGP-Ptycho: Deep Generative Priors for Electron Ptychography

A PyTorch implementation of DGP-enabled ptychographic reconstruction.

Main components:
- DGPPtychographyReconstructor: Complete 3-stage reconstruction pipeline
- MultisliceForwardModel: Forward model for ptychography
- create_dgp: Create U-Net DGP models
- ConventionalReconstructor: Standard pixelated reconstruction

Example:
    >>> from dgp_ptycho import DGPPtychographyReconstructor
    >>> reconstructor = DGPPtychographyReconstructor(
    ...     measured_intensities=data,
    ...     scan_positions=positions,
    ...     pixel_size=0.1,
    ...     energy=300e3
    ... )
    >>> results = reconstructor.reconstruct()
"""

__version__ = "0.1.0"
__author__ = "DGP Ptycho Contributors"

from .reconstructor import DGPPtychographyReconstructor
from .conventional import ConventionalReconstructor
from .forward_model import (
    MultisliceForwardModel,
    MultislicePropagator,
    electron_wavelength,
    compute_probe_positions
)
from .models import create_dgp, UNetDGP, count_parameters
from .losses import (
    PtychographyLoss,
    TotalVariationLoss,
    SurfaceZeroLoss,
    CombinedLoss
)
from .utils import (
    complex_to_rgb,
    plot_complex,
    plot_reconstruction_comparison,
    calculate_fft_power_spectrum,
    estimate_information_limit,
    calculate_ssim_score,
    add_noise,
    save_results,
    load_results
)

__all__ = [
    # Main classes
    'DGPPtychographyReconstructor',
    'ConventionalReconstructor',
    
    # Forward model
    'MultisliceForwardModel',
    'MultislicePropagator',
    'electron_wavelength',
    'compute_probe_positions',
    
    # Models
    'create_dgp',
    'UNetDGP',
    'count_parameters',
    
    # Losses
    'PtychographyLoss',
    'TotalVariationLoss',
    'SurfaceZeroLoss',
    'CombinedLoss',
    
    # Utils
    'complex_to_rgb',
    'plot_complex',
    'plot_reconstruction_comparison',
    'calculate_fft_power_spectrum',
    'estimate_information_limit',
    'calculate_ssim_score',
    'add_noise',
    'save_results',
    'load_results',
]
