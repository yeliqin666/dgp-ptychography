"""
Utility Functions

Helper functions for data processing, visualization, and analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, Dict
from skimage.metrics import structural_similarity as ssim


def complex_to_rgb(
    complex_array: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    phase_cmap: str = 'hsv'
) -> np.ndarray:
    """
    Convert complex array to RGB image.
    
    Magnitude maps to brightness, phase maps to hue.
    
    Args:
        complex_array: Complex-valued array
        vmin: Minimum magnitude for scaling
        vmax: Maximum magnitude for scaling
        phase_cmap: Colormap for phase
    
    Returns:
        RGB image array
    """
    magnitude = np.abs(complex_array)
    phase = np.angle(complex_array)
    
    # Normalize magnitude
    if vmin is None:
        vmin = magnitude.min()
    if vmax is None:
        vmax = magnitude.max()
    
    magnitude_norm = np.clip((magnitude - vmin) / (vmax - vmin + 1e-10), 0, 1)
    
    # Normalize phase to [0, 1]
    phase_norm = (phase + np.pi) / (2 * np.pi)
    
    # Create RGB
    from matplotlib import cm
    cmap = cm.get_cmap(phase_cmap)
    
    # Get colors from phase
    phase_colors = cmap(phase_norm)[:, :, :3]
    
    # Modulate by magnitude
    rgb = phase_colors * magnitude_norm[:, :, np.newaxis]
    
    return rgb


def plot_complex(
    complex_array: np.ndarray,
    title: str = "Complex Field",
    figsize: Tuple[int, int] = (12, 4),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Plot magnitude and phase of complex array.
    
    Args:
        complex_array: Complex-valued 2D array
        title: Figure title
        figsize: Figure size
        cmap: Colormap
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    magnitude = np.abs(complex_array)
    phase = np.angle(complex_array)
    
    # Magnitude
    im0 = axes[0].imshow(magnitude, cmap=cmap)
    axes[0].set_title(f"{title} - Magnitude")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Phase
    im1 = axes[1].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"{title} - Phase")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Complex RGB
    rgb = complex_to_rgb(complex_array)
    axes[2].imshow(rgb)
    axes[2].set_title(f"{title} - Complex RGB")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig


def plot_reconstruction_comparison(
    conventional: Dict,
    dgp: Dict,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Compare conventional and DGP reconstructions.
    
    Args:
        conventional: Results from conventional reconstruction
        dgp: Results from DGP reconstruction
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Conventional object
    ax1 = fig.add_subplot(gs[0, 0])
    obj_conv = conventional['object'][0] if conventional['object'].ndim == 3 else conventional['object']
    im1 = ax1.imshow(np.abs(obj_conv), cmap='viridis')
    ax1.set_title("Conventional - Object Magnitude")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(np.angle(obj_conv), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax2.set_title("Conventional - Object Phase")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # DGP object
    ax3 = fig.add_subplot(gs[0, 2])
    obj_dgp = dgp['object'][0] if dgp['object'].ndim == 3 else dgp['object']
    im3 = ax3.imshow(np.abs(obj_dgp), cmap='viridis')
    ax3.set_title("DGP - Object Magnitude")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(np.angle(obj_dgp), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax4.set_title("DGP - Object Phase")
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Probes
    ax5 = fig.add_subplot(gs[1, 0])
    probe_conv = conventional['probe'][0] if conventional['probe'].ndim == 3 else conventional['probe']
    im5 = ax5.imshow(np.abs(probe_conv), cmap='viridis')
    ax5.set_title("Conventional - Probe")
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    probe_dgp = dgp['probe'][0] if dgp['probe'].ndim == 3 else dgp['probe']
    im6 = ax6.imshow(np.abs(probe_dgp), cmap='viridis')
    ax6.set_title("DGP - Probe")
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Loss curves
    ax7 = fig.add_subplot(gs[1, 2:])
    if 'loss_history' in conventional:
        ax7.plot(conventional['loss_history'], label='Conventional', linewidth=2)
    if 'history' in dgp and 'stage3' in dgp['history']:
        offset = len(conventional.get('loss_history', []))
        x = np.arange(len(dgp['history']['stage3'])) + offset
        ax7.plot(x, dgp['history']['stage3'], label='DGP (Stage 3)', linewidth=2)
    
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Loss')
    ax7.set_title('Convergence Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')
    
    return fig


def calculate_fft_power_spectrum(
    image: np.ndarray,
    pixel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate radially averaged FFT power spectrum.
    
    Args:
        image: 2D image array
        pixel_size: Pixel size in Angstroms
    
    Returns:
        spatial_freq: Spatial frequencies (1/Angstroms)
        power: Radially averaged power
    """
    # FFT
    fft_image = np.fft.fft2(image)
    fft_image = np.fft.fftshift(fft_image)
    power_2d = np.abs(fft_image)**2
    
    # Frequency coordinates
    ny, nx = image.shape
    fy = np.fft.fftfreq(ny, d=pixel_size)
    fx = np.fft.fftfreq(nx, d=pixel_size)
    fy = np.fft.fftshift(fy)
    fx = np.fft.fftshift(fx)
    
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    freq_mag = np.sqrt(FX**2 + FY**2)
    
    # Radial average
    max_freq = freq_mag.max()
    n_bins = min(ny, nx) // 2
    bins = np.linspace(0, max_freq, n_bins)
    
    power_radial = np.zeros(n_bins - 1)
    freq_radial = (bins[:-1] + bins[1:]) / 2
    
    for i in range(n_bins - 1):
        mask = (freq_mag >= bins[i]) & (freq_mag < bins[i+1])
        if mask.sum() > 0:
            power_radial[i] = power_2d[mask].mean()
    
    return freq_radial, power_radial


def estimate_information_limit(
    power_spectrum: np.ndarray,
    spatial_freq: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Estimate information limit from power spectrum.
    
    Args:
        power_spectrum: Radially averaged power spectrum
        spatial_freq: Corresponding spatial frequencies (1/Angstroms)
        threshold: Threshold fraction of maximum power
    
    Returns:
        Information limit in Angstroms
    """
    # Normalize
    power_norm = power_spectrum / power_spectrum.max()
    
    # Find where power drops below threshold
    above_threshold = power_norm > threshold
    
    if not above_threshold.any():
        return np.inf
    
    # Find last frequency above threshold
    last_idx = np.where(above_threshold)[0][-1]
    
    if last_idx >= len(spatial_freq) - 1:
        return 1.0 / spatial_freq[-1]
    
    # Information limit in Angstroms
    info_limit = 1.0 / spatial_freq[last_idx]
    
    return info_limit


def calculate_ssim_score(
    image1: np.ndarray,
    image2: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Calculate structural similarity index (SSIM).
    
    Args:
        image1: First image
        image2: Second image
        data_range: Data range for images
    
    Returns:
        SSIM score
    """
    if np.iscomplexobj(image1):
        image1 = np.abs(image1)
    if np.iscomplexobj(image2):
        image2 = np.abs(image2)
    
    if data_range is None:
        data_range = max(image1.max(), image2.max()) - min(image1.min(), image2.min())
    
    return ssim(image1, image2, data_range=data_range)


def add_noise(
    data: np.ndarray,
    noise_type: str = 'poisson',
    snr: Optional[float] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add noise to data.
    
    Args:
        data: Input data
        noise_type: 'poisson', 'gaussian', or 'both'
        snr: Signal-to-noise ratio (for Gaussian)
        seed: Random seed
    
    Returns:
        Noisy data
    """
    if seed is not None:
        np.random.seed(seed)
    
    if noise_type == 'poisson':
        # Poisson noise
        noisy = np.random.poisson(data).astype(float)
    
    elif noise_type == 'gaussian':
        # Gaussian noise
        if snr is None:
            snr = 20  # Default SNR in dB
        
        signal_power = np.mean(data**2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        
        noise = np.random.normal(0, noise_std, data.shape)
        noisy = data + noise
        noisy = np.maximum(noisy, 0)  # Ensure non-negative
    
    elif noise_type == 'both':
        # Poisson then Gaussian
        noisy = np.random.poisson(data).astype(float)
        
        signal_power = np.mean(noisy**2)
        noise_power = signal_power / (10 ** ((snr or 20) / 10))
        noise_std = np.sqrt(noise_power)
        
        noise = np.random.normal(0, noise_std, noisy.shape)
        noisy = noisy + noise
        noisy = np.maximum(noisy, 0)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy


def save_results(
    results: Dict,
    filename: str,
    compress: bool = True
):
    """
    Save reconstruction results to file.
    
    Args:
        results: Results dictionary
        filename: Output filename (.npz)
        compress: Use compression
    """
    save_dict = {}
    
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, dict):
            # Flatten nested dicts
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (list, np.ndarray)):
                    save_dict[f"{key}_{subkey}"] = np.array(subvalue)
    
    if compress:
        np.savez_compressed(filename, **save_dict)
    else:
        np.savez(filename, **save_dict)


def load_results(filename: str) -> Dict:
    """
    Load reconstruction results from file.
    
    Args:
        filename: Input filename (.npz)
    
    Returns:
        Results dictionary
    """
    data = np.load(filename, allow_pickle=True)
    
    results = {}
    for key in data.files:
        results[key] = data[key]
    
    return results
