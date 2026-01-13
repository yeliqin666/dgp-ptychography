"""
Complete Example: DGP Ptychography Reconstruction

This script demonstrates the full three-stage DGP reconstruction pipeline:
1. Conventional pixelated reconstruction
2. DGP pre-training
3. Joint DGP optimization

Run with: python examples/complete_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dgp_ptycho import DGPPtychographyReconstructor
from dgp_ptycho.simulator import create_test_dataset
from dgp_ptycho.utils import (
    plot_complex,
    plot_reconstruction_comparison,
    calculate_fft_power_spectrum,
    estimate_information_limit,
    save_results
)


def main():
    """Run complete DGP ptychography example."""
    
    print("="*60)
    print("DGP Ptychography - Complete Example")
    print("="*60)
    
    # ========================================
    # 1. Create simulated dataset
    # ========================================
    print("\n[1/4] Creating simulated test dataset...")
    
    dataset = create_test_dataset(
        object_type='atoms',  # Options: 'phase_gradient', 'gaussian', 'circles', 'atoms'
        scan_shape=(12, 12),
        probe_shape=(64, 64),
        object_size=(200, 200),
        pixel_size=0.1,  # Angstroms
        energy=300e3,  # eV
        step_size=8.0,  # Angstroms
        dose_per_position=1e5,
        add_noise=True,
        seed=42
    )
    
    print(f"   Object shape: {dataset['object'].shape}")
    print(f"   Probe shape: {dataset['probe'].shape}")
    print(f"   Scan positions: {dataset['positions'].shape[0]}")
    print(f"   Pixel size: {dataset['pixel_size']:.3f} Å")
    print(f"   Wavelength: {dataset['wavelength']:.4f} Å")
    
    # Visualize ground truth
    fig = plot_complex(dataset['object'], title="Ground Truth Object")
    plt.savefig('ground_truth_object.png', dpi=150, bbox_inches='tight')
    print("   Saved: ground_truth_object.png")
    
    fig = plot_complex(dataset['probe'], title="Ground Truth Probe")
    plt.savefig('ground_truth_probe.png', dpi=150, bbox_inches='tight')
    print("   Saved: ground_truth_probe.png")
    
    plt.close('all')
    
    # ========================================
    # 2. Initialize reconstructor
    # ========================================
    print("\n[2/4] Initializing DGP reconstructor...")
    
    reconstructor = DGPPtychographyReconstructor(
        measured_intensities=dataset['intensities'],
        scan_positions=dataset['positions'],
        pixel_size=dataset['pixel_size'],
        energy=dataset['energy'],
        wavelength=dataset['wavelength'],
        num_slices=1,  # Single-slice for this example
        device='cuda',  # Use 'cpu' if no GPU available
        verbose=True
    )
    
    # ========================================
    # 3. Run reconstruction
    # ========================================
    print("\n[3/4] Running three-stage reconstruction...")
    print("\nThis will take a few minutes...")
    
    results = reconstructor.reconstruct(
        # Stage 1: Conventional reconstruction
        stage1_iterations=30,
        stage1_method='gradient_descent',
        
        # Stage 2: DGP pre-training
        stage2_iterations=50,
        stage2_lr=1e-3,
        
        # Stage 3: Joint optimization
        stage3_iterations=100,
        stage3_lr_obj=1e-4,
        stage3_lr_probe=1e-4,
        
        # DGP architecture
        num_layers=3,
        start_filters=16,
        
        # Regularization (optional)
        tv_weight_xy=0.0,  # Set > 0 to enable TV regularization
        tv_weight_z=0.0,
        surface_zero_weight=0.0,
        
        # Optimization
        use_adam=True,
        noise_sigma=0.025,
        batch_size=16
    )
    
    print("\n✓ Reconstruction complete!")
    
    # ========================================
    # 4. Analyze and visualize results
    # ========================================
    print("\n[4/4] Analyzing results...")
    
    # Extract reconstructions
    obj_recon = results['object']
    probe_recon = results['probe']
    
    # Remove batch/slice dimensions if present
    if obj_recon.ndim > 2:
        obj_recon = obj_recon[0] if obj_recon.shape[0] == 1 else obj_recon.sum(axis=0)
    if probe_recon.ndim > 2:
        probe_recon = probe_recon[0]
    
    # Plot reconstructed object
    fig = plot_complex(obj_recon, title="Reconstructed Object (DGP)")
    plt.savefig('reconstructed_object.png', dpi=150, bbox_inches='tight')
    print("   Saved: reconstructed_object.png")
    
    # Plot reconstructed probe
    fig = plot_complex(probe_recon, title="Reconstructed Probe (DGP)")
    plt.savefig('reconstructed_probe.png', dpi=150, bbox_inches='tight')
    print("   Saved: reconstructed_probe.png")
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Stage 1
    if 'stage1' in results['history']:
        axes[0].plot(results['history']['stage1'], linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Stage 1: Conventional Reconstruction')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
    
    # Stage 3
    if 'stage3' in results['history']:
        axes[1].plot(results['history']['stage3'], linewidth=2, color='green')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Stage 3: Joint DGP Optimization')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
    print("   Saved: convergence.png")
    
    # Calculate information limit
    freq, power = calculate_fft_power_spectrum(
        np.abs(obj_recon),
        dataset['pixel_size']
    )
    info_limit = estimate_information_limit(power, freq)
    print(f"\n   Information limit: {info_limit:.2f} Å")
    
    # Plot power spectrum
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(1/freq, power, linewidth=2)
    ax.axvline(info_limit, color='r', linestyle='--', label=f'Info limit: {info_limit:.2f} Å')
    ax.set_xlabel('Spatial Period (Å)')
    ax.set_ylabel('Power')
    ax.set_title('FFT Power Spectrum')
    ax.set_yscale('log')
    ax.set_xlim(0, 20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('power_spectrum.png', dpi=150, bbox_inches='tight')
    print("   Saved: power_spectrum.png")
    
    # Save results
    save_results(results, 'reconstruction_results.npz')
    print("   Saved: reconstruction_results.npz")
    
    print("\n" + "="*60)
    print("Example complete! Check output files for results.")
    print("="*60)


if __name__ == "__main__":
    main()
