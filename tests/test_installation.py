"""
Test Installation

Quick test to verify DGP-Ptycho installation and basic functionality.
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        import torch
        print("  ✓ NumPy and PyTorch")
    except ImportError as e:
        print(f"  ✗ Failed to import NumPy/PyTorch: {e}")
        return False
    
    try:
        from dgp_ptycho import (
            DGPPtychographyReconstructor,
            ConventionalReconstructor,
            create_dgp,
            electron_wavelength
        )
        print("  ✓ DGP-Ptycho package")
    except ImportError as e:
        print(f"  ✗ Failed to import DGP-Ptycho: {e}")
        return False
    
    return True


def test_forward_model():
    """Test the forward model."""
    print("\nTesting forward model...")
    
    try:
        import torch
        from dgp_ptycho.forward_model import MultisliceForwardModel, electron_wavelength
        
        # Create small forward model
        wavelength = electron_wavelength(300e3)
        
        model = MultisliceForwardModel(
            probe_shape=(32, 32),
            object_shape=(1, 64, 64),
            pixel_size=0.1,
            wavelength=wavelength,
            slice_thickness=1.0,
            device='cpu'
        )
        
        # Test forward pass
        probe = torch.randn(1, 32, 32, dtype=torch.complex64)
        obj = torch.exp(1j * torch.randn(1, 64, 64))
        positions = torch.tensor([[16.0, 16.0]])
        
        output = model(probe, obj, positions)
        
        assert output.shape == (1, 32, 32), f"Expected shape (1, 32, 32), got {output.shape}"
        print("  ✓ Forward model works")
        return True
        
    except Exception as e:
        print(f"  ✗ Forward model test failed: {e}")
        return False


def test_dgp_model():
    """Test DGP model creation."""
    print("\nTesting DGP models...")
    
    try:
        import torch
        from dgp_ptycho.models import create_dgp, count_parameters
        
        # Create DGP
        dgp = create_dgp(
            in_channels=2,
            out_channels=2,
            num_layers=3,
            start_filters=16,
            output_complex=False
        )
        
        n_params = count_parameters(dgp)
        print(f"    DGP has {n_params:,} parameters")
        
        # Test forward pass
        x = torch.randn(1, 2, 64, 64)
        y = dgp.dgp(x)
        
        assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
        print("  ✓ DGP model works")
        return True
        
    except Exception as e:
        print(f"  ✗ DGP model test failed: {e}")
        return False


def test_simulator():
    """Test data simulator."""
    print("\nTesting data simulator...")
    
    try:
        from dgp_ptycho.simulator import create_test_dataset
        
        dataset = create_test_dataset(
            object_type='atoms',
            scan_shape=(4, 4),
            probe_shape=(32, 32),
            object_size=(64, 64),
            seed=42
        )
        
        assert 'object' in dataset
        assert 'probe' in dataset
        assert 'intensities' in dataset
        assert 'positions' in dataset
        
        print(f"    Object: {dataset['object'].shape}")
        print(f"    Probe: {dataset['probe'].shape}")
        print(f"    Intensities: {dataset['intensities'].shape}")
        print("  ✓ Simulator works")
        return True
        
    except Exception as e:
        print(f"  ✗ Simulator test failed: {e}")
        return False


def test_small_reconstruction():
    """Test a very small reconstruction (few iterations)."""
    print("\nTesting small reconstruction...")
    
    try:
        from dgp_ptycho import DGPPtychographyReconstructor
        from dgp_ptycho.simulator import create_test_dataset
        
        # Create tiny dataset
        dataset = create_test_dataset(
            object_type='gaussian',
            scan_shape=(3, 3),
            probe_shape=(32, 32),
            object_size=(50, 50),
            seed=42
        )
        
        # Quick reconstruction
        reconstructor = DGPPtychographyReconstructor(
            measured_intensities=dataset['intensities'],
            scan_positions=dataset['positions'],
            pixel_size=dataset['pixel_size'],
            energy=dataset['energy'],
            device='cpu',
            verbose=False
        )
        
        results = reconstructor.reconstruct(
            stage1_iterations=3,
            stage2_iterations=3,
            stage3_iterations=3
        )
        
        assert 'object' in results
        assert 'probe' in results
        print("  ✓ Reconstruction works")
        return True
        
    except Exception as e:
        print(f"  ✗ Reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("DGP-Ptycho Installation Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Forward Model", test_forward_model),
        ("DGP Model", test_dgp_model),
        ("Simulator", test_simulator),
        ("Small Reconstruction", test_small_reconstruction),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nUnexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Installation successful.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
