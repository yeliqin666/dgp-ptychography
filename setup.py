"""
Deep Generative Priors for Electron Ptychography

A PyTorch implementation of DGP-enabled ptychographic reconstruction
based on McCray et al., "Deep generative priors for robust and efficient 
electron ptychography" (2025).
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dgp-ptycho",
    version="0.1.0",
    author="DGP Ptycho Contributors",
    description="Deep Generative Priors for Electron Ptychography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dgp-ptycho",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
        "scikit-image>=0.18.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "jupyter>=1.0",
        ],
    },
)
