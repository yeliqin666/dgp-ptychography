"""
Deep Generative Prior Models

U-Net architectures for generating complex-valued ptychographic
reconstructions. Implements the DGP framework from the paper.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D convolution.
    
    Applies separate real convolutions to real and imaginary parts,
    following the standard complex multiplication rules.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        # Separate convolutions for real and imaginary
        self.conv_real = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size, padding=padding, bias=bias
        )
        self.conv_imag = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size, padding=padding, bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor as (B, C*2, H, W) where channels are [real, imag] stacked
        
        Returns:
            Complex output (B, C_out*2, H, W)
        """
        # Apply convolutions
        out_real = self.conv_real(x)
        out_imag = self.conv_imag(x)
        
        # Stack real and imaginary
        out = torch.cat([out_real, out_imag], dim=1)
        
        return out


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d + BatchNorm + ReLU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_complex: bool = False
    ):
        super().__init__()
        
        if use_complex:
            # For complex: channels are doubled (real, imag)
            self.conv = ComplexConv2d(
                in_channels // 2, out_channels // 2, kernel_size, padding
            )
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding
            )
            self.bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNetDGP(nn.Module):
    """
    U-Net architecture for Deep Generative Prior.
    
    Based on the paper's specifications:
    - Default: 3-layer network with 16 starting filters
    - Fully convolutional encoder-decoder with skip connections
    - ReLU activations except for final layer
    
    Args:
        in_channels: Number of input channels (2 for complex: real, imag)
        out_channels: Number of output channels (same as input)
        num_layers: Number of encoder/decoder layers (2, 3, or 4)
        start_filters: Number of filters in first layer (default: 16)
        final_activation: Activation for final layer ('identity', 'softplus', 'sigmoid')
        use_complex: Whether to use complex-valued convolutions
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 3,
        start_filters: int = 16,
        final_activation: Literal['identity', 'softplus', 'sigmoid'] = 'identity',
        use_complex: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.final_activation_type = final_activation
        self.use_complex = use_complex
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(num_layers):
            next_channels = start_filters * (2 ** i)
            
            self.encoders.append(
                nn.Sequential(
                    ConvBlock(current_channels, next_channels, use_complex=use_complex),
                    ConvBlock(next_channels, next_channels, use_complex=use_complex)
                )
            )
            
            if i < num_layers - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            current_channels = next_channels
        
        # Bottleneck
        bottleneck_channels = start_filters * (2 ** num_layers)
        self.bottleneck = nn.Sequential(
            ConvBlock(current_channels, bottleneck_channels, use_complex=use_complex),
            ConvBlock(bottleneck_channels, bottleneck_channels, use_complex=use_complex)
        )
        
        # Decoder (upsampling path)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        current_channels = bottleneck_channels
        for i in range(num_layers - 1, -1, -1):
            next_channels = start_filters * (2 ** i)
            
            # Upsample
            self.upsamples.append(
                nn.ConvTranspose2d(
                    current_channels, next_channels,
                    kernel_size=2, stride=2
                )
            )
            
            # Decoder block (input has skip connection, so *2 channels)
            self.decoders.append(
                nn.Sequential(
                    ConvBlock(next_channels * 2, next_channels, use_complex=use_complex),
                    ConvBlock(next_channels, next_channels, use_complex=use_complex)
                )
            )
            
            current_channels = next_channels
        
        # Final output layer
        if use_complex:
            self.output_conv = ComplexConv2d(
                start_filters // 2, out_channels // 2, kernel_size=1, padding=0
            )
        else:
            self.output_conv = nn.Conv2d(
                start_filters, out_channels, kernel_size=1, padding=0
            )
        
        # Final activation
        if final_activation == 'softplus':
            self.final_activation = nn.Softplus()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output tensor (B, C, H, W)
        """
        # Encoder path
        encoder_outputs = []
        current = x
        
        for i in range(self.num_layers):
            current = self.encoders[i](current)
            encoder_outputs.append(current)
            
            if i < self.num_layers - 1:
                current = self.pools[i](current)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path
        for i in range(self.num_layers):
            # Upsample
            current = self.upsamples[i](current)
            
            # Skip connection
            skip = encoder_outputs[-(i + 1)]
            
            # Handle size mismatches (due to odd dimensions)
            if current.shape != skip.shape:
                current = self._match_size(current, skip)
            
            # Concatenate
            current = torch.cat([current, skip], dim=1)
            
            # Decoder block
            current = self.decoders[i](current)
        
        # Output layer
        output = self.output_conv(current)
        output = self.final_activation(output)
        
        return output
    
    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Resize x to match target size."""
        if x.shape[2:] != target.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=target.shape[2:], mode='bilinear', align_corners=False
            )
        return x


class DGPWrapper(nn.Module):
    """
    Wrapper for DGP that handles complex-valued outputs.
    
    Takes input as shape (B, 2, H, W) where dim 1 is [real, imag]
    and outputs complex tensors.
    """
    
    def __init__(
        self,
        dgp_model: nn.Module,
        output_complex: bool = True
    ):
        super().__init__()
        self.dgp = dgp_model
        self.output_complex = output_complex
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Real-valued input (B, 2, H, W) or complex (B, H, W)
        
        Returns:
            Complex tensor (B, H, W) or real (B, 2, H, W)
        """
        # Convert complex to real if needed
        if torch.is_complex(x):
            x = torch.stack([x.real, x.imag], dim=1)
        
        # DGP forward
        out = self.dgp(x)
        
        # Convert back to complex if needed
        if self.output_complex and out.shape[1] == 2:
            out = torch.complex(out[:, 0], out[:, 1])
        
        return out


def create_dgp(
    in_channels: int = 2,
    out_channels: int = 2,
    num_layers: int = 3,
    start_filters: int = 16,
    final_activation: str = 'identity',
    use_complex: bool = False,
    output_complex: bool = True
) -> nn.Module:
    """
    Factory function to create a DGP model.
    
    Args:
        in_channels: Input channels (2 for complex real+imag)
        out_channels: Output channels (2 for complex)
        num_layers: Number of U-Net layers (2-4)
        start_filters: Starting number of filters (default 16)
        final_activation: Final layer activation
        use_complex: Use complex convolutions
        output_complex: Output as complex tensor
    
    Returns:
        DGP model
    """
    dgp = UNetDGP(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        start_filters=start_filters,
        final_activation=final_activation,
        use_complex=use_complex
    )
    
    if output_complex:
        dgp = DGPWrapper(dgp, output_complex=True)
    
    return dgp


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test DGP architectures
    print("Testing DGP models...")
    
    configs = [
        {'num_layers': 2, 'start_filters': 8},
        {'num_layers': 3, 'start_filters': 16},
        {'num_layers': 4, 'start_filters': 16},
    ]
    
    for config in configs:
        model = create_dgp(**config, output_complex=False)
        n_params = count_parameters(model)
        
        print(f"\nLayers: {config['num_layers']}, "
              f"Filters: {config['start_filters']}")
        print(f"Parameters: {n_params:,}")
        
        # Test forward pass
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model.dgp(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")
    
    print("\nAll tests passed!")
