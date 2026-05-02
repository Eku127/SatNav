"""Visual encoder for VLN models.

This module implements ResNet-based visual encoders using torchvision's
pretrained models. Supports RGB satellite imagery encoding.

Note: SatNav uses satellite overhead imagery, so there is NO depth encoder
(unlike VLN-CE which uses both RGB and depth for indoor navigation).

Reference:
    - VLN-CE: vlnce_baselines/models/encoders/resnet_encoders.py
    - TorchVisionResNet50 class (lines 118-220)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

# Try to import weights enum for newer torchvision versions
try:
    from torchvision.models import ResNet50_Weights, ResNet18_Weights
    TORCHVISION_HAS_WEIGHTS = True
except ImportError:
    TORCHVISION_HAS_WEIGHTS = False


class TorchVisionResNet50(nn.Module):
    """ResNet-50 visual encoder using torchvision's pretrained model.
    
    This encoder uses a ResNet-50 pretrained on ImageNet to extract
    visual features from RGB images. The final fully-connected layer
    is replaced with a new layer of specified output size.
    
    Features:
    - Pretrained on ImageNet (automatic download)
    - Optional normalization with ImageNet statistics
    - Trainable or frozen backbone
    - Spatial output support (for attention mechanisms)
    
    Args:
        output_size (int): Size of the output feature vector (default: 256)
        normalize_visual_inputs (bool): Whether to normalize with ImageNet
            statistics (default: False)
        trainable (bool): Whether to allow gradient updates (default: False)
        spatial_output (bool): Whether to return spatial features for
            attention mechanisms (default: False)
        single_spatial_filter (bool): For spatial output mode (default: True)
    
    Reference:
        VLN-CE config (vlnce_baselines/config/default.py lines 239-242):
            cnn_type: TorchVisionResNet50
            output_size: 256
            trainable: False
            normalize_visual_inputs: False (not used in VLN-CE for RGB)
    """
    
    def __init__(
        self,
        output_size: int,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
        single_spatial_filter: bool = True,
        pretrained: bool = True,
    ):
        """Initialize the ResNet-50 encoder.
        
        Args:
            output_size: Size of output features
            normalize_visual_inputs: Whether to apply ImageNet normalization
            trainable: Whether the backbone is trainable
            spatial_output: Whether to return spatial feature maps
            single_spatial_filter: Whether to use single spatial filter
            pretrained: Whether to load ImageNet weights from TorchVision cache
        """
        super().__init__()
        
        self.normalize_visual_inputs = normalize_visual_inputs
        self.spatial_output = spatial_output
        
        # Load ResNet-50 from torchvision. Quickstart configs can disable
        # pretrained weights to avoid network downloads in fresh environments.
        if TORCHVISION_HAS_WEIGHTS:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        modules = list(resnet.children())
        self.resnet_layer_size = modules[-1].in_features
        self.cnn = nn.Sequential(*modules[:-1])
        
        # Freeze/unfreeze parameters based on trainable flag
        for param in self.cnn.parameters():
            param.requires_grad_(trainable)
        self.cnn.train(trainable)
        
        if not self.spatial_output:
            # Standard mode: output is a feature vector
            self.output_shape = (output_size,)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.resnet_layer_size, output_size),
                nn.ReLU(),
            )
        else:
            # Spatial mode: output is a feature map (for attention)
            
            class SpatialAvgPool(nn.Module):
                """Custom spatial average pooling to 4x4 grid."""
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))
                    return x
            
            if single_spatial_filter:
                self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            
            self.cnn.avgpool = SpatialAvgPool()
            self.spatial_embeddings = nn.Embedding(4 * 4, 64)
            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )
    
    @property
    def is_blind(self):
        """Whether the encoder is blind (always False for visual encoder)."""
        return False
    
    def forward(self, observations: Dict) -> Tensor:
        """Encode RGB visual observations.
        
        Args:
            observations: Dictionary containing:
                - "rgb": Tensor of shape [batch_size, height, width, 3]
                  with pixel values in range [0, 255]
                - "rgb_features": (optional) Precomputed features
        
        Returns:
            Visual features:
                - If spatial_output=False: [batch_size, output_size]
                - If spatial_output=True: [batch_size, feature_dim, 4, 4]
        
        Note:
            Input images are expected in HWC format with values [0, 255].
            They are converted to CHW format and normalized to [0, 1].
            Optionally, ImageNet normalization is applied.
        
        Reference:
            VLN-CE: vlnce_baselines/models/encoders/resnet_encoders.py
                    TorchVisionResNet.forward() (lines 170-220)
        """
        def normalize(imgs: Tensor) -> Tensor:
            """Normalize images.
            
            Steps:
            1. Scale pixel values from [0, 255] to [0, 1]
            2. (Optional) Apply ImageNet normalization:
               - Mean: [0.485, 0.456, 0.406]
               - Std: [0.229, 0.224, 0.225]
            
            Args:
                imgs: Images with pixel values [0, 255], shape [B, 3, H, W]
            
            Returns:
                Normalized images
            
            Reference:
                ImageNet normalization is standard for torchvision models.
                Mean and std computed on ImageNet training set.
            """
            imgs = imgs.contiguous() / 255.0
            
            if self.normalize_visual_inputs:
                mean_norm = torch.tensor([0.485, 0.456, 0.406]).to(
                    device=imgs.device
                )[None, :, None, None]
                std_norm = torch.tensor([0.229, 0.224, 0.225]).to(
                    device=imgs.device
                )[None, :, None, None]
                return imgs.sub(mean_norm).div(std_norm)
            else:
                return imgs
        
        # Check if precomputed features are available
        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # Permute from HWC to CHW format
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            # Normalize and pass through ResNet
            resnet_output = self.cnn(normalize(rgb_observations))
        
        if self.spatial_output:
            # Spatial output mode: add spatial embeddings
            b, c, h, w = resnet_output.size()
            
            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )
            
            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            # Standard output mode: feature vector
            return self.fc(resnet_output)


class TorchVisionResNet18(TorchVisionResNet50):
    """ResNet-18 visual encoder (lighter version).
    
    Identical to TorchVisionResNet50 but uses ResNet-18 architecture.
    Useful for faster training and inference with slightly lower accuracy.
    
    Reference:
        VLN-CE: vlnce_baselines/models/encoders/resnet_encoders.py (lines 227-229)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize ResNet-18 encoder.
        
        Note: This overrides the parent's __init__ to use resnet18 instead of resnet50.
        """
        # We need to override the parent's __init__ properly
        nn.Module.__init__(self)  # Skip parent's __init__
        
        self.normalize_visual_inputs = kwargs.get("normalize_visual_inputs", False)
        self.spatial_output = kwargs.get("spatial_output", False)
        output_size = args[0] if args else kwargs.get("output_size", 256)
        trainable = kwargs.get("trainable", False)
        single_spatial_filter = kwargs.get("single_spatial_filter", True)
        pretrained = kwargs.get("pretrained", True)
        
        # Load ResNet-18 (not ResNet-50).
        if TORCHVISION_HAS_WEIGHTS:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(pretrained=pretrained)
        
        modules = list(resnet.children())
        self.resnet_layer_size = modules[-1].in_features
        self.cnn = nn.Sequential(*modules[:-1])
        
        for param in self.cnn.parameters():
            param.requires_grad_(trainable)
        self.cnn.train(trainable)
        
        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.resnet_layer_size, output_size),
                nn.ReLU(),
            )
        else:
            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))
                    return x
            
            if single_spatial_filter:
                self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            
            self.cnn.avgpool = SpatialAvgPool()
            self.spatial_embeddings = nn.Embedding(4 * 4, 64)
            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )
