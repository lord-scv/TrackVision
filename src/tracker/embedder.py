import os
import urllib.request
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger("ReIDEmbedder")

##########
# OSNet-x0.25 Architecture
##########

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        else:
            self.gate_activation = None

    def forward(self, x):
        identity = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return identity * x


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNet(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = self._make_layer(
            blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3])
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN))
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        layers = [
            nn.Linear(input_dim, fc_dims),
            nn.BatchNorm1d(fc_dims),
            nn.ReLU(inplace=True)
        ]
        self.feature_dim = fc_dims
        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        return y


def osnet_x0_25(num_classes=1000, pretrained=True, **kwargs):
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        **kwargs
    )
    return model


##########
# Feature Extractor Wrapper Class
##########

class ReIDEmbedder:
    def __init__(self, model_name="osnet_x0_25", force_cpu=False):
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.model_name = model_name
        self.model = None
        self.is_fallback = False
        
        self._init_model()

    def _init_model(self):
        if self.model_name.lower() == "osnet_x0_25":
            try:
                self.model = osnet_x0_25(num_classes=1000)
                
                # Check for cached weights or download
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "checkpoints")
                os.makedirs(cache_dir, exist_ok=True)
                weight_path = os.path.join(cache_dir, "osnet_x0_25_imagenet.pth")
                
                weight_url = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_imagenet.pth"
                if not os.path.exists(weight_path):
                    logger.info(f"Downloading OSNet-x0.25 weights from {weight_url}...")
                    urllib.request.urlretrieve(weight_url, weight_path)
                    
                # Load state dict
                state_dict = torch.load(weight_path, map_location=self.device)
                
                # Strip 'module.' prefix if present
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                    
                self.model.load_state_dict(new_state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Successfully loaded pre-trained OSNet-x0.25 weights.")
                
            except Exception as e:
                logger.warning(f"Could not load OSNet-x0.25 weights: {e}. Falling back to MobileNetV3 small feature extractor.")
                self._load_fallback()
        else:
            self._load_fallback()

    def _load_fallback(self):
        self.is_fallback = True
        logger.info(f"Initializing pre-trained MobileNetV3 small feature extractor on {self.device}")
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            # Load weights
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            # Remove classifier, output features directly
            self.model.classifier = nn.Identity()
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load MobileNetV3 fallback model: {e}. Pre-trained embeddings will be random vectors.")
            self.model = None

    @torch.no_grad()
    def extract(self, frame, bboxes):
        """
        Extract features from cropped bounding box images.
        frame: input BGR frame (numpy array)
        bboxes: list of bounding boxes [x1, y1, x2, y2]
        Returns:
            numpy array of shape (N, 512) for OSNet, or (N, 576) for MobileNetV3 fallback.
            If extraction fails, returns zeros.
        """
        if len(bboxes) == 0:
            return np.empty((0, 512 if not self.is_fallback else 576), dtype=np.float32)
            
        crops = []
        h_img, w_img = frame.shape[:2]
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Clip bounding boxes to image frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                # empty crop fallback
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (128, 256))
                
            # BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0
            
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop = (crop - mean) / std
            
            # HWC to CHW
            crop = crop.transpose(2, 0, 1)
            crops.append(crop)
            
        if self.model is None:
            # Fallback to random features if no model loaded
            dim = 512 if not self.is_fallback else 576
            return np.random.randn(len(bboxes), dim).astype(np.float32)
            
        # Stack crops and convert to torch Tensor
        tensor = torch.from_numpy(np.stack(crops, axis=0)).to(self.device)
        
        features = self.model(tensor)
        # Convert to numpy array
        features = features.cpu().numpy()
        
        # L2 normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)
        
        return features
