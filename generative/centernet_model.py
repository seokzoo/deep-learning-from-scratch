import math
import torch
import torch.nn as nn

try:
    import torchvision.ops as ops
    HAS_TORCHVISION_OPS = True
except ImportError:
    HAS_TORCHVISION_OPS = False

def fill_up_weights(up):
    """
    Initializes the nn.ConvTranspose2d layer to perform bilinear upsampling.
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformableConvBlock(nn.Module):
    """
    Deformable Convolution block using torchvision.ops.DeformConv2d.
    Falls back to standard Conv2d if torchvision.ops is not available.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConvBlock, self).__init__()
        self.use_dcn = HAS_TORCHVISION_OPS
        
        if self.use_dcn:
            self.offset_conv = nn.Conv2d(
                in_channels,
                2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.deform_conv = ops.DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            # Initialize offset conv weights to zero so it behaves like standard conv initially
            nn.init.constant_(self.offset_conv.weight, 0.0)
            nn.init.constant_(self.offset_conv.bias, 0.0)
        else:
            self.deform_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.use_dcn:
            offset = self.offset_conv(x)
            x = self.deform_conv(x, offset)
        else:
            x = self.deform_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CenterNetNeck(nn.Module):
    """
    CenterNet Neck (Upsampling Decoder).
    Takes a backbone output feature map (usually downsampled by 32x)
    and upsamples it by 8x using three deconvolution layers (to stride 4).
    """
    def __init__(self, in_channels, deconv_filters=[256, 128, 64], deconv_kernels=[4, 4, 4], use_dcn=False):
        super(CenterNetNeck, self).__init__()
        self.in_channels = in_channels
        self.deconv_filters = deconv_filters
        self.deconv_kernels = deconv_kernels
        self.use_dcn = use_dcn
        
        self.inplanes = in_channels
        self.deconv_layers = self._make_deconv_layer(
            len(deconv_filters),
            deconv_filters,
            deconv_kernels
        )
        
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Unsupported deconv kernel size: {deconv_kernel}")
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters)
        assert num_layers == len(num_kernels)
        
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            
            # Optional Deformable Convolution before Deconv layer
            if self.use_dcn:
                layers.append(
                    DeformableConvBlock(
                        in_channels=self.inplanes,
                        out_channels=self.inplanes,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
            
            # Deconvolution Layer
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
            
        return nn.Sequential(*layers)
        
    def init_weights(self):
        for m in self.deconv_layers:
            if isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.deconv_layers(x)


class CenterNetHead(nn.Module):
    """
    CenterNet Task-specific Heads.
    Computes output predictions for heatmaps, bounding box size (w, h), and local offset.
    """
    def __init__(self, in_channels=64, heads={'hm': 80, 'wh': 2, 'reg': 2}, head_conv=64):
        super(CenterNetHead, self).__init__()
        self.heads = heads
        
        for head in self.heads:
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0, bias=True)
                )
            else:
                fc = nn.Conv2d(in_channels, num_output, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module(head, fc)
            
    def init_weights(self):
        for head in self.heads:
            m = getattr(self, head)
            for sub_m in m.modules():
                if isinstance(sub_m, nn.Conv2d):
                    # Check if it is the final projection Conv2d layer
                    if sub_m.out_channels == self.heads[head]:
                        if 'hm' in head:
                            # Prior probability initialization for focal loss
                            nn.init.constant_(sub_m.bias, -2.19)
                        else:
                            nn.init.normal_(sub_m.weight, std=0.001)
                            nn.init.constant_(sub_m.bias, 0.0)
                    else:
                        # Intermediate Conv2d layers
                        nn.init.normal_(sub_m.weight, std=0.001)
                        nn.init.constant_(sub_m.bias, 0.0)
                        
    def forward(self, x):
        outputs = {}
        for head in self.heads:
            outputs[head] = getattr(self, head)(x)
        return outputs


class CenterNet(nn.Module):
    """
    Combined CenterNet Module.
    Integrates a feature extraction backbone, CenterNetNeck upsampler, and task-specific heads.
    """
    def __init__(self, backbone, backbone_out_channels, num_classes=80, deconv_filters=[256, 128, 64], deconv_kernels=[4, 4, 4], head_conv=64, use_dcn=False):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        self.neck = CenterNetNeck(
            in_channels=backbone_out_channels,
            deconv_filters=deconv_filters,
            deconv_kernels=deconv_kernels,
            use_dcn=use_dcn
        )
        self.heads = {
            'hm': num_classes,
            'wh': 2,
            'reg': 2
        }
        self.head = CenterNetHead(
            in_channels=deconv_filters[-1],
            heads=self.heads,
            head_conv=head_conv
        )
        self.init_weights()
        
    def init_weights(self):
        self.neck.init_weights()
        self.head.init_weights()
        
    def forward(self, x):
        # Extracts backbone features (downsampled by 32x)
        feat = self.backbone(x)
        # Upsamples features by 8x to output stride 4
        feat = self.neck(feat)
        # Predicts head maps
        outputs = self.head(feat)
        return outputs


# =====================================================================
# Decoding and Post-processing Utilities
# =====================================================================

def _nms(heat, kernel=3):
    """
    Applies Non-Maximum Suppression (NMS) on heatmap using max-pooling.
    """
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    """
    Extracts features at specified spatial indices.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes features from [B, C, H, W] to [B, H*W, C] and gathers values at indices.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    """
    Extracts top K peak detection scores and coordinates from category heatmaps.
    """
    batch, cat, height, width = scores.size()
    
    # Get top K scores and indices for each class separately
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode='trunc').float()
    topk_xs = (topk_inds % width).float()
      
    # Get top K scores and indices across all classes combined
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.div(topk_ind, K, rounding_mode='trunc').int()
    
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    """
    Decodes CenterNet raw predictions into final bounding box detections.
    
    Args:
        heat (Tensor): Heatmap tensor of shape [B, C, H, W] (e.g. keypoint centers).
        wh (Tensor): Size tensor of shape [B, 2, H, W] (width, height).
        reg (Tensor, optional): Local offset tensor of shape [B, 2, H, W].
        cat_spec_wh (bool): Whether width/height are class-specific.
        K (int): Number of top detections to keep.
        
    Returns:
        Tensor: Detections tensor of shape [B, K, 6] in the format [x1, y1, x2, y2, score, class].
    """
    batch, cat, height, width = heat.size()
    
    # 1. Perform max-pooling NMS to find peak center points
    heat = _nms(heat)
    
    # 2. Retrieve top K predictions
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    
    # 3. Add sub-pixel offset refinement if available
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
        
    # 4. Extract width and height at object centers
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
        
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    
    # 5. Build final bounding boxes [x1, y1, x2, y2] in stride-4 space
    bboxes = torch.cat([
        xs - wh[..., 0:1] / 2,
        ys - wh[..., 1:2] / 2,
        xs + wh[..., 0:1] / 2,
        ys + wh[..., 1:2] / 2
    ], dim=2)
    
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


# =====================================================================
# Backbone Builder Utility
# =====================================================================

def get_resnet_backbone(model_name='resnet18', pretrained=False):
    """
    Builds a fully convolutional backbone from a torchvision ResNet model
    by stripping the final global average pooling and FC classification layers.
    Returns the backbone module and its output channel dimension.
    """
    import torchvision.models as models
    
    if hasattr(models, model_name):
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = getattr(models, model_name)(weights=weights)
        except AttributeError:
            resnet = getattr(models, model_name)(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")
        
    backbone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4
    )
    
    if model_name in ['resnet18', 'resnet34']:
        out_channels = 512
    else:
        out_channels = 2048
        
    return backbone, out_channels
