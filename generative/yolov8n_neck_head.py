import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # Using eps=0.001 and momentum=0.03 to match YOLOv8 default batchnorm parameters
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """Apply convolution, batch normalization and activation."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initialize a standard bottleneck module."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # If k is a tuple of tuples/ints (e.g. ((3,3), (3,3)) or (3, 3))
        k1 = k[0]
        k2 = k[1]
        self.cv1 = Conv(c1, c_, k1, 1)
        self.cv2 = Conv(c_, c2, k2, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize a CSP bottleneck with 2 convolutions."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL)."""

    def __init__(self, c1=16):
        """Initialize DFL module."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data.copy_(x.view(1, c1, 1, 1))
        self.conv.weight.requires_grad = False
        self.c1 = c1

    def forward(self, x):
        """Apply the DFL module to input tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class YOLOv8nNeck(nn.Module):
    """YOLOv8n Neck module (PANet structure for feature fusion)."""

    def __init__(self):
        super().__init__()
        # Backbone outputs: P3 (64 ch), P4 (128 ch), P5 (256 ch)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Top-down pathway
        # Concat of upsampled P5 (256) + P4 (128) -> 384 channels. Output: 128 channels
        self.c2f_p4_up = C2f(384, 128, n=1, shortcut=False)
        
        # Concat of upsampled P4_up (128) + P3 (64) -> 192 channels. Output: 64 channels
        self.c2f_p3_up = C2f(192, 64, n=1, shortcut=False)
        
        # Bottom-up pathway
        # Downsample P3_out (64) -> 64 channels
        self.conv_p3_down = Conv(64, 64, k=3, s=2)
        # Concat of downsampled P3_out (64) + P4_up (128) -> 192 channels. Output: 128 channels
        self.c2f_p4_down = C2f(192, 128, n=1, shortcut=False)
        
        # Downsample P4_out (128) -> 128 channels
        self.conv_p4_down = Conv(128, 128, k=3, s=2)
        # Concat of downsampled P4_out (128) + P5 (256) -> 384 channels. Output: 256 channels
        self.c2f_p5_down = C2f(384, 256, n=1, shortcut=False)

    def forward(self, p3, p4, p5):
        """
        Args:
            p3: Backbone features at stride 8  [B, 64, H/8, W/8]
            p4: Backbone features at stride 16 [B, 128, H/16, W/16]
            p5: Backbone features at stride 32 [B, 256, H/32, W/32]
        """
        # Top-down fusion
        p5_up = self.upsample(p5)                       # [B, 256, H/16, W/16]
        p4_concat = torch.cat([p5_up, p4], dim=1)        # [B, 384, H/16, W/16]
        p4_up = self.c2f_p4_up(p4_concat)               # [B, 128, H/16, W/16]
        
        p4_up_up = self.upsample(p4_up)                  # [B, 128, H/8, W/8]
        p3_concat = torch.cat([p4_up_up, p3], dim=1)     # [B, 192, H/8, W/8]
        p3_out = self.c2f_p3_up(p3_concat)               # [B, 64, H/8, W/8]
        
        # Bottom-up fusion
        p3_down = self.conv_p3_down(p3_out)              # [B, 64, H/16, W/16]
        p4_down_concat = torch.cat([p3_down, p4_up], dim=1) # [B, 192, H/16, W/16]
        p4_out = self.c2f_p4_down(p4_down_concat)        # [B, 128, H/16, W/16]
        
        p4_down = self.conv_p4_down(p4_out)              # [B, 128, H/32, W/32]
        p5_down_concat = torch.cat([p4_down, p5], dim=1) # [B, 384, H/32, W/32]
        p5_out = self.c2f_p5_down(p5_down_concat)        # [B, 256, H/32, W/32]
        
        return p3_out, p4_out, p5_out


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points and strides tensor based on feature map sizes."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance (ltrb) to bounding box (xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class YOLOv8nDetectHead(nn.Module):
    """Decoupled Detection Head for YOLOv8n."""

    def __init__(self, nc=80, reg_max=16, ch=(64, 128, 256), stride=(8.0, 16.0, 32.0)):
        """Initialize decoupled detection head."""
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        
        self.register_buffer("stride", torch.tensor(stride, dtype=torch.float32), persistent=False)
        
        # Bbox regression branch cv2
        c2 = max((16, ch[0] // 4, reg_max * 4))  # channels: 64
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * reg_max, 1)
            ) for x in ch
        )
        
        # Class probability branch cv3
        c3 = max(ch[0], min(nc, 100))  # channels: 80
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, nc, 1)
            ) for x in ch
        )
        
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        
        # Cache for anchor points and strides
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

    def forward(self, x):
        """
        Args:
            x: list of Neck features [p3_neck, p4_neck, p5_neck]
        """
        bs = x[0].shape[0]
        
        boxes = []
        scores = []
        for i in range(self.nl):
            reg_out = self.cv2[i](x[i])
            cls_out = self.cv3[i](x[i])
            boxes.append(reg_out.view(bs, 4 * self.reg_max, -1))
            scores.append(cls_out.view(bs, self.nc, -1))
            
        # Concatenate along the anchor dimension
        boxes = torch.cat(boxes, dim=-1)   # [B, 4 * reg_max, total_anchors]
        scores = torch.cat(scores, dim=-1) # [B, nc, total_anchors]
        
        preds = {"boxes": boxes, "scores": scores, "feats": x}
        
        if self.training:
            return preds
            
        # Inference decoding
        shape = x[0].shape  # BCHW of first feature map (P3)
        if self.shape != shape or self.anchors.numel() == 0 or self.anchors.device != x[0].device:
            anchors, strides = make_anchors(x, self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1) # [2, total_anchors]
            self.strides = strides.transpose(0, 1) # [1, total_anchors]
            self.shape = shape
            
        raw_boxes = self.dfl(boxes)  # [B, 4, total_anchors]
        dbox = dist2bbox(raw_boxes, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        scores_sigmoid = scores.sigmoid()
        out = torch.cat((dbox, scores_sigmoid), dim=1) # [B, 4 + nc, total_anchors]
        
        return out, preds


class YOLOv8nDetector(nn.Module):
    """Full YOLOv8n Model excluding the Backbone."""

    def __init__(self, nc=80, reg_max=16):
        super().__init__()
        self.neck = YOLOv8nNeck()
        self.detect = YOLOv8nDetectHead(nc=nc, reg_max=reg_max, ch=(64, 128, 256))

    def forward(self, p3, p4, p5):
        """
        Forward pass taking outputs from backbone:
            p3: stride 8 features  (e.g., [B, 64, 80, 80] for 640x640 input)
            p4: stride 16 features (e.g., [B, 128, 40, 40] for 640x640 input)
            p5: stride 32 features (e.g., [B, 256, 20, 20] for 640x640 input)
        """
        p3_neck, p4_neck, p5_neck = self.neck(p3, p4, p5)
        return self.detect([p3_neck, p4_neck, p5_neck])


def load_from_ultralytics_checkpoint(model, checkpoint_path):
    """Loads weights from an official Ultralytics YOLOv8n checkpoint into our custom model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        orig_state_dict = checkpoint["model"].state_dict()
    else:
        orig_state_dict = checkpoint

    new_state_dict = {}
    
    # Layer index mapping: original layer index -> our module prefix
    layer_mapping = {
        12: "neck.c2f_p4_up",
        15: "neck.c2f_p3_up",
        16: "neck.conv_p3_down",
        18: "neck.c2f_p4_down",
        19: "neck.conv_p4_down",
        21: "neck.c2f_p5_down",
        22: "detect",
    }
    
    for key, val in orig_state_dict.items():
        if not key.startswith("model."):
            continue
            
        parts = key.split(".")
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
            
        if layer_idx in layer_mapping:
            new_prefix = layer_mapping[layer_idx]
            suffix = ".".join(parts[2:])
            new_key = f"{new_prefix}.{suffix}"
            new_state_dict[new_key] = val
            
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights from {checkpoint_path}")
    if missing_keys:
        print(f"Missing keys (should be empty except for non-param caches): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    return missing_keys, unexpected_keys
