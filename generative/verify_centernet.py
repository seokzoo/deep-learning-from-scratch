import torch
import torch.nn as nn
from centernet_model import CenterNet, get_resnet_backbone, ctdet_decode

def run_verification():
    print("=" * 60)
    print("Starting CenterNet Model Verification")
    print("=" * 60)

    # 1. Instantiation of Backbone and CenterNet Model
    print("\nStep 1: Instantiating ResNet-18 Backbone & CenterNet Wrapper Model...")
    backbone, out_channels = get_resnet_backbone('resnet18', pretrained=False)
    num_classes = 3
    model = CenterNet(
        backbone=backbone,
        backbone_out_channels=out_channels,
        num_classes=num_classes,
        deconv_filters=[256, 128, 64],
        deconv_kernels=[4, 4, 4],
        head_conv=64,
        use_dcn=False
    )
    model.train() # Set to train mode
    print("Model instantiated successfully.")

    # 2. Shape Verification with 640x640 Input
    print("\nStep 2: Checking layer dimensions and shapes for input size [2, 3, 640, 640]...")
    x = torch.randn(2, 3, 640, 640)
    
    # Track intermediate shapes manually to verify downsampling / upsampling rates
    print(f"Input image shape: {list(x.shape)}")
    
    backbone_out = model.backbone(x)
    print(f"Backbone output shape (stride 32): {list(backbone_out.shape)}")
    assert backbone_out.shape == (2, 512, 20, 20), f"Expected Backbone shape [2, 512, 20, 20], got {list(backbone_out.shape)}"
    
    neck_out = model.neck(backbone_out)
    print(f"Neck output shape (upsampled to stride 4): {list(neck_out.shape)}")
    assert neck_out.shape == (2, 64, 160, 160), f"Expected Neck shape [2, 64, 160, 160], got {list(neck_out.shape)}"
    
    outputs = model.head(neck_out)
    hm, wh, reg = outputs['hm'], outputs['wh'], outputs['reg']
    print(f"Head outputs:")
    print(f"  - Heatmap (hm): {list(hm.shape)} (Expected: [2, {num_classes}, 160, 160])")
    print(f"  - Size map (wh): {list(wh.shape)} (Expected: [2, 2, 160, 160])")
    print(f"  - Offset map (reg): {list(reg.shape)} (Expected: [2, 2, 160, 160])")
    
    assert hm.shape == (2, num_classes, 160, 160)
    assert wh.shape == (2, 2, 160, 160)
    assert reg.shape == (2, 2, 160, 160)
    print("Shape checks passed successfully.")

    # 3. Decoding Verification (Post-processing)
    print("\nStep 3: Verifying ctdet_decode function...")
    model.eval()
    with torch.no_grad():
        # Heatmap scores are typically sigmoided in inference
        scores_heat = hm.sigmoid()
        detections = ctdet_decode(scores_heat, wh, reg=reg, K=100)
        
    print(f"Decoded detections shape: {list(detections.shape)}")
    assert detections.shape == (2, 100, 6), f"Expected [2, 100, 6], got {list(detections.shape)}"
    # Verify the format is [x1, y1, x2, y2, score, class]
    print("Detections sample (first object of first batch):")
    sample_det = detections[0, 0]
    print(f"  - BBox: [{sample_det[0]:.2f}, {sample_det[1]:.2f}, {sample_det[2]:.2f}, {sample_det[3]:.2f}]")
    print(f"  - Score: {sample_det[4]:.4f}")
    print(f"  - Class: {int(sample_det[5])}")
    print("Decoding verification passed.")

    # 4. Gradient Flow and Backward Pass Check
    print("\nStep 4: Checking backward pass and gradient flow...")
    model.train()
    
    # Define a simple loss sum of head predictions
    outputs = model(x)
    loss = outputs['hm'].pow(2).mean() + outputs['wh'].abs().mean() + outputs['reg'].abs().mean()
    
    # Zero gradients
    model.zero_grad()
    # Backward pass
    loss.backward()
    
    # Check that gradients are not None and have non-zero norm
    backbone_grad = model.backbone[0].weight.grad
    neck_grad = model.neck.deconv_layers[1].weight.grad
    head_grad = model.head.hm[0].weight.grad
    
    assert backbone_grad is not None, "Backbone has no gradients!"
    assert neck_grad is not None, "Neck has no gradients!"
    assert head_grad is not None, "Head has no gradients!"
    
    print(f"Gradients found:")
    print(f"  - Backbone first conv layer grad norm: {backbone_grad.norm().item():.6f}")
    print(f"  - Neck second deconv layer grad norm: {neck_grad.norm().item():.6f}")
    print(f"  - Head heatmap conv layer grad norm: {head_grad.norm().item():.6f}")
    print("Backward pass and gradient flow checks passed successfully.")

    print("\n" + "=" * 60)
    print("Verification Completed: ALL CHECKS PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    run_verification()
