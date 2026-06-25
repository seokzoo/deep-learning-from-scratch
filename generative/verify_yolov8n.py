import torch
import numpy as np
from yolov8n_neck_head import YOLOv8nDetector, load_from_ultralytics_checkpoint

def verify():
    print("=== Starting YOLOv8n Neck and Head Verification ===")
    
    # 1. Load the original ultralytics model
    print("Loading original pre-trained yolov8n.pt...")
    checkpoint = torch.load("yolov8n.pt", map_location="cpu", weights_only=False)
    orig_model = checkpoint["model"]
    orig_model.float()  # Convert to float32 for high precision comparison
    orig_model.eval()
    
    # 2. Instantiate our custom model and load weights
    print("Instantiating custom YOLOv8n detector...")
    custom_model = YOLOv8nDetector(nc=80, reg_max=16)
    custom_model.float()
    
    print("Loading weights into custom model...")
    missing, unexpected = load_from_ultralytics_checkpoint(custom_model, "yolov8n.pt")
    
    # Filter missing keys to see if any actual parameters were missed
    # (self.detect.anchors, strides, shape are not parameters or buffers, so they shouldn't show up. 
    # But any conv or bn weight/bias must not be in missing keys.)
    param_missing = [k for k in missing if not any(x in k for x in ["anchors", "strides", "shape"])]
    if param_missing:
        print(f"WARNING: Some parameters were not loaded: {param_missing}")
    else:
        print("Success: All standard weights and biases successfully loaded!")
        
    custom_model.eval()
    
    # 3. Register hooks to capture backbone outputs of the original model
    backbone_outputs = {}
    
    def get_activation(name):
        def hook(model, input, output):
            # In YOLOv8, output of C2f layers (like layer 4, 6) is a tensor
            backbone_outputs[name] = output
        return hook
        
    # Register forward hooks on layers 4 (P3), 6 (P4), and 9 (P5_sppf)
    orig_model.model[4].register_forward_hook(get_activation("p3"))
    orig_model.model[6].register_forward_hook(get_activation("p4"))
    orig_model.model[9].register_forward_hook(get_activation("p5"))
    
    # 4. Run forward pass with a dummy input
    print("Running forward pass with dummy input of shape [1, 3, 640, 640]...")
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        orig_out = orig_model(x)
        
        # Verify hooks captured the outputs
        assert "p3" in backbone_outputs, "Failed to capture P3 backbone output"
        assert "p4" in backbone_outputs, "Failed to capture P4 backbone output"
        assert "p5" in backbone_outputs, "Failed to capture P5 backbone output"
        
        p3_feat = backbone_outputs["p3"]
        p4_feat = backbone_outputs["p4"]
        p5_feat = backbone_outputs["p5"]
        
        print(f"Captured backbone P3 shape: {p3_feat.shape}")
        print(f"Captured backbone P4 shape: {p4_feat.shape}")
        print(f"Captured backbone P5 shape: {p5_feat.shape}")
        
        # Run our custom model using captured backbone outputs
        custom_out, custom_preds = custom_model(p3_feat, p4_feat, p5_feat)
        
    # 5. Compare results
    # original output format: (decoded_tensor, predictions_dict)
    # where decoded_tensor has shape [1, 84, 8400]
    # predictions_dict has keys: 'boxes', 'scores', 'feats'
    orig_decoded = orig_out[0]
    orig_preds = orig_out[1]
    
    print("\n=== Comparing Outputs ===")
    
    # Comparison 1: Raw regression boxes shape [1, 64, 8400]
    boxes_diff = torch.abs(orig_preds["boxes"] - custom_preds["boxes"]).max().item()
    print(f"Max absolute difference in raw boxes prediction: {boxes_diff:.2e}")
    
    # Comparison 2: Raw classification scores shape [1, 80, 8400]
    scores_diff = torch.abs(orig_preds["scores"] - custom_preds["scores"]).max().item()
    print(f"Max absolute difference in raw scores prediction: {scores_diff:.2e}")
    
    # Comparison 3: Decoded outputs (bbox + class sigmoids) shape [1, 84, 8400]
    decoded_diff = torch.abs(orig_decoded - custom_out).max().item()
    print(f"Max absolute difference in final decoded outputs (inference): {decoded_diff:.2e}")
    
    # Comparison 4: Intermediate neck outputs (feats)
    neck_p3_diff = torch.abs(orig_preds["feats"][0] - custom_preds["feats"][0]).max().item()
    neck_p4_diff = torch.abs(orig_preds["feats"][1] - custom_preds["feats"][1]).max().item()
    neck_p5_diff = torch.abs(orig_preds["feats"][2] - custom_preds["feats"][2]).max().item()
    print(f"Max absolute difference in Neck P3 outputs: {neck_p3_diff:.2e}")
    print(f"Max absolute difference in Neck P4 outputs: {neck_p4_diff:.2e}")
    print(f"Max absolute difference in Neck P5 outputs: {neck_p5_diff:.2e}")
    
    # Strict thresholds
    threshold = 1e-4
    success = (boxes_diff < threshold and 
               scores_diff < threshold and 
               decoded_diff < threshold and
               neck_p3_diff < threshold and
               neck_p4_diff < threshold and
               neck_p5_diff < threshold)
               
    if success:
        print("\nSUCCESS: Custom implementation output matches the original model perfectly!")
    else:
        print("\nFAILURE: Significant discrepancy found between custom and original model outputs.")
        
if __name__ == "__main__":
    verify()
