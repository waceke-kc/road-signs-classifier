#!/usr/bin/env python3
"""
Fixed NanoDet Model Conversion Script
Creates a clean PyTorch model that works with standard inference code
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

class NanoDetInference(nn.Module):
    """
    Clean inference wrapper for NanoDet that works with standard PyTorch loading
    """
    
    def __init__(self, nanodet_model, input_size=(320, 320), conf_threshold=0.35, nms_threshold=0.6):
        super().__init__()
        self.model = nanodet_model
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Store these as buffers so they're saved with the model
        self.register_buffer('mean', torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([57.375, 57.12, 58.395]).view(1, 3, 1, 1))
    
    def preprocess(self, x):
        """Preprocess input image"""
        # Normalize using ImageNet stats (adjust if your model uses different normalization)
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, x):
        """Forward pass with preprocessing"""
        x = self.preprocess(x)
        return self.model(x)
    
    def detect(self, x):
        """Full detection pipeline with postprocessing"""
        # Get model predictions
        outputs = self.forward(x)
        
        # Post-process outputs (this needs to be adapted based on NanoDet output format)
        detections = self.postprocess(outputs)
        return detections
    
    def postprocess(self, outputs):
        """Post-process model outputs to get final detections"""
        # This is a placeholder - you'll need to implement based on NanoDet's actual output format
        # NanoDet outputs are typically in a specific format that needs decoding
        
        # For now, return the raw outputs
        # You'll need to adapt this based on your specific model's output structure
        return outputs

def create_yolo_compatible_model(config_path, checkpoint_path, output_path, input_size=(320, 320)):
    """
    Create a model that's compatible with standard YOLO-style inference code
    """
    
    try:
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        
        print("Loading NanoDet model...")
        
        # Load config and build model
        load_config(cfg, config_path)
        nanodet_model = build_model(cfg.model)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        nanodet_model.load_state_dict(state_dict, strict=False)
        nanodet_model.eval()
        
        print("Creating inference wrapper...")
        
        # Create the inference wrapper
        inference_model = NanoDetInference(nanodet_model, input_size)
        inference_model.eval()
        
        # Test the model
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
        
        print("Testing model...")
        with torch.no_grad():
            output = inference_model(dummy_input)
        
        print(f"Model test successful. Output shape: {output.shape if hasattr(output, 'shape') else 'Complex output'}")
        
        # Save as regular PyTorch model (not TorchScript)
        print(f"Saving model to: {output_path}")
        
        # Save the complete model
        torch.save({
            'model': inference_model.state_dict(),
            'model_class': 'NanoDetInference',
            'input_size': input_size,
            'architecture': inference_model
        }, output_path)
        
        print("✅ Model saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_state_dict_model(checkpoint_path, output_path):
    """
    Create a simple state dict model that can be loaded with torch.load()
    """
    
    try:
        print("Creating simple state dict model...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Create a simple wrapper
        model_data = {
            'model_state_dict': state_dict,
            'type': 'nanodet',
            'format_version': '1.0'
        }
        
        # Save
        torch.save(model_data, output_path)
        
        print(f"✅ Simple model saved to: {output_path}")
        print("Note: You'll need to build the model architecture separately to use this")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple conversion failed: {e}")
        return False

def create_onnx_model(config_path, checkpoint_path, output_path, input_size=(320, 320)):
    """
    Export model to ONNX format for broader compatibility
    """
    
    try:
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        
        print("Creating ONNX model...")
        
        # Load model
        load_config(cfg, config_path)
        model = build_model(cfg.model)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
        
        # Export to ONNX
        onnx_path = output_path.replace('.pt', '.onnx')
        
        print(f"Exporting to ONNX: {onnx_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model saved to: {onnx_path}")
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

def create_inference_script(model_path, output_script_path):
    """
    Create a standalone inference script that works with your model
    """
    
    inference_code = '''#!/usr/bin/env python3
"""
NanoDet Inference Script
Usage: python inference.py --model model.pt --source image.jpg
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

class NanoDetDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the converted model"""
        try:
            # Try loading as a regular PyTorch model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'architecture' in checkpoint:
                model = checkpoint['architecture']
            elif 'model' in checkpoint:
                # If it's a state dict, you'll need to recreate the architecture
                print("Model contains state dict only - you need the architecture")
                return None
            else:
                # Try loading as TorchScript
                model = torch.jit.load(model_path, map_location=self.device)
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    
    def preprocess(self, image, input_size=(320, 320)):
        """Preprocess image for inference"""
        # Resize image
        img_resized = cv2.resize(image, input_size)
        
        # Convert to RGB if needed
        if len(img_resized.shape) == 3:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_resized
        
        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def detect(self, image_path):
        """Run detection on an image"""
        if self.model is None:
            print("Model not loaded!")
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Post-process outputs
        detections = self.postprocess(outputs, image.shape)
        
        return detections
    
    def postprocess(self, outputs, original_shape):
        """Post-process model outputs"""
        # This is a placeholder - implement based on your model's output format
        print(f"Model output shape: {outputs.shape if hasattr(outputs, 'shape') else 'Complex output'}")
        
        # Return dummy detections for now
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--source', required=True, help='Path to image or video')
    parser.add_argument('--device', default='cpu', help='Device to run on')
    
    args = parser.parse_args()
    
    # Create detector
    detector = NanoDetDetector(args.model, args.device)
    
    # Run detection
    detections = detector.detect(args.source)
    
    print(f"Found {len(detections) if detections else 0} detections")

if __name__ == '__main__':
    main()
'''
    
    with open(output_script_path, 'w') as f:
        f.write(inference_code)
    
    print(f"✅ Inference script created: {output_script_path}")

def main():
    parser = argparse.ArgumentParser(description='Fixed NanoDet Model Conversion')
    parser.add_argument('--config', help='Path to NanoDet config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output', required=True, help='Output path for converted model')
    parser.add_argument('--method', choices=['yolo_compatible', 'simple', 'onnx'], 
                        default='yolo_compatible', help='Conversion method')
    parser.add_argument('--input_size', nargs=2, type=int, default=[320, 320],
                        help='Model input size [width height]')
    parser.add_argument('--create_script', action='store_true', 
                        help='Create inference script')
    
    args = parser.parse_args()
    
    print(f"Converting model using method: {args.method}")
    
    success = False
    
    if args.method == 'yolo_compatible':
        if not args.config:
            print("❌ Config file required for yolo_compatible method")
            return 1
        success = create_yolo_compatible_model(
            args.config, args.checkpoint, args.output, tuple(args.input_size)
        )
    elif args.method == 'simple':
        success = create_simple_state_dict_model(args.checkpoint, args.output)
    elif args.method == 'onnx':
        if not args.config:
            print("❌ Config file required for ONNX export")
            return 1
        success = create_onnx_model(
            args.config, args.checkpoint, args.output, tuple(args.input_size)
        )
    
    if success and args.create_script:
        script_path = args.output.replace('.pt', '_inference.py').replace('.onnx', '_inference.py')
        create_inference_script(args.output, script_path)
    
    return 0 if success else 1

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        print("Fixed NanoDet Model Conversion")
        print("\nUsage:")
        print("python fixed_conversion.py --checkpoint model.pth --output model.pt --method simple")
        print("python fixed_conversion.py --config config.yml --checkpoint model.pth --output model.pt --method yolo_compatible")
        print("python fixed_conversion.py --config config.yml --checkpoint model.pth --output model.onnx --method onnx")
        print("\nMethods:")
        print("  simple: Just extract weights (most compatible)")
        print("  yolo_compatible: Create YOLO-style model (needs config)")
        print("  onnx: Export to ONNX format (needs config)")
        sys.exit(0)
    
    sys.exit(main())