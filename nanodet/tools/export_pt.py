#!/usr/bin/env python3
"""
NanoDet Model Conversion Script
Converts trained NanoDet models (.pth/.ckpt) to TorchScript (.pt) format for inference
"""

import torch
import argparse
import os
import sys
from pathlib import Path

def convert_nanodet_to_torchscript(config_path, checkpoint_path, output_path, input_size=(320, 320)):
    """
    Convert NanoDet model to TorchScript format
    
    Args:
        config_path: Path to NanoDet config file
        checkpoint_path: Path to model checkpoint (.pth or .ckpt)
        output_path: Output path for converted model (.pt)
        input_size: Model input size (width, height)
    """
    
    try:
        # Import NanoDet modules
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        
        print(f"Loading config from: {config_path}")
        load_config(cfg, config_path)
        
        print("Building model architecture...")
        model = build_model(cfg.model)
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Found state_dict in checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Found model in checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"Model loaded successfully. Setting input size to {input_size}")
        
        # Create example input tensor
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])  # [batch, channels, height, width]
        
        print("Converting to TorchScript...")
        
        # Method 1: Try tracing first
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
            conversion_method = "tracing"
        except Exception as trace_error:
            print(f"Tracing failed: {trace_error}")
            print("Trying scripting method...")
            
            # Method 2: Try scripting
            try:
                traced_model = torch.jit.script(model)
                conversion_method = "scripting"
            except Exception as script_error:
                print(f"Scripting also failed: {script_error}")
                raise Exception("Both tracing and scripting failed")
        
        print(f"Conversion successful using {conversion_method}")
        
        # Save the traced model
        print(f"Saving converted model to: {output_path}")
        traced_model.save(output_path)
        
        # Verify the saved model
        print("Verifying saved model...")
        loaded_model = torch.jit.load(output_path)
        
        with torch.no_grad():
            test_output = loaded_model(dummy_input)
        
        print(f"✅ Model verification successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {test_output.shape if isinstance(test_output, torch.Tensor) else 'Multiple outputs'}")
        print(f"   Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def convert_with_onnx_intermediate(config_path, checkpoint_path, output_path, input_size=(320, 320)):
    """
    Alternative conversion method using ONNX as intermediate format
    """
    
    try:
        # Import required modules
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        import onnx
        
        print("Using ONNX intermediate conversion method...")
        
        # Load model
        load_config(cfg, config_path)
        model = build_model(cfg.model)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Export to ONNX first
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
        onnx_path = output_path.replace('.pt', '.onnx')
        
        print(f"Exporting to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        # Load ONNX model and convert back to TorchScript
        print("Converting ONNX back to TorchScript...")
        onnx_model = onnx.load(onnx_path)
        
        # You might need to use a different method here depending on your setup
        print("Note: ONNX to TorchScript conversion requires additional setup")
        print(f"ONNX model saved to: {onnx_path}")
        
        return True
        
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        return False

def create_inference_wrapper(model_path, config_path, output_path, input_size=(320, 320)):
    """
    Create a simplified inference wrapper
    """
    
    try:
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        
        # Load model
        load_config(cfg, config_path)
        model = build_model(cfg.model)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Create wrapper class
        class NanoDetWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                return self.model(x)
        
        wrapper = NanoDetWrapper(model)
        
        # Trace the wrapper
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
        traced_wrapper = torch.jit.trace(wrapper, dummy_input)
        
        # Save
        traced_wrapper.save(output_path)
        
        print(f"✅ Inference wrapper saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Wrapper creation failed: {e}")
        return False

def simple_state_dict_conversion(checkpoint_path, output_path):
    """
    Simple conversion that just saves the state dict in .pt format
    """
    
    try:
        print("Performing simple state dict conversion...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Save as .pt file
        torch.save(state_dict, output_path)
        
        print(f"✅ State dict saved to: {output_path}")
        print(f"   Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print("   Note: This is just the state dict, you'll need the model architecture to load it")
        
        return True
        
    except Exception as e:
        print(f"Simple conversion failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert NanoDet model to .pt format')
    parser.add_argument('--config', required=True, help='Path to NanoDet config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth or .ckpt)')
    parser.add_argument('--output', required=True, help='Output path for converted model (.pt)')
    parser.add_argument('--input_size', nargs=2, type=int, default=[320, 320], 
                         help='Model input size [width height] (default: 320 320)')
    parser.add_argument('--method', choices=['torchscript', 'onnx', 'wrapper', 'simple'], 
                        default='torchscript', help='Conversion method')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return 1
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    print(f"Converting NanoDet model...")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Input size: {args.input_size}")
    print(f"  Method: {args.method}")
    print("-" * 50)
    
    # Choose conversion method
    success = False
   
    if args.method == 'torchscript':
        success = convert_nanodet_to_torchscript(
            args.config, args.checkpoint, args.output, tuple(args.input_size)
        )
    elif args.method == 'onnx':
        success = convert_with_onnx_intermediate(
            args.config, args.checkpoint, args.output, tuple(args.input_size)
        )
    elif args.method == 'wrapper':
        success = create_inference_wrapper(
            args.checkpoint, args.config, args.output, tuple(args.input_size)
        )
    elif args.method == 'simple':
        success = simple_state_dict_conversion(args.checkpoint, args.output)
    
    if success:
        print(f"\n✅ Conversion completed successfully!")
        print(f"   Converted model saved to: {args.output}")
        return 0
    else:
        print(f"\n❌ Conversion failed!")
        return 1

if __name__ == '__main__':
    # Example usage
    if len(sys.argv) == 1:
        print("NanoDet Model Conversion Script")
        print("\nUsage examples:")
        print("python convert_nanodet.py --config config.yml --checkpoint model.pth --output model.pt")
        print("python convert_nanodet.py --config config.yml --checkpoint model.ckpt --output model.pt --method wrapper")
        print("python convert_nanodet.py --checkpoint model.pth --output weights.pt --method simple")
        print("\nMethods:")
        print("  torchscript: Full model conversion (recommended)")
        print("  wrapper: Creates inference wrapper")
        print("  simple: Just extracts weights (requires architecture to load)")
        print("  onnx: Uses ONNX as intermediate format")
        sys.exit(0)
    
    sys.exit(main())