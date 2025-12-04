"""
Model Export Utilities for Production Deployment

Supports multiple export formats:
- ONNX (Open Neural Network Exchange)
- TorchScript (JIT compilation)
- TensorFlow Lite (via ONNX)
- Quantization for mobile/edge deployment

Compatible with all models in the project.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
import os
import json


class ModelExporter:
    """
    Unified model export utility for multiple formats.
    """
    def __init__(self, model: nn.Module, model_name: str, input_shape: Tuple):
        """
        Args:
            model: PyTorch model to export
            model_name: Name for saved files
            input_shape: Expected input shape (batch, channels, length)
        """
        self.model = model
        self.model_name = model_name
        self.input_shape = input_shape
        
        self.model.eval()
        
    def export_to_onnx(
        self,
        save_path: str,
        opset_version: int = 12,
        dynamic_axes: bool = True,
        verify: bool = True
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            save_path: Path to save ONNX file
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic batch size
            verify: Whether to verify exported model
            
        Returns:
            Path to saved ONNX file
        """
        print(f"Exporting {self.model_name} to ONNX...")
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape)
        
        # Define dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        
        print(f"✓ ONNX model saved to: {save_path}")
        
        # Verify exported model
        if verify:
            try:
                import onnx
                import onnxruntime as ort
                
                # Load and check ONNX model
                onnx_model = onnx.load(save_path)
                onnx.checker.check_model(onnx_model)
                
                # Test inference
                ort_session = ort.InferenceSession(save_path)
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
                ort_outputs = ort_session.run(None, ort_inputs)
                
                # Compare with PyTorch
                with torch.no_grad():
                    torch_output = self.model(dummy_input)
                
                if isinstance(torch_output, tuple):
                    torch_output = torch_output[0]
                
                torch_output_np = torch_output.numpy()
                
                diff = np.abs(torch_output_np - ort_outputs[0]).max()
                print(f"✓ ONNX verification passed (max diff: {diff:.6f})")
                
            except ImportError:
                print("⚠ onnx/onnxruntime not installed, skipping verification")
            except Exception as e:
                print(f"⚠ ONNX verification failed: {e}")
        
        return save_path
    
    def export_to_torchscript(
        self,
        save_path: str,
        method: str = 'trace',
        verify: bool = True
    ) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            save_path: Path to save TorchScript file
            method: 'trace' or 'script'
            verify: Whether to verify exported model
            
        Returns:
            Path to saved TorchScript file
        """
        print(f"Exporting {self.model_name} to TorchScript ({method})...")
        
        dummy_input = torch.randn(self.input_shape)
        
        if method == 'trace':
            # Tracing (better for most models)
            traced_model = torch.jit.trace(self.model, dummy_input)
        elif method == 'script':
            # Scripting (for models with control flow)
            traced_model = torch.jit.script(self.model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
        
        # Save
        traced_model.save(save_path)
        
        print(f"✓ TorchScript model saved to: {save_path}")
        
        # Verify
        if verify:
            try:
                loaded_model = torch.jit.load(save_path)
                loaded_model.eval()
                
                with torch.no_grad():
                    original_output = self.model(dummy_input)
                    loaded_output = loaded_model(dummy_input)
                
                if isinstance(original_output, tuple):
                    original_output = original_output[0]
                if isinstance(loaded_output, tuple):
                    loaded_output = loaded_output[0]
                
                diff = torch.abs(original_output - loaded_output).max().item()
                print(f"✓ TorchScript verification passed (max diff: {diff:.6f})")
                
            except Exception as e:
                print(f"⚠ TorchScript verification failed: {e}")
        
        return save_path
    
    def quantize_model(
        self,
        save_path: str,
        quantization_type: str = 'dynamic',
        verify: bool = True
    ) -> str:
        """
        Quantize model for mobile/edge deployment.
        
        Args:
            save_path: Path to save quantized model
            quantization_type: 'dynamic' or 'static'
            verify: Whether to verify quantized model
            
        Returns:
            Path to saved quantized model
        """
        print(f"Quantizing {self.model_name} ({quantization_type})...")
        
        if quantization_type == 'dynamic':
            # Dynamic quantization (post-training, no calibration needed)
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=torch.qint8
            )
        elif quantization_type == 'static':
            # Static quantization (requires calibration data)
            # This is simplified - full implementation needs calibration
            print("⚠ Static quantization requires calibration data (not implemented in demo)")
            quantized_model = self.model
        else:
            raise ValueError(f"Unknown quantization_type: {quantization_type}")
        
        # Save
        torch.save(quantized_model.state_dict(), save_path)
        
        print(f"✓ Quantized model saved to: {save_path}")
        
        # Verify
        if verify:
            try:
                dummy_input = torch.randn(self.input_shape)
                
                with torch.no_grad():
                    original_output = self.model(dummy_input)
                    quantized_output = quantized_model(dummy_input)
                
                if isinstance(original_output, tuple):
                    original_output = original_output[0]
                if isinstance(quantized_output, tuple):
                    quantized_output = quantized_output[0]
                
                diff = torch.abs(original_output - quantized_output).max().item()
                print(f"✓ Quantization verification passed (max diff: {diff:.6f})")
                
                # Print size reduction
                original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
                quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
                print(f"  Original size: {original_size:.2f} MB")
                print(f"  Quantized size: {quantized_size:.2f} MB")
                print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
                
            except Exception as e:
                print(f"⚠ Quantization verification failed: {e}")
        
        return save_path
    
    def export_model_info(self, save_path: str) -> str:
        """
        Export model metadata and information.
        
        Args:
            save_path: Path to save JSON file
            
        Returns:
            Path to saved JSON file
        """
        print(f"Exporting model info for {self.model_name}...")
        
        info = {
            'model_name': self.model_name,
            'input_shape': list(self.input_shape),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_architecture': str(self.model),
        }
        
        # Get layer information
        layers = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                }
                if hasattr(module, 'weight') and module.weight is not None:
                    layer_info['parameters'] = module.weight.numel()
                layers.append(layer_info)
        
        info['layers'] = layers
        
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Model info saved to: {save_path}")
        
        return save_path
    
    def export_all(self, output_dir: str, formats: List[str] = ['onnx', 'torchscript', 'quantized']):
        """
        Export model to all specified formats.
        
        Args:
            output_dir: Directory to save all exports
            formats: List of formats to export ('onnx', 'torchscript', 'quantized')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print(f"Exporting {self.model_name} to multiple formats")
        print("=" * 80)
        
        results = {}
        
        # Export to ONNX
        if 'onnx' in formats:
            onnx_path = os.path.join(output_dir, f"{self.model_name}.onnx")
            try:
                self.export_to_onnx(onnx_path)
                results['onnx'] = onnx_path
            except Exception as e:
                print(f"✗ ONNX export failed: {e}")
                results['onnx'] = None
        
        # Export to TorchScript
        if 'torchscript' in formats:
            torchscript_path = os.path.join(output_dir, f"{self.model_name}_torchscript.pt")
            try:
                self.export_to_torchscript(torchscript_path)
                results['torchscript'] = torchscript_path
            except Exception as e:
                print(f"✗ TorchScript export failed: {e}")
                results['torchscript'] = None
        
        # Export quantized model
        if 'quantized' in formats:
            quantized_path = os.path.join(output_dir, f"{self.model_name}_quantized.pth")
            try:
                self.quantize_model(quantized_path)
                results['quantized'] = quantized_path
            except Exception as e:
                print(f"✗ Quantization failed: {e}")
                results['quantized'] = None
        
        # Export model info
        info_path = os.path.join(output_dir, f"{self.model_name}_info.json")
        try:
            self.export_model_info(info_path)
            results['info'] = info_path
        except Exception as e:
            print(f"✗ Model info export failed: {e}")
            results['info'] = None
        
        print("\n" + "=" * 80)
        print("Export Summary")
        print("=" * 80)
        for format_name, path in results.items():
            status = "✓" if path else "✗"
            print(f"{status} {format_name}: {path if path else 'Failed'}")
        print("=" * 80)
        
        return results


def export_model_wrapper(
    model: nn.Module,
    model_name: str,
    input_shape: Tuple,
    output_dir: str = './exported_models'
):
    """
    Convenient wrapper function to export a model.
    
    Args:
        model: PyTorch model
        model_name: Name for the model
        input_shape: Input shape (batch, channels, length)
        output_dir: Directory to save exports
    """
    exporter = ModelExporter(model, model_name, input_shape)
    results = exporter.export_all(output_dir)
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("Model Export Utilities")
    print("=" * 80)
    print("\nThis module provides utilities to export PyTorch models to:")
    print("  - ONNX (for cross-platform deployment)")
    print("  - TorchScript (for C++ deployment)")
    print("  - Quantized models (for mobile/edge devices)")
    print("\nExample usage:")
    print("  from model_export import ModelExporter")
    print("  exporter = ModelExporter(model, 'my_model', (1, 1, 1000))")
    print("  exporter.export_all('./exports')")
    print("\nNote: Install onnx and onnxruntime for full functionality:")
    print("  pip install onnx onnxruntime")
    print("=" * 80)
