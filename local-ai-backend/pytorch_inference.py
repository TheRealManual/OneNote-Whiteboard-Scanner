"""
PyTorch Segmentation Inference - Alternative to ONNX Runtime
Uses TorchScript models (.pts files) instead of ONNX
"""
import torch
import numpy as np
from pathlib import Path


class PyTorchSegmentationInference:
    """Inference wrapper for PyTorch/TorchScript segmentation models"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize inference session
        
        Args:
            model_path: Path to .pts (TorchScript) model file
            device: 'cpu' or 'cuda'
        """
        self.device = device
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load TorchScript model
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()
        
        print(f"Loaded PyTorch model: {model_path} on {device}")
    
    def run(self, input_names, input_data):
        """
        Run inference (ONNX Runtime compatible interface)
        
        Args:
            input_names: List of input names (ignored for PyTorch)
            input_data: Dict mapping input names to numpy arrays
        
        Returns:
            List of output numpy arrays
        """
        # Get input tensor
        input_array = list(input_data.values())[0]
        
        # Convert to torch tensor
        input_tensor = torch.from_numpy(input_array).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Convert back to numpy
        output_array = output_tensor.cpu().numpy()
        
        return [output_array]
    
    def get_inputs(self):
        """Get input info (mock for ONNX Runtime compatibility)"""
        class InputInfo:
            def __init__(self):
                self.name = 'input'
                self.shape = ['batch', 3, 480, 640]
        return [InputInfo()]
    
    def get_outputs(self):
        """Get output info (mock for ONNX Runtime compatibility)"""
        class OutputInfo:
            def __init__(self):
                self.name = 'output'
                self.shape = ['batch', 4, 480, 640]
        return [OutputInfo()]


def create_inference_session(model_path, providers=None):
    """
    Create an inference session (ONNX Runtime compatible factory)
    
    Args:
        model_path: Path to model file (.pts for PyTorch, .onnx for ONNX Runtime)
        providers: Execution providers (ignored for PyTorch, uses CPU)
    
    Returns:
        Inference session object
    """
    model_path = Path(model_path)
    
    if model_path.suffix == '.pts':
        # Use PyTorch
        device = 'cuda' if torch.cuda.is_available() and providers and 'CUDAExecutionProvider' in providers else 'cpu'
        return PyTorchSegmentationInference(model_path, device=device)
    
    elif model_path.suffix == '.onnx':
        # Use ONNX Runtime
        import onnxruntime as ort
        return ort.InferenceSession(str(model_path), providers=providers or ['CPUExecutionProvider'])
    
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
