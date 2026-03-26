# ============================================================
# FILE: src/multi_crop_model.py
# ============================================================
"""Multi-crop model factory."""

from src.crop_config import CropConfig
from src.model import (
    create_simple_model,
    create_deep_model,
    create_regularized_model,
    create_presowing_focused_model
)

class MultiCropModelFactory:
    """Create models optimized for different crops."""
    
    def __init__(self):
        """Initialize model factory."""
        self.crop_config = CropConfig()
    
    def create_model_for_crop(self, crop_name, input_dim, model_type='regularized'):
        """Create model optimized for specific crop.
        
        Args:
            crop_name: Name of crop
            input_dim: Number of input dimensions
            model_type: Type of model to create
            
        Returns:
            Compiled Keras model
        """
        config = self.crop_config.get_crop_config(crop_name)
        print(f"\nCreating {model_type} model for {config['crop_name']}...")
        
        if model_type == 'simple':
            model = create_simple_model(input_dim)
        elif model_type == 'deep':
            model = create_deep_model(input_dim)
        elif model_type == 'regularized':
            model = create_regularized_model(input_dim)
        elif model_type == 'presowing':
            model = create_presowing_focused_model(input_dim)
        else:
            model = create_regularized_model(input_dim)
        
        return model
    
    def get_model_config_for_crop(self, crop_name):
        """Get model configuration for crop.
        
        Args:
            crop_name: Name of crop
            
        Returns:
            Dictionary with model parameters
        """
        return self.crop_config.get_model_params(crop_name)
    
    def print_crop_model_info(self, crop_name):
        """Print model info for crop.
        
        Args:
            crop_name: Name of crop
        """
        config = self.get_model_config_for_crop(crop_name)
        print(f"\n{'='*60}")
        print(f"MODEL CONFIG: {crop_name.upper()}")
        print(f"{'='*60}")
        for key, value in config.items():
            print(f"{key:.<35} {value}")
        print(f"{'='*60}\n")
