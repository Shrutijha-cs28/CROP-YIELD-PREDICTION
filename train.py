# ============================================================
# FILE: src/train.py
# ============================================================
"""Training utilities."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, verbose=1):
        """Initialize trainer.
        
        Args:
            model: Compiled Keras model
            verbose: Verbosity level
        """
        self.model = model
        self.verbose = verbose
        self.history = None
        self.predictions = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, early_stopping=True,
              patience=10):
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0] if X_val is not None else 'None'}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        callbacks = []
        
        if early_stopping and X_val is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        print("Training complete!\n")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        y_pred = self.model.predict(X_test, verbose=0)
        self.predictions = y_pred
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        print(f"Mean Squared Error (MSE):       {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE):      {mae:.4f}")
        print(f"R² Score:                       {r2:.4f}")
        print(f"Mean Absolute % Error (MAPE):   {mape:.4f}%")
        print("="*60 + "\n")
        
        return metrics
    
    def predict(self, X):
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def plot_training_history(self, figsize=(12, 4)):
        """Plot training history.
        
        Args:
            figsize: Figure size
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], 
                        label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss Over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if 'mae' in self.history.history:
            axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
            if 'val_mae' in self.history.history:
                axes[1].plot(self.history.history['val_mae'], 
                            label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error Over Epochs')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded Keras model
        """
        model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return model
