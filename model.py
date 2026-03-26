# ============================================================
# FILE: src/model.py
# ============================================================
"""TensorFlow neural network models."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

def create_simple_model(input_dim, output_dim=1):
    """Create simple neural network.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output dimensions
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='hidden_1'),
        layers.Dense(32, activation='relu', name='hidden_2'),
        layers.Dense(output_dim, activation='linear', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def create_deep_model(input_dim, output_dim=1):
    """Create deeper neural network.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output dimensions
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dense(32, activation='relu', name='hidden_3'),
        layers.Dense(16, activation='relu', name='hidden_4'),
        layers.Dense(output_dim, activation='linear', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_regularized_model(input_dim, output_dim=1, 
                            l2_lambda=0.001, dropout_rate=0.3):
    """Create model with regularization.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output dimensions
        l2_lambda: L2 regularization strength
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_lambda),
            name='hidden_1'
        ),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_lambda),
            name='hidden_2'
        ),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_lambda),
            name='hidden_3'
        ),
        layers.Dropout(dropout_rate * 0.5, name='dropout_3'),
        
        layers.Dense(output_dim, activation='linear', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def create_presowing_focused_model(input_dim, output_dim=1):
    """Create model with pre-sowing focus.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output dimensions
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='batch_norm_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    x = layers.Dense(32, activation='relu', name='dense_3')(x)
    x = layers.Dropout(0.2, name='dropout_3')(x)
    
    outputs = layers.Dense(output_dim, activation='linear', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def get_model_summary(model):
    """Print model summary.
    
    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60)
