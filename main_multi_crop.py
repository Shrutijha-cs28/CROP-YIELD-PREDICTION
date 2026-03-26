# ============================================================
# FILE: main_multi_crop.py
# ============================================================
"""Main entry point for multi-crop training and prediction."""

import argparse
import os
from src.crop_config import CropConfig
from src.data_loader import DataLoader
from src.multi_crop_feature_engineer import MultiCropFeatureEngineer
from src.multi_crop_model import MultiCropModelFactory
from src.train import ModelTrainer
from src.utils import print_metrics_report
import pandas as pd

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Multi-Crop Yield Prediction')
    parser.add_argument('--crop', type=str, 
                       choices=['rice', 'wheat', 'maize', 'makhana', 'lychee', 'oilseed', 'sugarcane', 'pulses'],
                       help='Crop to train/predict')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'train_all', 'info'],
                       default='train', help='Mode')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--input_path', type=str, help='Path to input for prediction')
    parser.add_argument('--sowing_month', type=int, help='Override sowing month')
    
    args = parser.parse_args()
    
    crop_config = CropConfig()
    
    if args.mode == 'info':
        if args.crop:
            crop_config.print_crop_info(args.crop)
        else:
            crop_config.print_all_crops()
    
    elif args.mode == 'train_all':
        print("\n🌾 Training models for all crops...")
        print("="*60)
        for crop in crop_config.get_all_crops():
            try:
                data_path = f'data/raw/{crop}/5_years_data.csv'
                if os.path.exists(data_path):
                    train_single_crop(crop, data_path)
                else:
                    print(f"⚠️  Data not found for {crop} at {data_path}")
            except Exception as e:
                print(f"❌ Error training {crop}: {str(e)}")
    
    elif args.mode == 'train':
        if not args.crop or not args.data_path:
            print("Error: --crop and --data_path required for train mode")
            return
        train_single_crop(args.crop, args.data_path, args.sowing_month)
    
    elif args.mode == 'predict':
        if not args.crop or not args.model_path or not args.input_path:
            print("Error: --crop, --model_path, and --input_path required for predict mode")
            return
        predict_single_crop(args.crop, args.model_path, args.input_path)

def train_single_crop(crop_name, data_path=None, sowing_month=None):
    """Train model for single crop.
    
    Args:
        crop_name: Name of crop
        data_path: Path to training data
        sowing_month: Override sowing month
    """
    crop_config = CropConfig()
    crop_config.print_crop_info(crop_name)
    
    if data_path is None:
        data_path = f'data/raw/{crop_name}/5_years_data.csv'
    
    print(f"Loading data from {data_path}...")
    
    # Load data
    loader = DataLoader()
    loader.load_data(data_path)
    loader.handle_missing_values(method='mean')
    loader.explore_data()
    
    X, y = loader.prepare_features(target_column='Yield')
    
    # Engineer features
    engineer = MultiCropFeatureEngineer()
    X_engineered = engineer.engineer_features_for_crop(X, crop_name)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X_engineered, y)
    
    # Create and train model
    model_factory = MultiCropModelFactory()
    model = model_factory.create_model_for_crop(crop_name, X_train.shape[1])
    model_factory.print_crop_model_info(crop_name)
    
    trainer = ModelTrainer(model)
    params = model_factory.get_model_config_for_crop(crop_name)
    
    trainer.train(X_train, y_train, X_val, y_val, 
                 epochs=params['epochs'],
                 batch_size=params['batch_size'])
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    print_metrics_report(metrics)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{crop_name}_model.h5'
    trainer.save_model(model_path)
    
    print(f"\n✅ {crop_name.upper()} model trained and saved!")

def predict_single_crop(crop_name, model_path, input_path):
    """Make predictions for single crop.
    
    Args:
        crop_name: Name of crop
        model_path: Path to saved model
        input_path: Path to input data
    """
    print(f"\n🔮 Making predictions for {crop_name.upper()}...")
    print(f"Loading model from {model_path}...")
    
    trainer = ModelTrainer(None)
    model = trainer.load_model(model_path)
    trainer.model = model
    
    # Load input data
    data = pd.read_csv(input_path)
    print(f"Loaded {len(data)} records for prediction")
    
    # Make predictions
    predictions = trainer.predict(data)
    
    # Save predictions
    os.makedirs('results', exist_ok=True)
    output_path = f'results/{crop_name}_predictions.csv'
    results_df = pd.DataFrame({
        'Predicted_Yield': predictions.flatten()
    })
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

if __name__ == '__main__':
    main()
