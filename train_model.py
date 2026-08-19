import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from data_prep import load_data, calculate_rul, calculate_train_rul, normalize_data, create_sliding_windows, create_test_windows, create_feature_deltas, create_rolling_slope, create_rolling_mean, create_rolling_std, create_baseline_features, split_train_val
import time
import mlflow
import argparse

# Default hyperparameters
WINDOW_SIZE = 80
STRIDE = 1
NUM_FEATURES = 31
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 256 # Windowing produces many highly correlated samples, so a larger batch size can help the model generalize better by avoiding fitting to small number of samples and then failing to generalize to the rest of the data. This is especially important in time series data where consecutive samples are often very similar.
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_DELTA = 0.0001 # Delta for early stopping to avoid stopping too early due to minor fluctuations in validation loss
EPOCHS = 500
SEED = 42
CAP = None  # Cap for RUL values to avoid extreme values affecting the model
WEIGHT_DECAY = 1e-4  # Weight decay for the optimizer
DROPOUT = 0.2  # Dropout rate
SCHEDULER_FACTOR = 0.5  # Factor by which the learning rate is reduced
SCHEDULER_MIN_LR = 1e-6  # Minimum learning rate for the scheduler
SCHEDULER_PATIENCE = 10  # Patience for the scheduler
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch_size, window_size, hidden_size)
        # We take the output of the last time step
        out = self.dropout(out[:, -1, :]) # Dropout applied to the last time step's output
        out = self.fc(out)
        return out

def build_parser():
    parser = argparse.ArgumentParser(description="Train an LSTM model for RUL prediction")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help="Size of the sliding window")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help="Stride for the sliding window")
    parser.add_argument("--num-features", type=int, default=NUM_FEATURES,
                        help="Number of input features per time step")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE,
                        help="Hidden size of the LSTM")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                        help="Number of stacked LSTM layers")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help="Number of epochs with no improvement in validation loss before stopping")
    parser.add_argument("--early-stopping-delta", type=float, default=EARLY_STOPPING_DELTA,
                        help="Minimum change in validation loss to qualify as an improvement")
    parser.add_argument("--cap", type=int, default=CAP,
                        help="Cap for RUL values to avoid extreme values affecting the model")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay for the optimizer")
    parser.add_argument("--scheduler-factor", type=float, default=SCHEDULER_FACTOR,
                        help="Factor by which the learning rate is reduced")
    parser.add_argument("--scheduler-min-lr", type=float, default=SCHEDULER_MIN_LR,
                        help="Minimum learning rate for the scheduler")
    parser.add_argument("--scheduler-patience", type=int, default=SCHEDULER_PATIENCE,
                        help="Patience for the scheduler")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=DEVICE,
        help="Device to run on (default inferred from CUDA availability)",
    )
    return parser

def main():
    # Build the parser with default hyperparameters and parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # Update hyperparameters from command line arguments
    WINDOW_SIZE = args.window_size
    STRIDE = args.stride
    NUM_FEATURES = args.num_features
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    SEED = args.seed
    DEVICE = args.device
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    EARLY_STOPPING_DELTA = args.early_stopping_delta
    CAP = args.cap
    WEIGHT_DECAY = args.weight_decay
    SCHEDULER_FACTOR = args.scheduler_factor
    SCHEDULER_MIN_LR = args.scheduler_min_lr
    SCHEDULER_PATIENCE = args.scheduler_patience
    DROPOUT = args.dropout

    # Set random seeds for reproducibility
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(SEED)

    # Initialize MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("LSTM_RUL_Prediction")
    
     # Load data
    print("Loading and preprocessing data...")
    base_path = '/home/mordicus/.cache/kagglehub/datasets/bishals098/nasa-turbofan-engine-degradation-simulation/versions/1'
    df, test_df, label_df = load_data(base_path)

    # Drop columns based on EDA findings to reduce noise and improve model performance
    columns_to_drop = ['sensor_1', 'sensor_5', 'sensor_6','sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Dynamically select remaining sensor columns present in the train dataframe
    feature_cols = [col for col in df.columns if col.startswith('sensor_')]
    op_cond_cols = [col for col in df.columns if col.startswith('op_cond_')]
 
    # Preprocess data
    df, feature_cols_with_baseline = create_baseline_features(df, feature_cols, baseline_window=50)    
    df = calculate_train_rul(train_df=df, cap=CAP)  # Cap RUL at specified value to avoid extreme values affecting the model
    train_df, val_df = split_train_val(df)
    # Process test data: create deltas and align RULs
    test_df, _ = create_baseline_features(test_df, feature_cols, baseline_window=50)    
    test_df = calculate_rul(test_df, label_df=label_df, cap=CAP)
    # Normalize all datasets (train, validation, test) using the training set statistics
    train_df, test_df, val_df = normalize_data(train_df, test_df, val_df, sensor_cols=feature_cols_with_baseline+op_cond_cols) # We only want to normalize the sensor columns
    
    print(f"Creating sliding windows...")
    train_windows, train_targets = create_sliding_windows(train_df, feature_cols_with_baseline+op_cond_cols, WINDOW_SIZE, STRIDE)  # Use a step size of STRIDE to reduce the number of highly correlated samples
    val_windows, val_targets = create_sliding_windows(val_df, feature_cols_with_baseline+op_cond_cols, WINDOW_SIZE, STRIDE)
    test_windows, test_targets, _ = create_test_windows(test_df, feature_cols_with_baseline+op_cond_cols, WINDOW_SIZE)
    
    # Convert to tensors
    X_train = torch.FloatTensor(train_windows)
    y_train = torch.FloatTensor(train_targets).view(-1, 1)
    X_val = torch.FloatTensor(val_windows)
    y_val = torch.FloatTensor(val_targets).view(-1, 1)
    X_test = torch.FloatTensor(test_windows)
    y_test = torch.FloatTensor(test_targets).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training LSTM model on {DEVICE}...")
    model = LSTMModel(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, 1, dropout=DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, min_lr=SCHEDULER_MIN_LR, patience=SCHEDULER_PATIENCE)
    
    # Log hyperparameters to MLflow and train
    with mlflow.start_run():
        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_param("num_features", NUM_FEATURES)
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("num_layers", NUM_LAYERS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("dropout", DROPOUT)
        mlflow.log_param("early_stopping_patience", EARLY_STOPPING_PATIENCE)
        mlflow.log_param("early_stopping_delta", EARLY_STOPPING_DELTA)
        mlflow.log_param("cap", CAP)
        mlflow.log_param("scheduler_factor", SCHEDULER_FACTOR)
        mlflow.log_param("scheduler_min_lr", SCHEDULER_MIN_LR)
        mlflow.log_param("scheduler_patience", SCHEDULER_PATIENCE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("device", DEVICE)
        mlflow.log_metric("train_samples", len(train_windows))
        mlflow.log_metric("test_samples", len(test_windows))
        start_time = time.time()
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_epoch = 0
    
        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)
            
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    total_val_loss += loss.item() * batch_x.size(0)
            
            avg_val_loss = total_val_loss / len(val_loader.dataset)
            avg_val_rmse = np.sqrt(avg_val_loss)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss - EARLY_STOPPING_DELTA:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_epoch = epoch + 1
                torch.save(model.state_dict(), 'best_lstm_model.pth')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                avg_train_loss = total_train_loss / len(train_loader.dataset)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val RMSE: {avg_val_rmse:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                mlflow.log_metric("train_loss", avg_train_loss)
                mlflow.log_metric("val_loss", avg_val_loss)
                mlflow.log_metric("val_rmse", avg_val_rmse)
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'])

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_val_rmse", np.sqrt(best_val_loss))
        mlflow.log_metric("best_epoch", best_epoch)
        model.train()
    
        end_time = time.time()
        training_time = end_time - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        print(f"Training completed in {training_time:.2f} seconds.")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(X_test.to(DEVICE))
            mse = criterion(predictions, y_test.to(DEVICE))
            print(f"Test MSE: {mse.item():.4f}")
            
            # Calculate RMSE
            rmse = np.sqrt(mse.item())
            print(f"Test RMSE: {rmse:.4f}")
        
        # Log evaluation metrics to MLflow
        mlflow.log_metric("test_mse", mse.item())
        mlflow.log_metric("test_rmse", rmse)

if __name__ == "__main__":
    main()
