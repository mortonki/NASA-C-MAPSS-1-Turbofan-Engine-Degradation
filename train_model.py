import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from data_prep import load_data, calculate_train_rul, normalize_data, create_sliding_windows
import time
import mlflow
import argparse

# Default hyperparameters
WINDOW_SIZE = 70
NUM_FEATURES = 14
HIDDEN_SIZE = 128
NUM_LAYERS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.00025
EPOCHS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch_size, window_size, hidden_size)
        # We take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

def build_parser():
    parser = argparse.ArgumentParser(description="Train an LSTM model for RUL prediction")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help="Size of the sliding window")
    parser.add_argument("--num-features", type=int, default=NUM_FEATURES,
                        help="Number of input features per time step")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE,
                        help="Hidden size of the LSTM")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                        help="Number of stacked LSTM layers")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train")

    # Optional: let user override device
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
    NUM_FEATURES = args.num_features
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    DEVICE = args.device

    base_path = '/home/mordicus/.cache/kagglehub/datasets/bishals098/nasa-turbofan-engine-degradation-simulation/versions/1'
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("LSTM_RUL_Prediction")
    
     # Load data
    print("Loading and preprocessing data...")
    train_df, test_df, label_df = load_data(base_path)

    # Drop columns based on EDA findings to reduce noise and improve model performance
    columns_to_drop = ['op_cond_1', 'op_cond_2', 'op_cond_3', 'sensor_1', 'sensor_5', 'sensor_6','sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    train_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Dynamically select remaining sensor columns present in the train dataframe
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    op_cond_cols = [col for col in train_df.columns if col.startswith('op_cond_')]
    feature_cols = sensor_cols + op_cond_cols

    # Preprocess data
    train_df = calculate_train_rul(train_df)
    train_df, test_df = normalize_data(train_df, test_df)
    
    # Align test RULs
    unique_units = test_df['unit_id'].unique()
    rul_mapping = dict(zip(unique_units, label_df['RUL'].values))
    test_df['RUL'] = test_df['unit_id'].map(rul_mapping)
    max_cycles_test = test_df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles_test.columns = ['unit_id', 'max_cycle_test']
    test_df = test_df.merge(max_cycles_test, on='unit_id')
    test_df['RUL'] = test_df.apply(lambda row: rul_mapping[row['unit_id']] + (row['max_cycle_test'] - row['cycle']), axis=1)
    
    print(f"Creating sliding windows...")
    train_windows, train_targets = create_sliding_windows(train_df, feature_cols, WINDOW_SIZE)
    test_windows, test_targets = create_sliding_windows(test_df, feature_cols, WINDOW_SIZE)
    
    # Convert to tensors
    X_train = torch.FloatTensor(train_windows)
    y_train = torch.FloatTensor(train_targets).view(-1, 1)
    X_test = torch.FloatTensor(test_windows)
    y_test = torch.FloatTensor(test_targets).view(-1, 1)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training LSTM model on {DEVICE}...")
    model = LSTMModel(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, 1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Log hyperparameters to MLflow and train
    with mlflow.start_run():
        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_param("num_features", NUM_FEATURES)
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("num_layers", NUM_LAYERS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("device", DEVICE)
        mlflow.log_metric("train_samples", len(train_windows))
        mlflow.log_metric("test_samples", len(test_windows))
        
        model.train()
        start_time = time.time()
    
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
                mlflow.log_metric("loss", avg_loss)
    
        end_time = time.time()
        training_time = end_time - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        # Save the model
        #torch.save(model.state_dict(), 'lstm_model.pth')
        #print("Model saved to lstm_model.pth")

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
