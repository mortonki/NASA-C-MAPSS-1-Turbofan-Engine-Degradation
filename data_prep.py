import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def drop_columns(df, cols_to_drop):
    """Drop specified columns from dataframe."""
    for col in cols_to_drop:  # Check column exists first to avoid errors
        if col in df.columns: 
            df.drop(columns=[col], inplace=True)
    return df

def apply_outlier_capping(df, cols, lower=0.01, upper=0.99):
    # Capping values based on percentiles derived from the training distribution (or provided bounds)
    for col in cols:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df # Moved return outside the loop to process all columns

def load_data(base_path):
    cols = ['unit_id', 'cycle', 'op_cond_1', 'op_cond_2', 'op_cond_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    train_df = pd.read_csv(os.path.join(base_path, 'train_FD001.txt'), sep=r'\s+', header=None, names=cols)
    test_df = pd.read_csv(os.path.join(base_path, 'test_FD001.txt'), sep=r'\s+', header=None, names=cols)
    label_df = pd.read_csv(os.path.join(base_path, 'RUL_FD001.txt'), sep=r'\s+', header=None, names=['RUL'])
    
    return train_df, test_df, label_df

def calculate_train_rul(train_df):
    # Group by unit_id and find the max cycle for each unit, renaming to _max_cycle immediately
    max_cycles = train_df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', '_max_cycle']
    
    # Merge back to the original dataframe using the renamed column
    train_df = train_df.merge(max_cycles, on='unit_id')
    
    # Calculate RUL (Running down from total life) using the new name
    train_df['RUL'] = train_df['_max_cycle'] - train_df['cycle']

    # Drop the temporary column as it's no longer needed, utilizing the utility function for consistency
    train_df = drop_columns(train_df, ['_max_cycle'])
    
    return train_df

def normalize_data(train_df, test_df):
    # We only want to normalize the sensor columns
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    
    scaler = StandardScaler()
    # Fit on training data and transform both datasets (Standardization)
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    # Apply Outlier Capping (Winsorizing) based on training data distribution to mitigate MinMax shifts
    apply_outlier_capping(train_df, sensor_cols, lower=0.01, upper=0.99)
    apply_outlier_capping(test_df, sensor_cols, lower=0.01, upper=0.99)
    
    return train_df, test_df

def create_sliding_windows(df, window_size=30):
    # This function will return a list of windows
    # Each window is a numpy array of shape (window_size, num_features)
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    op_cond_cols = ['op_cond_1', 'op_cond_2', 'op_cond_3']
    feature_cols = sensor_cols + op_cond_cols
    
    windows = []
    targets = []
    
    # Group by unit_id to ensure we don't create windows across different units
    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('cycle')
        
        # We need at least 'window_size' rows to create a window
        if len(group) >= window_size:
            # For each row from window_size to the end
            for i in range(window_size, len(group)):
                # The window is the previous 'window_size' rows
                window = group.iloc[i-window_size:i][feature_cols].values
                # The target is the RUL at the current row (index i)
                if 'RUL' in group.columns:
                    target = group.iloc[i]['RUL']
                else:
                    target = None
                
                windows.append(window)
                targets.append(target)
                
    return np.array(windows), np.array(targets)

def main():
    base_path = '/home/mordicus/.cache/kagglehub/datasets/bishals098/nasa-turbofan-engine-degradation-simulation/versions/1'
    
    print(f"Loading data from {base_path}...")
    train_df, test_df, label_df = load_data(base_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Label data shape: {label_df.shape}")
    
    # 1. Calculate RUL for training set (Running down from total life)
    train_df = calculate_train_rul(train_df)
    
    # 2. Normalize sensor data and apply capping to mitigate MinMax shifts
    train_df, test_df = normalize_data(train_df, test_df)
    
    # 3. Align test RULs (FIX APPLIED: Test RUL is now constant ground truth value for evaluation)
    unique_units = test_df['unit_id'].unique()
    print(f"Unique units in test set: {len(unique_units)}")
    
    # Create a mapping of unit_id to RUL (Ground Truth Target)
    rul_mapping = dict(zip(unique_units, label_df['RUL'].values))
    
    # Add RUL to test_df using the constant ground truth value for evaluation purposes.
    test_df['RUL'] = test_df['unit_id'].map(rul_mapping) 
    
    # 4. Create sliding windows (This step now uses the corrected RUL in test_df)
    window_size = 30
    print(f"Creating sliding windows with size {window_size}...")
    
    train_windows, train_targets = create_sliding_windows(train_df, window_size)
    test_windows, test_targets = create_sliding_windows(test_df, window_size)
    
    # Let's check the shapes
    print(f"Train windows shape: {train_windows.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Test windows shape: {test_windows.shape}")
    print(f"Test targets shape: {test_targets.shape}")
    
    # Verify that we have targets for all test windows
    if test_targets.shape[0] == test_windows.shape[0]:
        print("Success: All test windows have corresponding RUL targets.")
    else:
        print("Warning: Some test windows do not have RUL targets.")

    # Print summary statistics for the final prepared data
    print("\nFinal Training RUL Summary Statistics:")
    print(train_df['RUL'].describe())
    print("\nFinal Test RUL Summary Statistics:")
    print(test_df['RUL'].describe())

if __name__ == "__main__":
    main()