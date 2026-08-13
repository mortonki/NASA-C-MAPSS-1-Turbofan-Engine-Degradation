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

def split_train_val(train_df, val_split=0.2):
    """
    Splits the training dataframe into train and validation sets based on unit_id.
    Ensures that no unit_id exists in both sets.
    """
    unique_units = train_df['unit_id'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_units)
    
    split_idx = int(len(unique_units) * (1 - val_split))
    train_units = unique_units[:split_idx]
    val_units = unique_units[split_idx:]
    
    train_df_split = train_df[train_df['unit_id'].isin(train_units)].copy()
    val_df_split = train_df[train_df['unit_id'].isin(val_units)].copy()
    
    return train_df_split, val_df_split


def load_data(base_path):
    cols = ['unit_id', 'cycle', 'op_cond_1', 'op_cond_2', 'op_cond_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    train_df = pd.read_csv(os.path.join(base_path, 'train_FD001.txt'), sep=r'\s+', header=None, names=cols)
    test_df = pd.read_csv(os.path.join(base_path, 'test_FD001.txt'), sep=r'\s+', header=None, names=cols)
    label_df = pd.read_csv(os.path.join(base_path, 'RUL_FD001.txt'), sep=r'\s+', header=None, names=['RUL'])
    
    return train_df, test_df, label_df

def calculate_train_rul(train_df, cap=None):
    # Group by unit_id and find the max cycle for each unit, renaming to _max_cycle immediately
    max_cycles = train_df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', '_max_cycle']
    
    # Merge back to the original dataframe using the renamed column
    train_df = train_df.merge(max_cycles, on='unit_id')
    
    # Calculate RUL (Running down from total life) from max_cycle minus the current cycle, using the temporary _max_cycle column
    train_df['RUL'] = (train_df['_max_cycle'] - train_df['cycle']).clip(upper=cap) # Clip RUL to a maximum of cap cycles

    # Drop the temporary column as it's no longer needed, utilizing the utility function for consistency
    train_df.drop(columns=['_max_cycle'], inplace=True)
    
    return train_df

def normalize_data(train_df, test_df, val_df=None):
    # We only want to normalize the sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    
    scaler = StandardScaler()
    # Fit on training data and transform both datasets (Standardization)
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    
    if val_df is not None:
        val_df[sensor_cols] = scaler.transform(val_df[sensor_cols])

    # Apply Outlier Capping (Winsorizing) based on training data distribution to mitigate MinMax shifts
    lower_bounds = train_df[sensor_cols].quantile(0.01)
    upper_bounds = train_df[sensor_cols].quantile(0.99)

    for df in [train_df, val_df, test_df]:
        df[sensor_cols] = df[sensor_cols].clip(
            lower=lower_bounds,
            upper=upper_bounds,
            axis="columns"
        )
    
    if val_df is not None:
        return train_df, test_df, val_df
    return train_df, test_df

def create_sliding_windows(df, feature_cols, window_size=30):
    # This function will return a list of windows
    # Each window is a numpy array of shape (window_size, num_features)
    
    windows = []
    targets = []
    
    # Group by unit_id to ensure we don't create windows across different units
    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('cycle')
        
        # We need at least 'window_size' rows to create a window
        if len(group) >= window_size:
            # For each row from window_size to the end
            for i in range(window_size, len(group)):
                # The window is the 'window_size' rows ending at i (indices i-window_size+1 to i)
                window = group.iloc[i-window_size+1:i+1][feature_cols].values
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
    
    # Calculate RUL for training set (Running down from total life)
    train_df = calculate_train_rul(train_df)

    # Dynamically select remaining sensor columns present in the train dataframe
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    op_cond_cols = [col for col in train_df.columns if col.startswith('op_cond_')]
    feature_cols = sensor_cols + op_cond_cols
    
    # Normalize sensor data and apply capping to mitigate MinMax shifts
    train_df, test_df = normalize_data(train_df, test_df)
    
    # 3. Align test RULs
    unique_units = test_df['unit_id'].unique()
    print(f"Unique units in test set: {len(unique_units)}")
    
    # Create a mapping of unit_id to RUL (Ground Truth Target)
    rul_mapping = dict(zip(unique_units, label_df['RUL'].values))
    
    # Add RUL to test_df using the constant ground truth value for evaluation purposes.
    test_df['RUL'] = test_df['unit_id'].map(rul_mapping) 
    
    # Create sliding windows (This step now uses the corrected RUL in test_df)
    window_size = 30
    print(f"Creating sliding windows with size {window_size}...")
    
    train_windows, train_targets = create_sliding_windows(train_df, feature_cols, window_size)
    test_windows, test_targets = create_sliding_windows(test_df, feature_cols, window_size)
    
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