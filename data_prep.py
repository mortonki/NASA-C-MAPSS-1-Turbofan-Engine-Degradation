import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_data(base_path):
    cols = ['unit_id', 'cycle', 'op_cond_1', 'op_cond_2', 'op_cond_3'] + \
            [f'sensor_{i}' for i in range(1, 22)]
    
    train_df = pd.read_csv(os.path.join(base_path, 'train_FD001.txt'), sep=r'\s+', header=None, names=cols)
    test_df = pd.read_csv(os.path.join(base_path, 'test_FD001.txt'), sep=r'\s+', header=None, names=cols)
    label_df = pd.read_csv(os.path.join(base_path, 'RUL_FD001.txt'), sep=r'\s+', header=None, names=['RUL'])
    
    return train_df, test_df, label_df

def calculate_train_rul(train_df):
    # Group by unit_id and find the max cycle for each unit
    max_cycles = train_df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    # Merge back to the original dataframe
    train_df = train_df.merge(max_cycles, on='unit_id')
    
    # Calculate RUL
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    
    return train_df

def normalize_data(train_df, test_df):
    # We only want to normalize the sensor columns
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    
    scaler = StandardScaler()
    # Fit on training data
    scaler.fit(train_df[sensor_cols])
    
    # Transform both
    train_df[sensor_cols] = scaler.transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    
    return train_df, test_df

def create_sliding_windows(df, window_size=30):
    # This function will return a list of windows
    # Each window is a numpy array of shape (window_size, num_features)
    # We exclude 'unit_id' and 'cycle' from the features, but we need 'cycle' to know where we are
    # Actually, for RUL prediction, we want to predict RUL at the current cycle
    # So the input is the last 'window_size' cycles.
    
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
    
    # 1. Calculate RUL for training set
    train_df = calculate_train_rul(train_df)
    
    # 2. Normalize sensor data
    train_df, test_df = normalize_data(train_df, test_df)
    
    # 3. Align test RULs
    # The RUL_FD001.txt contains 100 values.
    # We assume they correspond to the 100 units in the test set in order of appearance.
    unique_units = test_df['unit_id'].unique()
    print(f"Unique units in test set: {len(unique_units)}")
    
    # Create a mapping of unit_id to RUL
    rul_mapping = dict(zip(unique_units, label_df['RUL'].values))
    
    # Add RUL to test_df
    test_df['RUL'] = test_df['unit_id'].map(rul_mapping)
    
    # Calculate RUL at each cycle for the test set
    # For each unit, the RUL at cycle c is RUL_at_last_cycle + (last_cycle - c)
    # We need to find the last cycle for each unit in the test set.
    max_cycles_test = test_df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles_test.columns = ['unit_id', 'max_cycle_test']
    test_df = test_df.merge(max_cycles_test, on='unit_id')
    
    # Now calculate the RUL at each cycle
    # RUL_at_cycle = RUL_at_last_cycle + (max_cycle_test - cycle)
    test_df['RUL'] = test_df.apply(lambda row: rul_mapping[row['unit_id']] + (row['max_cycle_test'] - row['cycle']), axis=1)
    
    # 4. Create sliding windows
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