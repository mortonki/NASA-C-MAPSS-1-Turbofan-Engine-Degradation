import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def apply_outlier_capping(df, cols, lower=0.01, upper=0.99):
    # Capping values based on percentiles derived from the training distribution (or provided bounds) (Winsorizing)
    for col in cols:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

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

def calculate_rul(df, label_df=None, cap=None):
    """
    Calculate RUL for both training and test sets.
    
    If label_df is provided (test set), it uses the RUL values from label_df 
    and adjusts them based on the current cycle and the maximum cycle in the dataframe.
    If label_df is not provided (training set), it calculates RUL as 
    (max_cycle - current_cycle).
    """
    # Group by unit_id and find the max cycle for each unit
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "_max_cycle"]
    
    # Merge back to the original dataframe
    df = df.merge(max_cycles, on="unit_id")
    
    if label_df is not None:
        # Test set logic
        if "unit_id" in label_df.columns:
            rul_mapping = dict(zip(label_df["unit_id"], label_df["RUL"]))
        else:
            unique_units = df["unit_id"].unique()
            rul_mapping = dict(zip(unique_units, label_df["RUL"].values))
            
        df["RUL"] = df["unit_id"].map(rul_mapping) + (df["_max_cycle"] - df["cycle"])
    else:
        # Training set logic
        df["RUL"] = (df["_max_cycle"] - df["cycle"])
        
    df["RUL"] = df["RUL"].clip(upper=cap)
    
    # Drop the temporary column
    df.drop(columns=["_max_cycle"], inplace=True)
    
    return df

def calculate_train_rul(train_df, cap=None):
    return calculate_rul(train_df, label_df=None, cap=cap)

def normalize_data(train_df, test_df, val_df=None, sensor_cols=None):
    if sensor_cols is None:
        sensor_cols = train_df.columns.tolist()
    
    scaler = StandardScaler()
    # Fit on training data and transform both datasets (Standardization)
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    
    if val_df is not None:
        val_df[sensor_cols] = scaler.transform(val_df[sensor_cols])

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
            for i in range(window_size - 1, len(group)):
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

def create_test_windows(df, feature_cols, window_size=30):
    windows = []
    targets = []
    unit_ids = []

    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('cycle')

        if len(group) >= window_size:
            window = group.iloc[-window_size:][feature_cols].values
            target = group.iloc[-1]['RUL']

            windows.append(window)
            targets.append(target)
            unit_ids.append(unit_id)

    return (
        np.array(windows),
        np.array(targets),
        np.array(unit_ids)
    )

def create_feature_deltas(df, feature_cols):
    """
    Creates deltas of the specified feature columns.
    The deltas are calculated within each unit_id group.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_cols (list): The list of columns to calculate deltas for.
        
    Returns:
        tuple: A tuple containing (df_deltas, updated_feature_cols)
    """
    df_deltas = df.copy()
    delta_cols = []
    for col in feature_cols:
        delta_col = f"{col}_delta"
        df_deltas[delta_col] = df_deltas.groupby("unit_id")[col].diff()
        delta_cols.append(delta_col)
    
    df_deltas = df_deltas.dropna(subset=delta_cols)
    return df_deltas, feature_cols + delta_cols

def create_rolling_slope(df, feature_cols, window=10, min_periods=None):
    """
    Creates rolling slopes of the specified feature columns.
    The slopes are calculated within each unit_id group using a linear regression
    over a rolling window.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_cols (list): The list of columns to calculate rolling slopes for.
        window (int): The size of the rolling window. Defaults to 10.
        min_periods (int, optional): Minimum number of observations in window. Defaults to None.
        
    Returns:
        tuple: A tuple containing (df_slopes, updated_feature_cols)
    """
    df_slopes = df.copy()
    slope_cols = []
    for col in feature_cols:
        slope_col = f"{col}_slope"
        
        def calculate_slope(y):
            if len(y) < window:
                return np.nan
            x = np.arange(len(y))
            # Use polyfit to get the slope (degree 1)
            # polyfit returns [slope, intercept]
            return np.polyfit(x, y, 1)[0]

        df_slopes[slope_col] = df_slopes.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).apply(calculate_slope, raw=True)
        )
        slope_cols.append(slope_col)
    
    df_slopes = df_slopes.dropna(subset=slope_cols)
    return df_slopes, feature_cols + slope_cols

def create_rolling_mean(df, feature_cols, window=10, min_periods=None):
    """
    Creates rolling means of the specified feature columns.
    The means are calculated within each unit_id group using a rolling window.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_cols (list): The list of columns to calculate rolling means for.
        window (int): The size of the rolling window. Defaults to 10.
        min_periods (int, optional): Minimum number of observations in window. Defaults to None.
        
    Returns:
        tuple: A tuple containing (df_mean, updated_feature_cols)
    """
    df_mean = df.copy()
    mean_cols = []
    for col in feature_cols:
        mean_col = f"{col}_mean"
        df_mean[mean_col] = df_mean.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).mean()
        )
        mean_cols.append(mean_col)
    
    df_mean = df_mean.dropna(subset=mean_cols)
    return df_mean, feature_cols + mean_cols

def create_rolling_std(df, feature_cols, window=10, min_periods=None):
    """
    Creates rolling standard deviations of the specified feature columns.
    The standard deviations are calculated within each unit_id group using a rolling window.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_cols (list): The list of columns to calculate rolling standard deviations for.
        window (int): The size of the rolling window. Defaults to 10.
        min_periods (int, optional): Minimum number of observations in window. Defaults to None.
        
    Returns:
        tuple: A tuple containing (df_std, updated_feature_cols)
    """
    df_std = df.copy()
    std_cols = []
    for col in feature_cols:
        std_col = f"{col}_std"
        df_std[std_col] = df_std.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).std()
        )
        std_cols.append(std_col)
    
    df_std = df_std.dropna(subset=std_cols)
    return df_std, feature_cols + std_cols

def create_baseline_features(df, feature_cols, baseline_window=20):
    """
    Establishes a baseline for each unit_id based on the first \"baseline_window\" 
    rows of each sequence and adds features representing the change with respect 
    to that baseline.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature_cols (list): The list of columns to establish baseline for.
        baseline_window (int): The number of initial rows to use for the baseline.
        
    Returns:
        tuple: A tuple containing (df_baseline, updated_feature_cols)
    """
    df_baseline = df.copy()
    
    # We need to calculate baselines for each unit_id.
    # To do this correctly, we need to know which rows are the first baseline rows for each unit_id.
    # We can do this by sorting a copy of the dataframe.
    sorted_df = df_baseline.sort_values(['unit_id', 'cycle'])
    
    baselines = []
    for unit_id, group in sorted_df.groupby("unit_id"):
        baseline_data = group.iloc[:baseline_window]
        baseline_values = baseline_data[feature_cols].mean().to_dict()
        baseline_values['unit_id'] = unit_id
        baselines.append(baseline_values)
    
    baseline_df = pd.DataFrame(baselines)
    
    # Merge the baseline values back into the original dataframe (df_baseline)
    # This will preserve the original order of df_baseline.
    df_baseline = df_baseline.merge(baseline_df, on='unit_id', how='left', suffixes=('', '_baseline'))
    
    # Create the delta columns
    new_cols = []
    for col in feature_cols:
        delta_col = f"{col}_baseline_delta"
        df_baseline[delta_col] = df_baseline[col] - df_baseline[f"{col}_baseline"]
        new_cols.append(delta_col)
        
    # Drop the intermediate baseline columns
    baseline_cols_to_drop = [f"{col}_baseline" for col in feature_cols]
    df_baseline.drop(columns=baseline_cols_to_drop, inplace=True)
    
    return df_baseline, feature_cols + new_cols


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
    
    # Establish baseline features
    train_df = create_baseline_features(train_df, feature_cols)
    test_df = create_baseline_features(test_df, feature_cols)
    
    # Update feature_cols to include the new baseline delta features
    feature_cols = feature_cols + [f"{col}_baseline_delta" for col in feature_cols]
    
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