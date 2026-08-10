import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import kagglehub
    from pathlib import Path
    import pandas as pd
    import matplotlib
    matplotlib.use("module://marimo._output.mpl")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import marimo as mo
    from sklearn.cluster import KMeans
    from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting in Matplotlib
    from data_prep import load_data, calculate_train_rul, normalize_data, apply_outlier_capping, create_sliding_windows

    return (
        KMeans,
        Path,
        calculate_train_rul,
        kagglehub,
        load_data,
        mo,
        pd,
        plt,
        sm,
        sns,
    )


@app.cell
def _(kagglehub):
    # Download latest version of the dataset from Kaggle
    base_path = kagglehub.dataset_download("bishals098/nasa-turbofan-engine-degradation-simulation")

    print(f"Path to dataset files: {base_path}")
    return (base_path,)


@app.cell
def _(Path, base_path):
    # Convert the string path to a Path object
    dataset_dir = Path(base_path)
    return (dataset_dir,)


@app.cell
def _(dataset_dir):
    # List all files in the directory
    for file in dataset_dir.iterdir():
        print(file.name)
    return


@app.cell
def _(dataset_dir):
    # Define and verify required files
    required_files = {'train': 'train_FD004.txt', 'test': 'test_FD004.txt', 'label': 'RUL_FD004.txt'}

    paths = {}
    for key, filename in required_files.items():
        file_path = dataset_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f'Required file {filename} not found in {dataset_dir}')
        paths[key] = file_path
        print(f'Verified: {filename} exists at {file_path}')
    _train_file = paths['train']
    _test_file = paths['test']
    _label_file = paths['label']
    return (required_files,)


@app.cell
def _(dataset_dir, required_files):
    # Define file paths
    _train_file = dataset_dir / required_files['train']
    _test_file = dataset_dir / required_files['test']
    _label_file = dataset_dir / required_files['label']
    print(f'Training file: {_train_file}')
    print(f'Test file: {_test_file}')
    print(f'Label file: {_label_file}')
    return


@app.cell
def _(base_path, load_data):
    # Load and process data using functions from data_prep.py for EDA purposes only (no normalization/windowing)
    train_df, test_df, label_df = load_data(base_path)

    # Display basic information using the processed dataframes
    print(f"Training data shape: {train_df.shape}")
    print(f"Label data shape: {label_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return label_df, test_df, train_df


@app.cell
def _(train_df):
    print("\nTraining data head:")
    train_df.head(1)
    return


@app.cell
def _(label_df):
    print("\nLabel data head:")
    label_df.head(1)
    return


@app.cell
def _(test_df):
    print("\nTest data head:")
    test_df.head(1)
    return


@app.cell
def _(train_df):
    train_df.info()
    return


@app.cell
def _(train_df):
    train_df.drop(['cycle'], axis=1).describe().T
    return


@app.cell
def _(test_df, train_df):
    print('Train_df NUll Val: ',train_df.isnull().sum().sum())
    print('Test_df NUll Val: ',test_df.isnull().sum().sum())
    return


@app.cell
def _(calculate_train_rul, train_df):
    # Calculate RUL for training set before EDA checks
    train_df_rul = calculate_train_rul(train_df=train_df.copy())
    return (train_df_rul,)


@app.cell
def _(train_df_rul):
    # Sanity check: final row for each engine should have RUL = 0
    train_df_rul.groupby("unit_id")["RUL"].min().value_counts()
    return


@app.cell
def _(train_df_rul):
    # Index the training dataframe by 'unit_id' for easier access and analysis
    train_df_rul.set_index('unit_id', inplace=True)
    return


@app.cell
def _(train_df_rul):
    # Select only feature columns (excluding 'RUL') from the processed train_df.
    feature_cols = [col for col in train_df_rul.columns if col not in ['unit_id', 'cycle', 'RUL', 'op_cond_1', 'op_cond_2', 'op_cond_3']]
    return (feature_cols,)


@app.cell
def _(feature_cols, mo, plt, sns, train_df_rul):
    # Key Feature Distribution Analysis: Plot histograms and KDEs for all representative features to examine their individual distributions (skewness, modality).
    plots = []

    for column in feature_cols:
        if column in train_df_rul.columns:
            fig, ax = plt.subplots(figsize=(10, 4))

            sns.histplot(
                train_df_rul[column].dropna(),
                bins=30,
                kde=True,
                ax=ax,
            )

            ax.set_title(f"Distribution of Feature: {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

            plots.append(mo.mpl.interactive(fig))

    plots
    return


@app.cell
def _(feature_cols, plt, sns, train_df_rul):
    # Generate a heatmap to visualize pairwise correlations among all features for multicollinearity insight.
    features_df = train_df_rul[feature_cols]
    correlation_matrix = features_df.corr()
    plt.figure(figsize=(18, 16))
    # Calculate and plot the correlation matrix.
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Pairwise Feature Correlation Heatmap')
    plt.show()
    return


@app.cell
def _(feature_cols, pd, sm, train_df_rul):
    # Calculate Variance Inflation Factor (VIF) for each feature to assess multicollinearity.
    def calculate_vif(df):
        vif_dict = {}
        for col in df.columns:
            X = df.drop(columns=[col])
            X_const = sm.add_constant(X)  # add constant to the model
            model = sm.OLS(df[col], X_const).fit()
            r_sq = model.rsquared
            vif = 1 / (1 - r_sq) if 1 - r_sq != 0 else float('inf')
            vif_dict[col] = round(vif, 2)
        return vif_dict
    vif_results = calculate_vif(train_df_rul[feature_cols])
    vif_df = pd.Series(vif_results).to_frame(name='VIF')
    vif_df.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The VIF analysis shows that sensor_9 and sensor_14 are multicollinear with other sensors, meaning that they can be explained in terms of others.
    """)
    return


@app.cell
def _(feature_cols, test_df, train_df_rul):
    # Drop the least informative features based on the analysis
    sensors_to_drop = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    feature_cols_dropped = [col for col in feature_cols if col not in sensors_to_drop]
    train_df_rul.copy().drop(columns=sensors_to_drop, inplace=True) # Use copy() to avoid SettingWithCopyWarning in notebook context
    test_df.copy().drop(columns=sensors_to_drop, inplace=True) # Use copy() to avoid SettingWithCopyWarning in notebook context
    return (feature_cols_dropped,)


@app.cell
def _(plt, sns, train_df_rul):
    # Visualize the frequency distribution of Remaining Useful Life (RUL).
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df_rul['RUL'], bins=30, kde=True)
    plt.title('Frequency Distribution of Remaining Useful Life (RUL)')
    plt.xlabel('Remaining Useful Life (RUL)')
    plt.ylabel('Frequency')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The RUL distribution shows that the majority of engines have RUL of about 150 cycles or less with fewer extending towards 350+ cycles.
    """)
    return


@app.cell
def _(feature_cols, train_df_rul):
    # Sensor correlation analysis: Identify and visualize the correlation of each sensor with RUL to determine their predictive power.
    correlations = train_df_rul[feature_cols + ['RUL']].corr()['RUL'].abs().sort_values(ascending=False)
    correlations

    return


@app.cell
def _():
    return


@app.cell
def _(plt, sns, train_df_rul):
    # Sequence sanity check: how many cycles per engine?
    unit_lengths = train_df_rul.groupby("unit_id").size()

    print(unit_lengths.describe())
    print(f"Engines with at least 30 cycles: {(unit_lengths >= 30).sum()} / {len(unit_lengths)}")

    plt.figure(figsize=(8, 4))
    sns.histplot(unit_lengths, bins=30, kde=False)
    plt.title("Distribution of sequence lengths per engine")
    plt.xlabel("Number of cycles")
    plt.ylabel("Count")
    plt.show()
    return


@app.cell
def _(feature_cols_dropped, plt, train_df_rul):
    # Average sensor trajectories across all units

    ncols = 3
    nrows = len(feature_cols_dropped) // ncols + (len(feature_cols_dropped) % ncols > 0) # Create enough rows to accommodate all features

    fig_sensor_trajectory, axes = plt.subplots(nrows, ncols, figsize=(16, 12), sharex=True)
    axes_sensor_trajectory = axes.flatten()

    for ax_sensor_trajectory, sensor_trajectory_col in zip(axes_sensor_trajectory, feature_cols_dropped):
        # Mean and standard deviation for each cycle
        stats = (
            train_df_rul
            .groupby("RUL")[sensor_trajectory_col]
            .agg(["mean", "std"])
            .sort_index(ascending=False)   # False = failure on the right 
            .reset_index()
        )

        ax_sensor_trajectory.plot(
            stats["RUL"],
            stats["mean"],
            color="C0",
            linewidth=2,
            label="Mean"
        )

        ax_sensor_trajectory.set_xlim(stats["RUL"].max(), 0) # Invert x-axis to match failure on the right

        ax_sensor_trajectory.fill_between(
            stats["RUL"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color="C0",
            alpha=0.25,
            label="±1 SD"
        )

        ax_sensor_trajectory.set_title(sensor_trajectory_col)
        ax_sensor_trajectory.set_xlabel("RUL")
        ax_sensor_trajectory.set_ylabel("Value")
        ax_sensor_trajectory.grid(alpha=0.3)

    axes_sensor_trajectory[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Except for the sensor 6, the remaining sensors show clear trajectories towards failure values as RUL runs down to zero. This justifies using sequence models.
    """)
    return


@app.cell
def _(pd, test_df, train_df_rul):
    # Test for covariate shift between training and test datasets
    report = {}
    for col in train_df_rul.drop(columns=['RUL']).columns:
        train_col = train_df_rul[col]
        test_col = test_df[col]
        report[col] = {'Train Mean': train_col.mean(), 'Test Mean': test_col.mean(), 'Mean Difference': abs(train_col.mean() - test_col.mean()), 'Train Std': train_col.std(), 'Test Std': test_col.std(), 'Std Difference': abs(train_col.std() - test_col.std()), 'Train Min': train_col.min(), 'Test Min': test_col.min(), 'Train Max': train_col.max(), 'Test Max': test_col.max(), 'Test Out of Train Range': test_col.min() < train_col.min() or test_col.max() > train_col.max()}
    simple_report_df = pd.DataFrame(report).T
    simple_report_df  #location shift  #scale shift  #MinMax shift to detect the model extraplotation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some sensors have values in test set that are out train range. This is an issue to consider in model generalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Based on the documentation provided with the dataset, six different flight conditions were simulated that comprised of a range of values for three operational conditions: altitude (0-42K ft.), Mach number (0-0.84), and TRA (20-100).
    """)
    return


@app.cell
def _(train_df_rul):
    # Select operational condition columns for further analysis
    op_cond_cols = ['op_cond_1', 'op_cond_2', 'op_cond_3']
    train_df_op = train_df_rul[op_cond_cols].copy()
    return (train_df_op,)


@app.cell
def _(KMeans, train_df_op):
    # Cluster operating conditions columns. 
    kmeans = KMeans(n_clusters = 6,random_state = 42) # 6 clusters for the six different flight conditions in the dataset. This is based on domain knowledge  of the operational condition features.
    train_df_op['op_clusters']  = kmeans.fit_predict(train_df_op)
    #test_df['op_clusters'] = kmeans.predict(test_df[train_df_op])
    train_df_op.head(1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
