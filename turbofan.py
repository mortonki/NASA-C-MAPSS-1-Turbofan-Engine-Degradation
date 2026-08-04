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
    from data_prep import load_data, calculate_train_rul, normalize_data, apply_outlier_capping, create_sliding_windows

    return (
        Path,
        calculate_train_rul,
        kagglehub,
        load_data,
        matplotlib,
        mo,
        pd,
        plt,
        sm,
        sns,
    )


@app.cell
def _(matplotlib, mo):
    # Check the current matplotlib backend
    print(matplotlib.get_backend())

    # Check the version of marimo
    print(mo.__version__)
    return


@app.cell
def _(kagglehub):
    # Download latest version
    base_path = kagglehub.dataset_download("bishals098/nasa-turbofan-engine-degradation-simulation")

    print(f"Path to dataset files: {base_path}")
    return (base_path,)


@app.cell
def _(Path, base_path):
    # Convert the string path to a Path object
    dataset_dir = Path(base_path)
    required_files = {'train': 'train_FD001.txt', 'test': 'test_FD001.txt', 'label': 'RUL_FD001.txt'}
    # Define and verify required files
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
    return (dataset_dir,)


@app.cell
def _(dataset_dir):
    # List all files in the directory
    for file in dataset_dir.iterdir():
        print(file.name)
    return


@app.cell
def _(dataset_dir):
    # Define file paths
    _train_file = dataset_dir / 'train_FD001.txt'
    _test_file = dataset_dir / 'test_FD001.txt'
    _label_file = dataset_dir / 'RUL_FD001.txt'
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
    train_df.describe().T
    return


@app.cell
def _(test_df, train_df):
    print('Train_df NUll Val: ',train_df.isnull().sum().sum())
    print('Test_df NUll Val: ',test_df.isnull().sum().sum())
    return


@app.cell
def _(train_df):
    train_df.set_index('unit_id', inplace=True)
    return


@app.cell
def _(calculate_train_rul, train_df):
    # Calculate RUL for training set as per original notebook intent before EDA checks
    train_df_rul = calculate_train_rul(train_df=train_df.copy())
    return (train_df_rul,)


@app.cell
def _(train_df_rul):
    # Select only feature columns (excluding 'RUL') from the processed train_df.
    feature_cols = [col for col in train_df_rul.columns if col not in ['RUL', 'unit_id', 'cycle']]
    return (feature_cols,)


@app.cell
def _(feature_cols, mo, plt, sns, train_df_rul):
    # Key Feature Distribution Analysis: Plot histograms and KDEs for all representative features to examine their individual distributions (skewness, modality).
    print("Generating feature distribution plots...")

    plots = []

    for column in feature_cols:
        if column in train_df_rul.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

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
    features_df = train_df_rul[feature_cols]  # Exclude 'RUL' and 'unit_id' from features
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
    vif_df = pd.DataFrame.from_dict(vif_results, orient='index', columns=['VIF'])
    print('\nVIF DataFrame:')
    vif_df.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The VIF analysis shows that sensor_9 and sensor_14 are multicollinear with other sensors, meaning that they can be explained in terms of others.
    """)
    return


@app.cell
def _(test_df, train_df_rul):
    # Drop least informative features based on correlation analysis
    sensors_to_drop = ['op_cond_1', 'op_cond_2', 'op_cond_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    train_df_rul.copy().drop(columns=sensors_to_drop, inplace=True) # Use copy() to avoid SettingWithCopyWarning in notebook context
    test_df.copy().drop(columns=sensors_to_drop, inplace=True) # Use copy() to avoid SettingWithCopyWarning in notebook context
    return


@app.cell
def _(plt, sns, train_df_rul):
    # Visualize the frequency distribution of Remaining Useful Life (RUL).
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df_rul['RUL'], bins=30, kde=True)  # Assuming train_df is available and contains 'RUL' column 
    plt.title('Frequency Distribution of Remaining Useful Life (RUL)')
    plt.xlabel('Remaining Useful Life (RUL)')
    plt.ylabel('Frequency')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The RUL distribution shows that the majority of engines have RUL of about 150 cycles or less with fewer extending towards 350 cycles. The spike at 0 cycles is because some sequences are shorter than the time window.
    """)
    return


@app.cell
def _():
    # Dynamically determine features to drop based on low correlation with RUL (mimicking previous logic but making it robust)
    #correlations = train_df_1.drop(['cycle', 'unit_id'], axis=1).corr()['RUL'].abs().sort_values(ascending=False)
    #least_informative = correlations.tail(10).index.tolist() # Get the names of the 10 least informative features
    #print(f"Dropping least informative features: {least_informative}")

    # Apply dropping only if columns exist in train_df_1 (and assuming test_df_1 is available/passed correctly in execution flow)
    #cols_to_drop = [col for col in least_informative if col in train_df_1.columns]
    #train_df_1.drop(columns=cols_to_drop, inplace=True)
    # NOTE: In a real Marimo environment, we would need to pass test_df_1 here too. For this fix, I focus on making the logic robust for train_df_1 based on available context.
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
def _(feature_cols, plt, train_df_rul):
    # Average sensor trajectories across all units

    ncols = 3
    nrows = len(feature_cols) // ncols + (len(feature_cols) % ncols > 0)

    fig_sensor_trajectory, axes = plt.subplots(nrows, ncols, figsize=(16, 12), sharex=True)
    axes_sensor_trajectory = axes.flatten()

    for ax_sensor_trajectory, sensor_trajectory_col in zip(axes_sensor_trajectory, feature_cols):
        # Mean and standard deviation for each cycle
        stats = (
            train_df_rul
            .groupby("RUL")[sensor_trajectory_col]
            .agg(["mean", "std"])
            .sort_index(ascending=False)   # failure on the right
            .reset_index()
        )

        ax_sensor_trajectory.plot(
            stats["RUL"],
            stats["mean"],
            color="C0",
            linewidth=2,
            label="Mean"
        )

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
def _(feature_cols, pd, sample_units, train_df_rul):
    # Quantitative sequence trajectory check: summarize how selected sensors change over the life of a few units

    trajectory_summary = []
    for sample_unit in sample_units:
        _unit_df = train_df_rul[train_df_rul["unit_id"] == sample_unit].sort_values("cycle")
        for sensor in feature_cols:
            start_val = _unit_df[sensor].iloc[0]
            end_val = _unit_df[sensor].iloc[-1]
            delta_val = end_val - start_val
            mean_step = _unit_df[sensor].diff().dropna().mean() if len(_unit_df) > 1 else 0.0
            corr = _unit_df[sensor].corr(_unit_df["cycle"])
            trajectory_summary.append({
                "unit_id": sample_unit,
                "sensor": sensor,
                "start": start_val,
                "end": end_val,
                "delta": delta_val,
                "mean_step": mean_step,
                "cycle_corr": corr,
            })

    trajectory_summary_df = pd.DataFrame(trajectory_summary)
    trajectory_summary_df = trajectory_summary_df.sort_values(["unit_id", "sensor"]).reset_index(drop=True)
    trajectory_summary_df
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
    So sensors have values in test set that are out train range. This is an issue to consider in model generalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
