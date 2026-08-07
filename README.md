# NASA C-MAPSS-1 Turbofan Engine Degradation Prediction Project

## Overview

This project implements a machine learning solution for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS-1 Turbofan Engine Degradation dataset from Kaggle. The system employs LSTM (Long Short-Term Memory) neural networks to analyze multivariate time series sensor data and predict engine failure cycles.

## Dataset

The project is based on the **NASA C-MAPSS-1 Turbofan Engine Degradation Simulation** dataset available on Kaggle (dataset: `bishals098/nasa-turbofan-engine-degradation-simulation`).

### Dataset Description

The dataset consists of multivariate time series data from a fleet of turbofan engines. Each data set is divided into training and test subsets:

| Dataset | Train Trajectories | Test Trajectories | Conditions | Fault Modes |
|---------|-------------------|-------------------|------------|-------------|
| FD001   | 100               | 100               | ONE (Sea Level) | ONE (HPC Degradation) |
| FD002   | 260               | 259               | SIX        | ONE (HPC Degradation) |
| FD003   | 100               | 100               | ONE (Sea Level) | TWO (HPC Degradation, Fan Degradation) |
| FD004   | 248               | 249               | SIX        | TWO (HPC Degradation, Fan Degradation) |

### Data Structure

Each data set contains 26 columns representing:
1. Unit number
2. Time (in cycles)
3. Operational setting 1
4. Operational setting 2
5. Operational setting 3
6-26. Sensor measurements (21 sensor readings)

### Key Characteristics

- Each time series represents a different engine from a fleet
- Engines start with varying degrees of initial wear and manufacturing variation
- Three operational settings significantly affect engine performance
- Data contains sensor noise
- Engines operate normally initially and develop faults over time
- Training set: fault grows until system failure
- Test set: time series ends before system failure

---

## Installation & Setup

### Prerequisites

You must have Python 3.8+ installed on your system. We recommend using a virtual environment to isolate project dependencies.

### Step 1: Create `requirements.txt` (If missing)
Before installation, ensure you have a `requirements.txt` file in the root directory listing all necessary packages (e.g., pandas, numpy, tensorflow/pytorch). If this file is missing, create it and populate it with your project's dependencies.

### Step 2: Install Dependencies using pip
Once `requirements.txt` is ready, install all required packages into your active environment:

```bash
pip install -r requirements.txt
```

### Step 3: Creating a Python Environment with uv (Recommended)

`uv` is a fast and modern tool for managing Python environments. It helps keep dependencies isolated from your system interpreter. While other tools like `conda` or `virtualenv` exist, we recommend using `uv`.

1. **Install uv:**
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | sh
   ```
2. **Create a new environment (e.g., named 'venv'):**
   ```bash
   uv venv venv 
   ```
3. **Activate the environment:**
   *On Linux/macOS:* `source venv/bin/activate`
   *On Windows (PowerShell):* `.\venv\Scripts\Activate.ps1`

After activation, proceed to Step 2 above (`pip install -r requirements.txt`).

### Other Environment Tools

If you prefer different tools:
- **Conda:** `conda create -n myenv python=3.8` followed by `conda activate myenv`.
- **virtualenv:** `virtualenv venv` followed by `source venv/bin/activate`.

---
