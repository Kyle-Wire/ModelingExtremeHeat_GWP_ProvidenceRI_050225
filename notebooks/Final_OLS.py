import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
import os
import numpy as np

print("--- OLS Regression Analysis Script ---")

# --- Configuration ---
# Path to your primary data file (CSV or GPKG if using eigenmaps)
# Option 1: Original CSV (if not using eigenmaps or they are already in this CSV)
CSV_INPUT_PATH = 'with_two_form_indices.csv'

# Option 2: GeoPackage containing base predictors, target, and Laplacian Eigenmaps
# This is created by your Final_Spatial_Model_FOLD.py script
TRAINING_OUTPUT_DIR = "Comprehensive_Spatial_ML_Workflow_Final"
GEOPACKAGE_INPUT_PATH = os.path.join(TRAINING_OUTPUT_DIR, "data_prepped_with_laplacian_basis.gpkg")

# --- CHOOSE YOUR DATA SOURCE AND PREDICTORS ---
# Set to True to use the GeoPackage and include Laplacian Eigenmaps as predictors.
# Set to False to use the CSV_INPUT_PATH and only the BASE_PREDICTOR_VARIABLES.
USE_LAPLACIAN_EIGENMAPS_AS_PREDICTORS = False # USER CHOICE

TARGET_VARIABLE = 'AAT_z' # Your target variable

# Predictors to use if USE_LAPLACIAN_EIGENMAPS_AS_PREDICTORS is False
BASE_PREDICTOR_VARIABLES = [
    'Distance_from_water_m',
    'Pct_Impervious',
    'Pct_Canopy',
    'Pct_GreenSpace',
    'Elevation_m'
]

# Output file for the OLS summary
OLS_SUMMARY_OUTPUT_PATH = os.path.join(TRAINING_OUTPUT_DIR, "ols_summary_results.txt")

# --- Data Loading and Preparation ---
data = None
predictor_variables_for_ols = []

if USE_LAPLACIAN_EIGENMAPS_AS_PREDICTORS:
    print(f"Loading data from GeoPackage: {GEOPACKAGE_INPUT_PATH}")
    if not os.path.exists(GEOPACKAGE_INPUT_PATH):
        print(f"ERROR: GeoPackage file not found at '{GEOPACKAGE_INPUT_PATH}'. Please run the training script first or check path.")
        exit()
    data = gpd.read_file(GEOPACKAGE_INPUT_PATH)
    print(f"  Data loaded. Shape: {data.shape}")
    
    # Identify Laplacian Eigenmap columns (assuming they start with 'LAPEIG_')
    laplacian_cols = [col for col in data.columns if col.startswith('LAPEIG_')]
    if not laplacian_cols:
        print("WARNING: USE_LAPLACIAN_EIGENMAPS_AS_PREDICTORS is True, but no 'LAPEIG_' columns found in GeoPackage.")
    predictor_variables_for_ols.extend(BASE_PREDICTOR_VARIABLES)
    predictor_variables_for_ols.extend(laplacian_cols)
    print(f"  Using base predictors and {len(laplacian_cols)} Laplacian Eigenmaps.")
else:
    print(f"Loading data from CSV: {CSV_INPUT_PATH}")
    if not os.path.exists(CSV_INPUT_PATH):
        print(f"ERROR: CSV file not found at '{CSV_INPUT_PATH}'. Please check path.")
        exit()
    data = pd.read_csv(CSV_INPUT_PATH)
    print(f"  Data loaded. Shape: {data.shape}")
    predictor_variables_for_ols.extend(BASE_PREDICTOR_VARIABLES)
    print(f"  Using base predictors only.")

# Ensure all selected columns exist
all_cols_for_ols = [TARGET_VARIABLE] + predictor_variables_for_ols
missing_cols = [col for col in all_cols_for_ols if col not in data.columns]
if missing_cols:
    print(f"ERROR: The following columns are missing from the loaded data: {missing_cols}")
    print(f"Available columns: {data.columns.tolist()}")
    exit()

# Prepare data for statsmodels
# Convert relevant columns to numeric, coercing errors
for col in all_cols_for_ols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN in target or predictors
data_for_ols = data[all_cols_for_ols].dropna()

if data_for_ols.empty:
    print("ERROR: No data remaining after dropping NaNs from target and predictor columns. Cannot run OLS.")
    exit()

print(f"  Shape of data for OLS after NaN removal: {data_for_ols.shape}")

# Define target (y) and predictors (X)
y = data_for_ols[TARGET_VARIABLE]
X = data_for_ols[predictor_variables_for_ols]

# Add a constant (intercept) to the predictors
# OLS in statsmodels doesn't add an intercept by default
X_with_constant = sm.add_constant(X)

# --- Run OLS Regression ---
print("\n--- Running OLS Regression ---")
try:
    ols_model = sm.OLS(y, X_with_constant)
    ols_results = ols_model.fit()

    # Print the OLS summary
    print("\nOLS Regression Results Summary:")
    print(ols_results.summary())

    # Save the summary to a file
    if not os.path.exists(TRAINING_OUTPUT_DIR): # Ensure output directory exists
        os.makedirs(TRAINING_OUTPUT_DIR)
        
    with open(OLS_SUMMARY_OUTPUT_PATH, 'w') as f:
        f.write(str(ols_results.summary()))
    print(f"\nOLS summary saved to: {OLS_SUMMARY_OUTPUT_PATH}")

    print("\n--- OLS Analysis Complete ---")

except Exception as e:
    print(f"An error occurred during OLS regression: {e}")
    import traceback
    traceback.print_exc()