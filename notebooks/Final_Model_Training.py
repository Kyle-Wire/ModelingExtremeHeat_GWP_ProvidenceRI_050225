# Final_Model_Training_v2_skippable_enhanced.py
#
# PURPOSE:
# 1. Train all base and meta-models on the ENTIRE dataset.
# 2. Extract and save analytical outputs (GWR Coefficients, GAM plots via R).
# 3. Calculate and save performance metrics (R2, RMSE, Moran's I) on the full training data.
# 4. Save all final model components (scalers, models, parameters) for use
#    in the Phase 3 prediction script.
# 5. Add comprehensive skip logic for stages if outputs exist and OVERWRITE_EXISTING_OUTPUTS is False.

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import joblib
import subprocess
import json
import matplotlib.pyplot as plt
import tempfile
import shutil

# Preprocessing & Feature Engineering
from sklearn.preprocessing import StandardScaler
from libpysal.weights import KNN as PysalKNN
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

# Base Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import statsmodels.api as sm

# Meta-Learner & Refinement
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import shap

# Deep Kriging (Neural Network)
import tensorflow as tf
from keras.models import Sequential # Using Keras standalone namespace as in user's script
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping

# Moran's I for residuals
from esda.moran import Moran

print("--- Phase 2: Final Model Training & Analysis (v2 - Skippable Stages Enhanced) ---")

# --- Configuration ---
CSV_PATH = 'with_two_form_indices.csv'
TARGET = 'AAT_z'
COORDS_COLS = ['POINT_X', 'POINT_Y']
IDENTIFIER_COL = 'OBJECTID'
INITIAL_CRS = "EPSG:3438"
TARGET_PROJECTED_CRS = "EPSG:26919"

BASE_MODEL_PREDICTORS = ['Distance_from_water_m', 'Pct_Impervious', 'Pct_Canopy', 'Pct_GreenSpace', 'Elevation_m']
print(f"Using predictors for base models: {BASE_MODEL_PREDICTORS}")

INCLUDE_GWRF_IN_ENSEMBLE = True # Set to False to exclude GWRF from the final ensemble
OVERWRITE_EXISTING_OUTPUTS = False # <<<< SET THIS TO TRUE FOR A FULL RE-RUN OF ALL STAGES

CV_OUTPUT_DIR = "Comprehensive_Spatial_ML_Workflow_Final" # From your previous successful CV run (or V7)
OUTPUT_DIR_FULL_MODEL = "Final_Model_Outputs_Trained_All_Data"
if not os.path.exists(OUTPUT_DIR_FULL_MODEL):
    os.makedirs(OUTPUT_DIR_FULL_MODEL)
print(f"All final trained model outputs will be saved in: {os.path.join(os.getcwd(), OUTPUT_DIR_FULL_MODEL)}")



# Define ablation suffix for file naming and reporting
ABLATION_SUFFIX = "" if INCLUDE_GWRF_IN_ENSEMBLE else "_no_gwrf"

PREPPED_DATA_FROM_CV_PATH = os.path.join(CV_OUTPUT_DIR, "data_prepped_with_laplacian_basis.gpkg") # From V7 CV
PREPPED_DATA_FOR_FULL_RUN_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "final_data_prepped_with_basis_full_run.gpkg")

GWR_FULL_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "gwr_full_data_predictions.csv")
GWR_COEFFICIENTS_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "gwr_final_local_coefficients.gpkg")
GWR_BANDWIDTH_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "gwr_final_bandwidth.json")
GWR_PERFORMANCE_FULL_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "gwr_performance_on_full_data.csv")

GWRF_FULL_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "gwrf_full_data_predictions.csv")
GWRF_PERFORMANCE_FULL_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "gwrf_performance_on_full_data.csv")

GGPGAMSVC_R_MODEL_OBJECT_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "gam_svc_final_model.rds")
GGPGAMSVC_FULL_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "ggpgamsvc_full_data_predictions.csv")
GGPGAMSVC_PERFORMANCE_FULL_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "ggpgamsvc_performance_on_full_data.csv")
GAM_PLOTS_DIR = os.path.join(OUTPUT_DIR_FULL_MODEL, "gam_final_plots")
if not os.path.exists(GAM_PLOTS_DIR): os.makedirs(GAM_PLOTS_DIR)

PATH_TO_GGPGAMSVC_R_FULL_SCRIPT = "Final_GGPGAM_Full.r"  # R script for full data
# Try to locate the R executable in the current environment for portability.
R_EXECUTABLE = shutil.which("R") or r"C:\Users\KWire\AppData\Local\Programs\R\R-4.5.0\bin\x64\R.exe"

BASE_PREDICTORS_SCALER_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, 'final_base_predictors_scaler.pkl')
LGBM_META_MODEL_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, 'final_lgbm_meta_model_trained_all_data.pkl')
# Try to load tuned params from the CV run that corresponds to INCLUDE_GWRF_IN_ENSEMBLE setting
cv_ablation_suffix = "" if INCLUDE_GWRF_IN_ENSEMBLE else "_no_gwrf"
TUNED_LGBM_PARAMS_PATH = os.path.join(CV_OUTPUT_DIR, f'tuned_lgbm_params{cv_ablation_suffix}.json')
LGBM_FINAL_PREDICTIONS_FULL_DATA_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "lgbm_final_predictions_full_data.csv")
meta_feature_order_save_path = os.path.join(OUTPUT_DIR_FULL_MODEL, f'final_lgbm_meta_feature_order.json') # Define path for feature order

DK_SCALER_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, 'final_dk_scaler_trained_all_data.pkl')
DK_MODEL_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, 'final_deep_kriging_model_trained_all_data.keras')
DK_CORRECTIONS_FULL_DATA_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "dk_corrections_full_data.csv")

FINAL_MODEL_PERFORMANCE_FULL_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "final_combined_model_performance_on_full_data.csv")
FINAL_MODEL_RESIDUAL_MORANS_FULL_PATH = os.path.join(OUTPUT_DIR_FULL_MODEL, "final_combined_model_residual_morans_i_full_data.csv")
K_FOR_MORAN_SWM = 100
N_LAPLACIAN_EIGENMAPS = 150
K_FOR_LAPLACIAN_SWM_PREPROC = 10

# --- Stage 1: Data Preprocessing and Laplacian Eigenmaps (on Full Dataset) ---
print(f"\n--- [1] Preprocessing Full Dataset ---")
data_for_modeling = None
LAPLACIAN_COLS = []

if not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(PREPPED_DATA_FOR_FULL_RUN_PATH):
    print(f"    Loading existing preprocessed data for full run from {PREPPED_DATA_FOR_FULL_RUN_PATH}")
    data_for_modeling = gpd.read_file(PREPPED_DATA_FOR_FULL_RUN_PATH)
    LAPLACIAN_COLS = [col for col in data_for_modeling.columns if col.startswith('LAPEIG_')]
elif not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(PREPPED_DATA_FROM_CV_PATH):
    print(f"    Loading existing preprocessed data from CV run: {PREPPED_DATA_FROM_CV_PATH}")
    print(f"    Saving it to new location for this full run: {PREPPED_DATA_FOR_FULL_RUN_PATH}")
    data_for_modeling = gpd.read_file(PREPPED_DATA_FROM_CV_PATH)
    LAPLACIAN_COLS = [col for col in data_for_modeling.columns if col.startswith('LAPEIG_')]
    if data_for_modeling.index.name == IDENTIFIER_COL: data_for_modeling = data_for_modeling.reset_index()
    elif IDENTIFIER_COL not in data_for_modeling.columns: data_for_modeling[IDENTIFIER_COL] = data_for_modeling.index
    data_for_modeling.to_file(PREPPED_DATA_FOR_FULL_RUN_PATH, driver="GPKG", layer="data_final_full_run")
else:
    print(f"    Generating preprocessed data from raw CSV: {CSV_PATH}...")
    # (Full preprocessing and Laplacian Eigenmap generation code from your script)
    try: data = pd.read_csv(CSV_PATH, low_memory=False)
    except FileNotFoundError: print(f"ERROR: Raw CSV file not found at {CSV_PATH}."); exit()
    essential_cols = COORDS_COLS + BASE_MODEL_PREDICTORS + [TARGET, IDENTIFIER_COL]
    for col in essential_cols:
        if col not in data.columns: print(f"ERROR: Column '{col}' not found in CSV."); exit()
        if col in COORDS_COLS + [TARGET] + BASE_MODEL_PREDICTORS: data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=essential_cols, inplace=True)
    if data.empty: print("ERROR: Data empty after NA drop."); exit()
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[COORDS_COLS[0]], data[COORDS_COLS[1]]), crs=INITIAL_CRS).to_crs(TARGET_PROJECTED_CRS)
    gdf['projected_X'] = gdf.geometry.x; gdf['projected_Y'] = gdf.geometry.y
    print(f"    Generating {N_LAPLACIAN_EIGENMAPS} Laplacian Eigenmaps using k={K_FOR_LAPLACIAN_SWM_PREPROC} for SWM...")
    coords_for_laplacian = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    try:
        w_laplacian = PysalKNN.from_array(coords_for_laplacian, k=K_FOR_LAPLACIAN_SWM_PREPROC)
        L = laplacian(w_laplacian.sparse, normed=True)
        k_eigsh = min(N_LAPLACIAN_EIGENMAPS + 1, L.shape[0] - 2)
        if k_eigsh >= 2:
            vals, vecs = eigsh(L, k=k_eigsh, which='SM')
            eigenmaps_to_add = vecs[:, 1:min(vecs.shape[1], N_LAPLACIAN_EIGENMAPS + 1)]
            LAPLACIAN_COLS = [f'LAPEIG_{i+1}' for i in range(eigenmaps_to_add.shape[1])]
            eigenmap_df = pd.DataFrame(eigenmaps_to_add, columns=LAPLACIAN_COLS, index=gdf.index)
            data_for_modeling = gdf.join(eigenmap_df)
        else: data_for_modeling = gdf.copy(); LAPLACIAN_COLS = []
    except Exception as e: print(f"    Error generating Laplacian Eigenmaps: {e}."); data_for_modeling = gdf.copy(); LAPLACIAN_COLS = []
    if data_for_modeling.index.name == IDENTIFIER_COL: data_for_modeling = data_for_modeling.reset_index()
    elif IDENTIFIER_COL not in data_for_modeling.columns: data_for_modeling[IDENTIFIER_COL] = data_for_modeling.index
    data_for_modeling.to_file(PREPPED_DATA_FOR_FULL_RUN_PATH, driver="GPKG", layer="data_final_full_run")
    print(f"    Full dataset preprocessed. Shape: {data_for_modeling.shape}")
    print(f"    Saved prepped data with eigenmaps to: {PREPPED_DATA_FOR_FULL_RUN_PATH}")

if data_for_modeling is None or data_for_modeling.empty: print("ERROR: data_for_modeling is empty or None after Stage 1."); exit()
print(f"--- Stage 1 Processing Complete ---")

# --- Stage 2: Train Base Models on Full Dataset & Generate Meta-Features ---
print(f"\n--- [2] Training Base Models on Full Dataset ---")
data_for_modeling.dropna(subset=[TARGET] + BASE_MODEL_PREDICTORS, inplace=True) # Ensure no NaNs before training
if data_for_modeling.empty: print("ERROR: Data empty after NA drop before Stage 2."); exit()

gwr_full_predictions = pd.Series(np.nan, index=data_for_modeling.index, name='gwr_full_pred')
gwrf_full_predictions = pd.Series(np.nan, index=data_for_modeling.index, name='gwrf_full_pred')
ggpgamsvc_full_predictions = pd.Series(np.nan, index=data_for_modeling.index, name='ggpgamsvc_full_pred')
scaler_base = None

# --- [2.1] Geographically Weighted Regression (GWR) ---
print("\n  --- [2.1] GWR: Training, Extracting Coefficients, and Predicting ---")
gwr_stage_files_to_check = [GWR_FULL_PREDICTIONS_PATH, GWR_COEFFICIENTS_PATH, GWR_BANDWIDTH_PATH, BASE_PREDICTORS_SCALER_PATH, GWR_PERFORMANCE_FULL_PATH]
gwr_stage_outputs_exist = all(os.path.exists(f) for f in gwr_stage_files_to_check)

if not OVERWRITE_EXISTING_OUTPUTS and gwr_stage_outputs_exist:
    print(f"    Skipping GWR full model training. Loading existing outputs.")
    gwr_preds_df = pd.read_csv(GWR_FULL_PREDICTIONS_PATH)
    if IDENTIFIER_COL in gwr_preds_df.columns and 'gwr_full_pred' in gwr_preds_df.columns:
        gwr_full_predictions = data_for_modeling[IDENTIFIER_COL].map(pd.Series(gwr_preds_df['gwr_full_pred'].values, index=gwr_preds_df[IDENTIFIER_COL])).reindex(data_for_modeling.index).rename('gwr_full_pred')
    scaler_base = joblib.load(BASE_PREDICTORS_SCALER_PATH)
else:
    print("    Proceeding with GWR full model training...")
    # (Full GWR training logic - from your script)
    gwr_predictors_list = BASE_MODEL_PREDICTORS
    X_gwr_full_df_train = data_for_modeling[gwr_predictors_list]
    y_gwr_full_series_train = data_for_modeling[TARGET]
    coords_full_train = np.array(list(zip(data_for_modeling.geometry.x, data_for_modeling.geometry.y)))
    scaler_base = StandardScaler(); X_gwr_full_scaled_train = scaler_base.fit_transform(X_gwr_full_df_train)
    joblib.dump(scaler_base, BASE_PREDICTORS_SCALER_PATH); print(f"    Saved base predictors scaler: {BASE_PREDICTORS_SCALER_PATH}")
    try:
        gwr_selector_full = Sel_BW(coords_full_train, y_gwr_full_series_train.values.reshape(-1,1), X_gwr_full_scaled_train, kernel='gaussian', fixed=False, spherical=False)
        bw_gwr_val = gwr_selector_full.search(search_method='golden_section', criterion='AICc')
        min_adaptive_k_gwr = 500; bw_gwr_int = max(min_adaptive_k_gwr, X_gwr_full_scaled_train.shape[1] + 2, int(bw_gwr_val))
        bw_gwr_int = min(bw_gwr_int, len(X_gwr_full_scaled_train) -1 if len(X_gwr_full_scaled_train) >1 else 1); bw_gwr_int = max(1, bw_gwr_int)
        with open(GWR_BANDWIDTH_PATH, 'w') as f: json.dump({'adaptive_k': bw_gwr_int}, f)
        gwr_model_full = GWR(coords_full_train, y_gwr_full_series_train.values.reshape(-1,1), X_gwr_full_scaled_train, bw=bw_gwr_int, kernel='gaussian', fixed=False, spherical=False)
        gwr_results_full = gwr_model_full.fit()
        coeff_labels = ['coeff_intercept'] + [f'coeff_{p}' for p in gwr_predictors_list]
        coefficients_df_temp = pd.DataFrame(gwr_results_full.params, columns=coeff_labels, index=X_gwr_full_df_train.index)
        gwr_coeffs_gdf = data_for_modeling.loc[X_gwr_full_df_train.index, [IDENTIFIER_COL, 'geometry']].join(coefficients_df_temp)
        gwr_coeffs_gdf.to_file(GWR_COEFFICIENTS_PATH, driver='GPKG'); print(f"    GWR local coefficients saved: {GWR_COEFFICIENTS_PATH}")
        temp_preds = pd.Series(gwr_results_full.predy.flatten(), index=X_gwr_full_df_train.index)
        gwr_full_predictions.update(temp_preds)
        if gwr_full_predictions.loc[X_gwr_full_df_train.index].notna().sum() >=2:
            r2_gwr_full = r2_score(y_gwr_full_series_train, gwr_full_predictions.loc[X_gwr_full_df_train.index].dropna()); rmse_gwr_full = np.sqrt(mean_squared_error(y_gwr_full_series_train, gwr_full_predictions.loc[X_gwr_full_df_train.index].dropna()))
            print(f"    GWR In-Sample Performance: R2 = {r2_gwr_full:.4f}, RMSE = {rmse_gwr_full:.4f}")
            pd.DataFrame([{'model':'GWR_FullData', 'R2':r2_gwr_full, 'RMSE':rmse_gwr_full}]).to_csv(GWR_PERFORMANCE_FULL_PATH, index=False)
    except Exception as e: print(f"    ERROR during GWR full model training: {e}"); gwr_full_predictions.loc[X_gwr_full_df_train.index] = np.nan
    pd.DataFrame({IDENTIFIER_COL: data_for_modeling.loc[gwr_full_predictions.index, IDENTIFIER_COL], 'gwr_full_pred': gwr_full_predictions}).to_csv(GWR_FULL_PREDICTIONS_PATH, index=False)
    print(f"    GWR predictions on full dataset saved to {GWR_FULL_PREDICTIONS_PATH}")

if scaler_base is None: # If GWR was skipped, try to load scaler
    if os.path.exists(BASE_PREDICTORS_SCALER_PATH): scaler_base = joblib.load(BASE_PREDICTORS_SCALER_PATH)
    else: print("CRITICAL ERROR: scaler_base not available and GWR was skipped. GWRF may fail."); exit()


# --- [2.2] Geographically Weighted Random Forest (GWRF) ---
# (Function definition `predict_gwrf_on_full_data` from your script is assumed here)
def predict_gwrf_on_full_data(all_training_data_gdf, points_to_predict_gdf, predictors_list, target_col_name, scaler_to_use, bandwidth_k=50, adaptive=True, rf_n_estimators=100, rf_min_samples_leaf=10, rf_max_features='sqrt', n_jobs=1):
    train_coords = all_training_data_gdf[['projected_X', 'projected_Y']].values
    predict_coords = points_to_predict_gdf[['projected_X', 'projected_Y']].values
    X_train_scaled_df = pd.DataFrame(scaler_to_use.transform(all_training_data_gdf[predictors_list]), columns=predictors_list, index=all_training_data_gdf.index)
    y_train_series = all_training_data_gdf[target_col_name]
    ball_tree = BallTree(train_coords, metric='euclidean')
    final_preds_gwrf = pd.Series(np.nan, index=points_to_predict_gdf.index)
    print(f"    Predicting for {len(points_to_predict_gdf)} points using GWRF (k={bandwidth_k})...")
    for i in range(len(points_to_predict_gdf)):
        current_predict_index = points_to_predict_gdf.index[i]; point_coords_current = predict_coords[i].reshape(1, -1)
        distances_from_point, indices_from_query = ball_tree.query(point_coords_current, k=len(train_coords))
        distances_from_point = distances_from_point[0]; indices_from_query = indices_from_query[0]
        neighbor_original_indices = all_training_data_gdf.index[indices_from_query]
        mask_not_self = (neighbor_original_indices != current_predict_index)
        local_distances = distances_from_point[mask_not_self]; local_training_indices_pos = indices_from_query[mask_not_self]
        if len(local_distances) == 0: final_preds_gwrf.loc[current_predict_index] = np.nan; continue
        local_bw_dist = 0.0
        if adaptive:
            k_for_bw = int(min(bandwidth_k, len(local_distances) -1)); k_for_bw = max(0, k_for_bw)
            sorted_local_distances = np.sort(local_distances)
            if k_for_bw < len(sorted_local_distances): local_bw_dist = sorted_local_distances[k_for_bw]
            elif sorted_local_distances.size > 0: local_bw_dist = sorted_local_distances[-1]
        else: local_bw_dist = bandwidth_k
        if local_bw_dist == 0: weights = np.zeros_like(local_distances)
        else: # Using Gaussian Kernel as per user's original Final_Model_Training.py
            with np.errstate(divide='ignore', invalid='ignore'): weights = np.exp(-(local_distances**2) / (2 * (local_bw_dist**2)))
            weights[local_distances >= local_bw_dist*3] = 0.0 # Truncation for Gaussian
        weights[np.isnan(weights)] = 0.0
        sum_weights = np.sum(weights); nz_weights = np.sum(weights > 1e-9)
        if i % 1000 == 0: print(f"      GWRF processing point {i}/{len(points_to_predict_gdf)}: ID {current_predict_index}, LocalBW={local_bw_dist:.2f}, NZ_W={nz_weights}")
        if sum_weights < 1e-6 or nz_weights < rf_min_samples_leaf: final_preds_gwrf.loc[current_predict_index] = np.nan; continue
        local_X_train_rf = X_train_scaled_df.iloc[local_training_indices_pos]; local_y_train_rf = y_train_series.iloc[local_training_indices_pos]
        try:
            rf = RandomForestRegressor(n_estimators=rf_n_estimators, min_samples_leaf=rf_min_samples_leaf, random_state=42, n_jobs=n_jobs, max_features=rf_max_features)
            # CORRECTED LINE: sample_weight should be `weights`, not `weights[mask_not_self]` as weights is already correct length
            rf.fit(local_X_train_rf, local_y_train_rf, sample_weight=weights)
            pred = rf.predict(X_train_scaled_df.loc[[current_predict_index]])[0]
            final_preds_gwrf.loc[current_predict_index] = pred
        except Exception as e_rf: print(f"        Error in GWRF for point {current_predict_index} (Index Pos {i}): {e_rf}"); final_preds_gwrf.loc[current_predict_index] = np.nan
    return final_preds_gwrf

print("\n  --- [2.2] GWRF: Generating In-Sample Predictions (Full Data) ---")
gwrf_stage_outputs_exist = (not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(GWRF_FULL_PREDICTIONS_PATH) and os.path.exists(GWRF_PERFORMANCE_FULL_PATH))

if not INCLUDE_GWRF_IN_ENSEMBLE:
    print("    Skipping GWRF full data prediction as INCLUDE_GWRF_IN_ENSEMBLE is False.")
    gwrf_full_predictions.loc[:] = 0 # Or another appropriate placeholder if GWRF is excluded
elif gwrf_stage_outputs_exist:
    print(f"    Skipping GWRF full model prediction. Loading existing outputs from {GWRF_FULL_PREDICTIONS_PATH}.")
    gwrf_preds_df = pd.read_csv(GWRF_FULL_PREDICTIONS_PATH)
    if IDENTIFIER_COL in gwrf_preds_df.columns and 'gwrf_full_pred' in gwrf_preds_df.columns:
        gwrf_full_predictions = data_for_modeling[IDENTIFIER_COL].map(pd.Series(gwrf_preds_df['gwrf_full_pred'].values, index=gwrf_preds_df[IDENTIFIER_COL])).reindex(data_for_modeling.index).rename('gwrf_full_pred')
else:
    print("    Proceeding with GWRF full data prediction...")
    try:
        gwrf_full_predictions = predict_gwrf_on_full_data(data_for_modeling, data_for_modeling, BASE_MODEL_PREDICTORS, TARGET, scaler_base, bandwidth_k=50, rf_n_estimators=100, rf_min_samples_leaf=10)
        if gwrf_full_predictions.notna().sum() >=2:
            valid_indices_gwrf = data_for_modeling[TARGET].index.intersection(gwrf_full_predictions.dropna().index)
            r2_gwrf_full = r2_score(data_for_modeling.loc[valid_indices_gwrf, TARGET], gwrf_full_predictions.loc[valid_indices_gwrf])
            rmse_gwrf_full = np.sqrt(mean_squared_error(data_for_modeling.loc[valid_indices_gwrf, TARGET], gwrf_full_predictions.loc[valid_indices_gwrf]))
            print(f"    GWRF In-Sample Performance: R2 = {r2_gwrf_full:.4f}, RMSE = {rmse_gwrf_full:.4f}")
            pd.DataFrame([{'model':'GWRF_FullData', 'R2':r2_gwrf_full, 'RMSE':rmse_gwrf_full}]).to_csv(GWRF_PERFORMANCE_FULL_PATH, index=False)
    except Exception as e_gwrf_full: print(f"    ERROR during GWRF full data prediction: {e_gwrf_full}")
    pd.DataFrame({IDENTIFIER_COL: data_for_modeling[IDENTIFIER_COL], 'gwrf_full_pred': gwrf_full_predictions}).to_csv(GWRF_FULL_PREDICTIONS_PATH, index=False)
    print(f"    GWRF predictions on full dataset saved to {GWRF_FULL_PREDICTIONS_PATH}")


# --- [2.3] GGP-GAM-SVC (via R) ---
print("\n  --- [2.3] GGP-GAM-SVC: Training, Plotting, and Predicting via R (Full Data) ---")
ggpgamsvc_stage_outputs_exist = (not OVERWRITE_EXISTING_OUTPUTS and
                                 os.path.exists(GGPGAMSVC_FULL_PREDICTIONS_PATH) and
                                 # Check for the .rds model object directly in OUTPUT_DIR_FULL_MODEL
                                 os.path.exists(os.path.join(OUTPUT_DIR_FULL_MODEL, "gam_svc_final_model.rds")) and
                                 os.path.exists(GGPGAMSVC_PERFORMANCE_FULL_PATH))

if ggpgamsvc_stage_outputs_exist:
    print(f"    Skipping GGP-GAM-SVC full model training. Loading existing predictions from {GGPGAMSVC_FULL_PREDICTIONS_PATH}.")
    ggpgamsvc_preds_df = pd.read_csv(GGPGAMSVC_FULL_PREDICTIONS_PATH)
    # Attempt to determine the prediction column name from common patterns
    pred_col_r_load = None
    if 'ggpgamsvc_pred' in ggpgamsvc_preds_df.columns:
        pred_col_r_load = 'ggpgamsvc_pred'
    elif 'ggpgamsvc_oof_prediction' in ggpgamsvc_preds_df.columns: # Fallback from potential OOF naming
        pred_col_r_load = 'ggpgamsvc_oof_prediction'
    
    if IDENTIFIER_COL in ggpgamsvc_preds_df.columns and pred_col_r_load:
        id_to_pred_map = pd.Series(ggpgamsvc_preds_df[pred_col_r_load].values, index=ggpgamsvc_preds_df[IDENTIFIER_COL])
        ggpgamsvc_full_predictions = data_for_modeling[IDENTIFIER_COL].map(id_to_pred_map).reindex(data_for_modeling.index).rename('ggpgamsvc_full_pred')
        print(f"      Loaded GGP-GAM-SVC full data predictions.")
    else:
        print(f"    WARNING: Could not load GGP-GAM-SVC full predictions from {GGPGAMSVC_FULL_PREDICTIONS_PATH} due to missing columns. Will attempt to regenerate if OVERWRITE is True.")
        if OVERWRITE_EXISTING_OUTPUTS: ggpgamsvc_stage_outputs_exist = False # Force regeneration
        else: ggpgamsvc_full_predictions.loc[:] = np.nan # Ensure it's NaN if load fails and not overwriting
else:
    print("    Proceeding with GGP-GAM-SVC full model training via R...")
    full_data_for_r_path = os.path.join(OUTPUT_DIR_FULL_MODEL, "temp_full_data_for_r_ggpgamsvc.csv")
    
    # Ensure all necessary columns are present for R script
    # R script will look for TARGET ('AAT_z') within this file.
    cols_for_final_r = list(set([IDENTIFIER_COL, TARGET, 'projected_X', 'projected_Y'] + BASE_MODEL_PREDICTORS))
    # Filter for columns that actually exist in data_for_modeling to avoid errors
    actual_cols_for_final_r = [col for col in cols_for_final_r if col in data_for_modeling.columns]
    final_r_data_df = data_for_modeling[actual_cols_for_final_r].copy()
    final_r_data_df.to_csv(full_data_for_r_path, index=False)

    # Corrected R script command arguments
    r_script_command_full_gam = [
        R_EXECUTABLE, "--vanilla", "-f", PATH_TO_GGPGAMSVC_R_FULL_SCRIPT,
        "--args",
        full_data_for_r_path,               # Arg 1: Input data path (contains TARGET)
        IDENTIFIER_COL,                     # Arg 2: Identifier column name
        OUTPUT_DIR_FULL_MODEL,              # Arg 3: Base output directory for model object and plots subdir
        GGPGAMSVC_FULL_PREDICTIONS_PATH     # Arg 4: Path to save predictions
    ]
    print(f"    Executing R script: {' '.join(r_script_command_full_gam)}")
    try:
        process = subprocess.run(r_script_command_full_gam, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
        print(f"    R script for GGP-GAM-SVC (full data) executed successfully.")
        if process.stdout: print(f"    R STDOUT:\n{process.stdout}")
        if process.stderr: print(f"    R STDERR:\n{process.stderr}") # Print stderr even on success for diagnostics
        
        print(f"    GAM diagnostic plots should be saved in: {GAM_PLOTS_DIR}") # GAM_PLOTS_DIR is created inside OUTPUT_DIR_FULL_MODEL by R script
        print(f"    Trained GAM R model object (.rds) should be saved in: {os.path.join(OUTPUT_DIR_FULL_MODEL, 'gam_svc_final_model.rds')}") # R script saves it here

        if os.path.exists(GGPGAMSVC_FULL_PREDICTIONS_PATH):
            ggpgamsvc_preds_df = pd.read_csv(GGPGAMSVC_FULL_PREDICTIONS_PATH)
            pred_col_r = 'ggpgamsvc_pred' # This should be the column name your R script consistently saves
            if IDENTIFIER_COL in ggpgamsvc_preds_df.columns and pred_col_r in ggpgamsvc_preds_df.columns:
                id_to_pred_map = pd.Series(ggpgamsvc_preds_df[pred_col_r].values, index=ggpgamsvc_preds_df[IDENTIFIER_COL].astype(data_for_modeling[IDENTIFIER_COL].dtype))
                ggpgamsvc_full_predictions = data_for_modeling[IDENTIFIER_COL].map(id_to_pred_map).reindex(data_for_modeling.index).rename('ggpgamsvc_full_pred')
                print(f"    Successfully loaded GGP-GAM-SVC predictions from: {GGPGAMSVC_FULL_PREDICTIONS_PATH}")
                
                if ggpgamsvc_full_predictions.notna().sum() >=2:
                    # Align y_true (data_for_modeling[TARGET]) with valid predictions for metric calculation
                    valid_indices_gam = data_for_modeling[TARGET].index.intersection(ggpgamsvc_full_predictions.dropna().index)
                    if len(valid_indices_gam) >= 2:
                        r2_gam_full = r2_score(data_for_modeling.loc[valid_indices_gam, TARGET], ggpgamsvc_full_predictions.loc[valid_indices_gam])
                        rmse_gam_full = np.sqrt(mean_squared_error(data_for_modeling.loc[valid_indices_gam, TARGET], ggpgamsvc_full_predictions.loc[valid_indices_gam]))
                        print(f"    GGP-GAM-SVC In-Sample Performance: R2 = {r2_gam_full:.4f}, RMSE = {rmse_gam_full:.4f}")
                        pd.DataFrame([{'model':'GGPGAMSVC_FullData', 'R2':r2_gam_full, 'RMSE':rmse_gam_full}]).to_csv(GGPGAMSVC_PERFORMANCE_FULL_PATH, index=False)
                    else:
                        print("    Not enough valid aligned GGP-GAM-SVC predictions for performance metrics.")
            else:
                print(f"    ERROR: Predictions CSV from R ('{GGPGAMSVC_FULL_PREDICTIONS_PATH}') is missing expected columns ('{IDENTIFIER_COL}', '{pred_col_r}'). Found: {list(ggpgamsvc_preds_df.columns)}")
        else:
            print(f"    ERROR: Predictions CSV from R ('{GGPGAMSVC_FULL_PREDICTIONS_PATH}') was not found after R script execution.")
            ggpgamsvc_full_predictions.loc[:] = np.nan # Ensure it's NaN
            
    except subprocess.CalledProcessError as e:
        print(f"    CRITICAL ERROR: R script for GGP-GAM-SVC (full model) failed.")
        print(f"    Return Code: {e.returncode}")
        print(f"    R Stderr:\n{e.stderr}")
        print(f"    R Stdout:\n{e.stdout}")
        ggpgamsvc_full_predictions.loc[:] = np.nan # Ensure it's NaN on error
    finally:
        if os.path.exists(full_data_for_r_path):
            try: os.remove(full_data_for_r_path)
            except Exception: pass

# Ensure the series is filled if loading failed and no overwrite
if ggpgamsvc_full_predictions.isnull().all() and not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(GGPGAMSVC_FULL_PREDICTIONS_PATH):
     print(f"    Attempting to reload GGP-GAM-SVC predictions as it was all NaN after skip logic.")
     ggpgamsvc_preds_df = pd.read_csv(GGPGAMSVC_FULL_PREDICTIONS_PATH)
     pred_col_r_load = 'ggpgamsvc_pred' if 'ggpgamsvc_pred' in ggpgamsvc_preds_df.columns else ('ggpgamsvc_oof_prediction' if 'ggpgamsvc_oof_prediction' in ggpgamsvc_preds_df.columns else None)
     if IDENTIFIER_COL in ggpgamsvc_preds_df.columns and pred_col_r_load:
        id_to_pred_map = pd.Series(ggpgamsvc_preds_df[pred_col_r_load].values, index=ggpgamsvc_preds_df[IDENTIFIER_COL])
        ggpgamsvc_full_predictions = data_for_modeling[IDENTIFIER_COL].map(id_to_pred_map).reindex(data_for_modeling.index).rename('ggpgamsvc_full_pred')


print(f"--- Stage 2 (Base Model Training on Full Data) Complete ---") # This print was inside the GGP-GAM-SVC block, moved out


# --- Stage 3: Train Final LightGBM Meta-Ensemble ---
print(f"\n--- [3] Training Final LightGBM Meta-Ensemble ---")
# Define lgbm_final_predictions_on_full_data before the conditional block
lgbm_final_predictions_on_full_data = pd.Series(np.nan, index=data_for_modeling.index) # Use data_for_modeling index for full alignment
final_lgbm_model_prod = None

stage_3_outputs_exist = (not OVERWRITE_EXISTING_OUTPUTS and
                         os.path.exists(LGBM_META_MODEL_PATH) and
                         os.path.exists(LGBM_FINAL_PREDICTIONS_FULL_DATA_PATH)) # Check for predictions file

if stage_3_outputs_exist:
    print(f"    Skipping LightGBM meta-model training. Loading existing model and predictions.")
    final_lgbm_model_prod = joblib.load(LGBM_META_MODEL_PATH)
    lgbm_preds_df = pd.read_csv(LGBM_FINAL_PREDICTIONS_FULL_DATA_PATH)
    # Assuming saved CSV has IDENTIFIER_COL and the prediction column, map it back
    if IDENTIFIER_COL in lgbm_preds_df.columns and 'lgbm_pred_full' in lgbm_preds_df.columns: # Check for actual column name
        lgbm_final_predictions_on_full_data = data_for_modeling[IDENTIFIER_COL].map(
            pd.Series(lgbm_preds_df['lgbm_pred_full'].values, index=lgbm_preds_df[IDENTIFIER_COL])
        ).reindex(data_for_modeling.index)
    else:
        print(f"    WARNING: Could not load LGBM final predictions from {LGBM_FINAL_PREDICTIONS_FULL_DATA_PATH} due to missing columns.")
    # Construct X_meta_full_df needed for SHAP even if loading
    meta_features_list_for_lgbm_final = [gwr_full_predictions, ggpgamsvc_full_predictions]
    if INCLUDE_GWRF_IN_ENSEMBLE and gwrf_full_predictions is not None and gwrf_full_predictions.notna().any(): meta_features_list_for_lgbm_final.insert(1, gwrf_full_predictions)
    aligned_base_preds_load = [pred.reindex(data_for_modeling.index) for pred in meta_features_list_for_lgbm_final if pred is not None and pred.notna().any()]
    X_meta_full_df = pd.concat([data_for_modeling[['projected_X', 'projected_Y'] + LAPLACIAN_COLS]] + aligned_base_preds_load, axis=1)
    current_cols_load = ['projected_X', 'projected_Y'] + LAPLACIAN_COLS + [p.name for p in aligned_base_preds_load if p is not None]
    X_meta_full_df.columns = current_cols_load
    y_meta_full_series = data_for_modeling[TARGET].copy() # y_meta_full_series defined here
    common_index = X_meta_full_df.index.intersection(y_meta_full_series.index) # Align
    X_meta_full_df = X_meta_full_df.loc[common_index]
    y_meta_full_series = y_meta_full_series.loc[common_index]
    for col in X_meta_full_df.columns: # Impute after alignment
        if X_meta_full_df[col].isnull().any(): X_meta_full_df[col].fillna(X_meta_full_df[col].mean(), inplace=True)


else:
    print("    Proceeding with LightGBM meta-model training...")
    # (Full LightGBM training logic - from your script, including Optuna for final model if that was added)
    # Ensure X_meta_full_df and y_meta_full_series are defined before this
    meta_features_list_for_lgbm_final = [gwr_full_predictions, ggpgamsvc_full_predictions]
    if INCLUDE_GWRF_IN_ENSEMBLE and gwrf_full_predictions is not None and gwrf_full_predictions.notna().any(): meta_features_list_for_lgbm_final.insert(1, gwrf_full_predictions)
    aligned_base_preds = [pred.reindex(data_for_modeling.index) for pred in meta_features_list_for_lgbm_final if pred is not None and pred.notna().any()]
    
    # Ensure LAPLACIAN_COLS are valid columns in data_for_modeling
    valid_laplacian_cols = [col for col in LAPLACIAN_COLS if col in data_for_modeling.columns]

    X_meta_full_df = pd.concat([data_for_modeling[['projected_X', 'projected_Y'] + valid_laplacian_cols]] + aligned_base_preds, axis=1)
    current_cols = ['projected_X', 'projected_Y'] + valid_laplacian_cols + [p.name for p in aligned_base_preds if p is not None]
    X_meta_full_df.columns = current_cols
    y_meta_full_series = data_for_modeling[TARGET].copy()
    
    # Align X and y, then drop NaNs that might come from base predictions
    common_index_meta = X_meta_full_df.index.intersection(y_meta_full_series.index)
    X_meta_full_df = X_meta_full_df.loc[common_index_meta]
    y_meta_full_series = y_meta_full_series.loc[common_index_meta]
    
    # Drop rows where any of the base model predictions (now columns in X_meta_full_df) are NaN
    base_pred_cols_in_X = [p.name for p in aligned_base_preds if p is not None]
    X_meta_full_df.dropna(subset=base_pred_cols_in_X, inplace=True)
    y_meta_full_series = y_meta_full_series.loc[X_meta_full_df.index] # Re-align y after drop

    for col in X_meta_full_df.columns: # Impute any remaining NaNs (e.g. in Laplacians)
        if X_meta_full_df[col].isnull().any(): X_meta_full_df[col].fillna(X_meta_full_df[col].mean(), inplace=True)
    
    if X_meta_full_df.empty or y_meta_full_series.empty: print("ERROR: Meta features empty after cleaning. Cannot train LGBM."); exit()
    meta_feature_order_save_path = os.path.join(OUTPUT_DIR_FULL_MODEL, f'final_lgbm_meta_feature_order{ABLATION_SUFFIX}.json')
try:
    with open(meta_feature_order_save_path, 'w') as f:
        json.dump(list(X_meta_full_df.columns), f) # Save the column order
    print(f"    Saved final LGBM meta-feature order to: {meta_feature_order_save_path}")
except Exception as e_save_json:
    print(f"    WARNING: Could not save LGBM meta-feature order: {e_save_json}")
# --- END OF BLOCK TO ADD ---
    
    print(f"    Loading/Using best LGBM parameters from CV run: {TUNED_LGBM_PARAMS_PATH}")
    try:
        with open(TUNED_LGBM_PARAMS_PATH, 'r') as f: best_lgbm_params = json.load(f)
    except FileNotFoundError:
        print(f"    WARNING: Tuned params file '{TUNED_LGBM_PARAMS_PATH}' not found. Using default LGBM parameters.")
        best_lgbm_params = {'random_state': 42, 'n_jobs': -1, 'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1}
    best_lgbm_params['verbose'] = -1
    
    print("    Training final LGBM meta-model on all data with selected/tuned parameters...")
    final_lgbm_model_prod = lgb.LGBMRegressor(**best_lgbm_params)
    final_lgbm_model_prod.fit(X_meta_full_df, y_meta_full_series)
    joblib.dump(final_lgbm_model_prod, LGBM_META_MODEL_PATH)
    print(f"    SUCCESS: Final LGBM meta-model for production saved to: {LGBM_META_MODEL_PATH}")
    
    lgbm_final_predictions_on_full_data = pd.Series(final_lgbm_model_prod.predict(X_meta_full_df), index=X_meta_full_df.index)
    # Save these predictions with IDENTIFIER_COL for potential loading
    df_to_save_lgbm_preds = pd.DataFrame({
        IDENTIFIER_COL: data_for_modeling.loc[X_meta_full_df.index, IDENTIFIER_COL], # Get ID from original df aligned with X_meta
        'lgbm_pred_full': lgbm_final_predictions_on_full_data
    })
    df_to_save_lgbm_preds.to_csv(LGBM_FINAL_PREDICTIONS_FULL_DATA_PATH, index=False)
    print(f"    LGBM final predictions on full data saved to {LGBM_FINAL_PREDICTIONS_FULL_DATA_PATH}")


# --- SHAP Interpretation for Final Production LGBM model ---
if final_lgbm_model_prod is not None and not X_meta_full_df.empty:
    print(f"\n  --- SHAP Interpretation for Final Production LightGBM Meta-Model ---")
    # ... (Full SHAP logic - from your script)
    try:
        explainer_prod = shap.TreeExplainer(final_lgbm_model_prod)
        shap_values_prod = explainer_prod.shap_values(X_meta_full_df)
        shap.summary_plot(shap_values_prod, X_meta_full_df, show=False, plot_size=(12, max(8, int(X_meta_full_df.shape[1]*0.35)))); plt.savefig(os.path.join(OUTPUT_DIR_FULL_MODEL, f'shap_summary_plot_final_model{ABLATION_SUFFIX}.png'), bbox_inches='tight'); plt.close()
        df_feature_importance_final = pd.DataFrame({'feature': X_meta_full_df.columns, 'importance': final_lgbm_model_prod.feature_importances_}).sort_values(by='importance', ascending=False)
        top_n_features_shap_final = min(5, len(df_feature_importance_final))
        final_shap_dep_plot_dir = os.path.join(OUTPUT_DIR_FULL_MODEL, f'shap_dependence_plots_final_model{ABLATION_SUFFIX}')
        if not os.path.exists(final_shap_dep_plot_dir): os.makedirs(final_shap_dep_plot_dir)
        for i in range(top_n_features_shap_final):
            top_feature = df_feature_importance_final.iloc[i]['feature']
            shap.dependence_plot(top_feature, shap_values_prod, X_meta_full_df, show=False, interaction_index="auto"); dep_plot_path = os.path.join(final_shap_dep_plot_dir, f'shap_dependence_{top_feature}_final.png'); plt.savefig(dep_plot_path, bbox_inches='tight'); plt.close()
    except Exception as e_shap_final: print(f"    Error during SHAP interpretation for final model: {e_shap_final}.")
else: print("   Skipping SHAP for final LGBM model as model or features are not available.")


# --- Stage 4: Train Final Deep Kriging Residual Model ---
print(f"\n--- [4] Training Final Deep Kriging Residual Model ---")
dk_model_final = None; scaler_dk_final = None
deep_kriging_corrections_full_data = pd.Series(0, index=X_meta_full_df.index) # Default, aligned with X_meta_full_df

stage_4_outputs_exist = (not OVERWRITE_EXISTING_OUTPUTS and
                         os.path.exists(DK_MODEL_PATH) and
                         os.path.exists(DK_SCALER_PATH) and
                         os.path.exists(DK_CORRECTIONS_FULL_DATA_PATH))

if stage_4_outputs_exist:
    print(f"    Skipping Deep Kriging training. Loading existing model, scaler, and corrections.")
    dk_model_final = tf.keras.models.load_model(DK_MODEL_PATH)
    scaler_dk_final = joblib.load(DK_SCALER_PATH)
    dk_corr_df = pd.read_csv(DK_CORRECTIONS_FULL_DATA_PATH)
    # This assumes DK_CORRECTIONS_FULL_DATA_PATH was saved with an index that matches X_meta_full_df or with IDENTIFIER_COL
    # For robustness, map it using IDENTIFIER_COL if present, else assume index alignment
    if IDENTIFIER_COL in dk_corr_df.columns and 'dk_correction' in dk_corr_df.columns:
         temp_dk_series = pd.Series(dk_corr_df['dk_correction'].values, index=dk_corr_df[IDENTIFIER_COL])
         # Align to X_meta_full_df's index, which itself is aligned from data_for_modeling
         aligned_ids = data_for_modeling.loc[X_meta_full_df.index, IDENTIFIER_COL]
         deep_kriging_corrections_full_data = aligned_ids.map(temp_dk_series).reindex(X_meta_full_df.index).fillna(0)
    elif 'dk_correction' in dk_corr_df.columns and dk_corr_df.index.equals(X_meta_full_df.index): # If saved with matching index
         deep_kriging_corrections_full_data = dk_corr_df['dk_correction']
    else:
        print(f"    WARNING: Could not properly load DK corrections from {DK_CORRECTIONS_FULL_DATA_PATH}. Using zero corrections.")
else:
    print("    Proceeding with Deep Kriging model training...")
    # (Full Deep Kriging training logic - from your script)
    # It should use `lgbm_final_predictions_on_full_data` and `y_meta_full_series`
    # and save `dk_model_final`, `scaler_dk_final`, and `deep_kriging_corrections_full_data`
    if lgbm_final_predictions_on_full_data is not None and y_meta_full_series is not None and not lgbm_final_predictions_on_full_data.isnull().all():
        final_residuals_for_dk_prod = y_meta_full_series.loc[lgbm_final_predictions_on_full_data.index] - lgbm_final_predictions_on_full_data # Align before subtraction
        dk_input_feature_names = ['projected_X', 'projected_Y'] + LAPLACIAN_COLS
        # Use X_meta_full_df for DK features, aligned with residuals
        dk_input_features_full_df = X_meta_full_df.loc[final_residuals_for_dk_prod.index, dk_input_feature_names].copy()
        
        if not dk_input_features_full_df.empty and not final_residuals_for_dk_prod.empty:
            scaler_dk_final = StandardScaler(); dk_input_features_full_scaled = scaler_dk_final.fit_transform(dk_input_features_full_df)
            joblib.dump(scaler_dk_final, DK_SCALER_PATH); print(f"    Saved final DK scaler: {DK_SCALER_PATH}")
            dk_model_final = Sequential([Input(shape=(dk_input_features_full_scaled.shape[1],)), Dense(128, activation='relu'), Dropout(0.2), Dense(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])
            dk_model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            X_dk_ff_train, X_dk_ff_val, y_dk_ff_train, y_dk_ff_val = train_test_split(dk_input_features_full_scaled, final_residuals_for_dk_prod, test_size=0.1, random_state=42)
            if len(X_dk_ff_train) > 0 and len(X_dk_ff_val) > 0:
                dk_model_final.fit(X_dk_ff_train, y_dk_ff_train, epochs=100, batch_size=256, verbose=0, validation_data=(X_dk_ff_val, y_dk_ff_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
                dk_model_final.save(DK_MODEL_PATH); print(f"    Final DK model saved: {DK_MODEL_PATH}")
                deep_kriging_corrections_full_data = pd.Series(dk_model_final.predict(dk_input_features_full_scaled).flatten(), index=dk_input_features_full_df.index)
                df_dk_corr_to_save = pd.DataFrame({IDENTIFIER_COL: data_for_modeling.loc[dk_input_features_full_df.index, IDENTIFIER_COL], 'dk_correction': deep_kriging_corrections_full_data})
                df_dk_corr_to_save.to_csv(DK_CORRECTIONS_FULL_DATA_PATH, index=False)
                print(f"    DK corrections saved to: {DK_CORRECTIONS_FULL_DATA_PATH}")
            else: print("    WARNING: Not enough data for DK training split.")
        else: print("    WARNING: Empty inputs/residuals for DK training.")
    else: print("    WARNING: LGBM predictions or target for DK residuals missing.")


# --- Stage 5: Final Combined Predictions and Performance on Full Data ---
# (This section is largely the same, ensure it uses the correctly loaded/generated variables)
print(f"\n--- [5] Final Combined Predictions and In-Sample Performance ---")
final_predictions_output_df = pd.DataFrame(index=data_for_modeling.index)
final_predictions_output_df[IDENTIFIER_COL] = data_for_modeling[IDENTIFIER_COL]
final_predictions_output_df[TARGET] = data_for_modeling[TARGET]

# Align lgbm_final_predictions_on_full_data (from X_meta_full_df index) to data_for_modeling index
final_predictions_output_df['lgbm_ensemble_pred_full'] = lgbm_final_predictions_on_full_data.reindex(data_for_modeling.index)
# Align deep_kriging_corrections_full_data (from X_meta_full_df index) to data_for_modeling index
final_predictions_output_df['deep_kriging_correction_full'] = deep_kriging_corrections_full_data.reindex(data_for_modeling.index).fillna(0)

final_predictions_output_df['final_combined_prediction_full'] = final_predictions_output_df['lgbm_ensemble_pred_full'] + final_predictions_output_df['deep_kriging_correction_full']
final_predictions_output_df.to_csv(os.path.join(OUTPUT_DIR_FULL_MODEL, "final_model_predictions_on_training_data.csv"), index=False)
print(f"    Final predictions on full training data saved.")

valid_final_preds_full = final_predictions_output_df.dropna(subset=[TARGET, 'final_combined_prediction_full'])
if not valid_final_preds_full.empty and len(valid_final_preds_full) >=2:
    r2_final_full = r2_score(valid_final_preds_full[TARGET], valid_final_preds_full['final_combined_prediction_full'])
    rmse_final_full = np.sqrt(mean_squared_error(valid_final_preds_full[TARGET], valid_final_preds_full['final_combined_prediction_full']))
    print(f"    Final Combined Model In-Sample Performance: R2 = {r2_final_full:.4f}, RMSE = {rmse_final_full:.4f}")
    pd.DataFrame([{'model':f'FINAL_COMBINED_IN_SAMPLE{ABLATION_SUFFIX}', 'R2':r2_final_full, 'RMSE':rmse_final_full}]).to_csv(FINAL_MODEL_PERFORMANCE_FULL_PATH, index=False)
    
    print("    Calculating Moran's I for final combined model in-sample residuals...")
    residuals_final_insample = valid_final_preds_full[TARGET] - valid_final_preds_full['final_combined_prediction_full']
    gdf_for_moran_full = data_for_modeling.loc[residuals_final_insample.index].copy()
    gdf_for_moran_full['residuals_insample'] = residuals_final_insample
    if not gdf_for_moran_full.empty and 'geometry' in gdf_for_moran_full.columns:
        coords_moran_full = np.array(list(zip(gdf_for_moran_full.geometry.x, gdf_for_moran_full.geometry.y)))
        try:
            w_moran = PysalKNN.from_array(coords_moran_full, k=K_FOR_MORAN_SWM); w_moran.transform = 'R'
            mi = Moran(gdf_for_moran_full['residuals_insample'].values, w_moran, permutations=999)
            print(f"    Moran's I for Final In-Sample Residuals: {mi.I:.4f} (p-value: {mi.p_sim:.4f})")
            pd.DataFrame([{'morans_I': mi.I, 'p_value_sim': mi.p_sim}]).to_csv(FINAL_MODEL_RESIDUAL_MORANS_FULL_PATH, index=False)
        except Exception as e_moran_insample: print(f"    Error calculating Moran's I for final in-sample residuals: {e_moran_insample}")
    else: print("    Skipping final Moran's I (in-sample): No valid data or geometry for SWM.")
else: print("    Final Combined Model In-Sample Performance: Not enough valid predictions for metrics.")

print(f"\n--- Phase 2 (Full Stack Training and Analysis) Complete ---")
print(f"Output directory: {OUTPUT_DIR_FULL_MODEL}")
print("You are now ready for 'Phase 3: Prediction on New Scenarios'.")