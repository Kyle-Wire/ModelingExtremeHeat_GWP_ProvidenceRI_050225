# Final_Predict_New_Scenario.py (Corrected Path)

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import joblib
import subprocess
import json
import tensorflow as tf
from keras.models import load_model # Using tensorflow.keras
from libpysal.weights import KNN as PysalKNN
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from mgwr.gwr import GWR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

print("--- Phase 3: Predicting on New Scenario Data (Corrected Path Logic) ---")

# --- Configuration ---
NEW_SCENARIO_CSV_PATH = '2070Scenario.csv' # YOUR NEW SCENARIO DATA

# --- Determine if GWRF was part of the loaded ensemble model ---
ENSEMBLE_INCLUDED_GWRF = True
# ---
ABLATION_SUFFIX_MODEL_LOAD = "" if ENSEMBLE_INCLUDED_GWRF else "_no_gwrf"

# Paths to the models and objects saved in your FINAL FULL DATA TRAINING run
MODEL_DIR = f"Final_Model_Outputs_Trained_All_Data{ABLATION_SUFFIX_MODEL_LOAD}"
print(f"Loading trained models from directory: {MODEL_DIR}")

# --- THIS IS THE CORRECTED LINE ---
# It now matches your exact directory name: "Prediction_Outputs_2070_Scenario"
PREDICTION_RUN_OUTPUT_DIR = f"Prediction_Outputs_2070_Scenario"

if not os.path.exists(PREDICTION_RUN_OUTPUT_DIR):
    os.makedirs(PREDICTION_RUN_OUTPUT_DIR)
print(f"Prediction outputs will be read from and saved in: {PREDICTION_RUN_OUTPUT_DIR}")

FINAL_PREDICTIONS_GPKG_PATH = os.path.join(PREDICTION_RUN_OUTPUT_DIR, f"predictions_on_{os.path.basename(NEW_SCENARIO_CSV_PATH).replace('.csv', '')}.gpkg")
ORIGINAL_PREPPED_DATA_PATH = os.path.join(MODEL_DIR, "final_data_prepped_with_basis_full_run.gpkg")

BASE_PREDICTORS_SCALER_LOAD_PATH = os.path.join(MODEL_DIR, 'final_base_predictors_scaler.pkl')
GWR_BANDWIDTH_LOAD_PATH = os.path.join(MODEL_DIR, 'gwr_final_bandwidth.json')
LGBM_META_MODEL_LOAD_PATH = os.path.join(MODEL_DIR, f'final_lgbm_meta_model_trained_all_data{ABLATION_SUFFIX_MODEL_LOAD}.pkl')
DK_SCALER_LOAD_PATH = os.path.join(MODEL_DIR, f'final_dk_scaler_trained_all_data{ABLATION_SUFFIX_MODEL_LOAD}.pkl')
DK_MODEL_LOAD_PATH = os.path.join(MODEL_DIR, f'final_deep_kriging_model_trained_all_data{ABLATION_SUFFIX_MODEL_LOAD}.keras')
R_MODEL_OBJECT_LOAD_PATH = os.path.join(MODEL_DIR, "gam_svc_final_model.rds")
META_FEATURE_ORDER_LOAD_PATH = os.path.join(MODEL_DIR, f'final_lgbm_meta_feature_order{ABLATION_SUFFIX_MODEL_LOAD}.json')

# Intermediate prediction files for "save as you go" for the NEW SCENARIO DATA
NEW_DATA_GWR_PREDS_CSV = os.path.join(PREDICTION_RUN_OUTPUT_DIR, "scenario_gwr_preds.csv")
NEW_DATA_GWRF_PREDS_CSV = os.path.join(PREDICTION_RUN_OUTPUT_DIR, "scenario_gwrf_preds.csv")
NEW_DATA_GGPGAMSVC_PREDS_CSV = os.path.join(PREDICTION_RUN_OUTPUT_DIR, "scenario_ggpgamsvc_preds.csv")

OVERWRITE_SCENARIO_BASE_PREDS = False

# File existence checks
required_files_to_load = [
    BASE_PREDICTORS_SCALER_LOAD_PATH, GWR_BANDWIDTH_LOAD_PATH, LGBM_META_MODEL_LOAD_PATH,
    DK_SCALER_LOAD_PATH, DK_MODEL_LOAD_PATH, R_MODEL_OBJECT_LOAD_PATH,
    ORIGINAL_PREPPED_DATA_PATH
]
for f_path in required_files_to_load:
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Required file for loading not found: {f_path}")
if not os.path.exists(META_FEATURE_ORDER_LOAD_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: Meta feature order file not found: {META_FEATURE_ORDER_LOAD_PATH}")
else:
    with open(META_FEATURE_ORDER_LOAD_PATH, 'r') as f:
        expected_meta_cols_ordered = json.load(f)
    print("    Successfully loaded meta-feature order for LightGBM.")

if not os.path.exists(NEW_SCENARIO_CSV_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: New scenario data file not found: {NEW_SCENARIO_CSV_PATH}")

# Fixed Parameters
TARGET = 'AAT_z'
COORDS_COLS = ['POINT_X', 'POINT_Y']
IDENTIFIER_COL = 'OBJECTID'
BASE_MODEL_PREDICTORS = ['Distance_from_water_m', 'Pct_Impervious', 'Pct_Canopy', 'Pct_GreenSpace', 'Elevation_m']
INITIAL_CRS = "EPSG:3438"
TARGET_PROJECTED_CRS = "EPSG:26919"
N_LAPLACIAN_EIGENMAPS = 150
K_FOR_LAPLACIAN_SWM = 10

# GWRF Hyperparameters
GWRF_ADAPTIVE_K_PRED = 50
GWRF_N_ESTIMATORS_PRED = 100
GWRF_MIN_LEAF_PRED = 10
GWRF_MAX_FEATURES_PRED = 'sqrt'
GWRF_KERNEL_TYPE = 'gaussian'

# R Script for GGP-GAM-SVC Prediction
PATH_TO_GGPGAMSVC_PREDICT_R_SCRIPT = "Final_GGPGAM_Predict.r"
R_EXECUTABLE = r"C:\Users\KWire\AppData\Local\Programs\R\R-4.5.0\bin\x64\R.exe"

# --- Stage 1: Load All Pre-Trained Models and Scalers ---
print(f"\n--- [1] Loading Pre-Trained Models ---")
scaler_base = joblib.load(BASE_PREDICTORS_SCALER_LOAD_PATH)
with open(GWR_BANDWIDTH_LOAD_PATH, 'r') as f: gwr_bandwidth_k = json.load(f)['adaptive_k']
model_lgbm = joblib.load(LGBM_META_MODEL_LOAD_PATH)
scaler_dk = joblib.load(DK_SCALER_LOAD_PATH)
model_dk = load_model(DK_MODEL_LOAD_PATH)
original_training_data_gdf = gpd.read_file(ORIGINAL_PREPPED_DATA_PATH)
print("    All models, scalers, and configurations loaded successfully.")

# --- Stage 2: Load and Preprocess New Scenario Data ---
print(f"\n--- [2] Loading and Preprocessing New Data from '{NEW_SCENARIO_CSV_PATH}' ---")
new_data = pd.read_csv(NEW_SCENARIO_CSV_PATH, low_memory=False)
print(f"    Loaded new scenario data. Initial shape: {new_data.shape}")
essential_cols_new_data = COORDS_COLS + BASE_MODEL_PREDICTORS + [IDENTIFIER_COL]
for col in essential_cols_new_data:
    if col not in new_data.columns: raise ValueError(f"Essential column '{col}' not found in new data.")
    if col != IDENTIFIER_COL: new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
new_data.dropna(subset=essential_cols_new_data, inplace=True)
if new_data.empty: raise ValueError("New scenario data is empty after NA drop on essential columns.")

new_data_gdf = gpd.GeoDataFrame(new_data, geometry=gpd.points_from_xy(new_data[COORDS_COLS[0]], new_data[COORDS_COLS[1]]), crs=INITIAL_CRS).to_crs(TARGET_PROJECTED_CRS)
new_data_gdf['projected_X'] = new_data_gdf.geometry.x; new_data_gdf['projected_Y'] = new_data_gdf.geometry.y

print("    Generating Laplacian Eigenmaps for new data points...")
coords_new = np.array(list(zip(new_data_gdf.geometry.x, new_data_gdf.geometry.y)))
LAPLACIAN_COLS = []
if len(coords_new) >= K_FOR_LAPLACIAN_SWM + 1 :
    try:
        w_laplacian_new = PysalKNN.from_array(coords_new, k=K_FOR_LAPLACIAN_SWM)
        L_new = laplacian(w_laplacian_new.sparse, normed=True)
        k_eigsh_new = min(N_LAPLACIAN_EIGENMAPS + 1, L_new.shape[0] - 2)
        if k_eigsh_new >= 2:
            vals, vecs = eigsh(L_new, k=k_eigsh_new, which='SM')
            eigenmaps_to_add = vecs[:, 1:min(vecs.shape[1], N_LAPLACIAN_EIGENMAPS + 1)]
            temp_lap_cols = [f'LAPEIG_{i+1}' for i in range(eigenmaps_to_add.shape[1])]
            eigenmap_df_new = pd.DataFrame(eigenmaps_to_add, columns=temp_lap_cols, index=new_data_gdf.index)
            new_data_gdf = new_data_gdf.join(eigenmap_df_new)
            LAPLACIAN_COLS = temp_lap_cols
            print(f"      Generated and added {len(LAPLACIAN_COLS)} Laplacian Eigenmaps for new data.")
        else: print(f"    Warning: Too few points for robust Laplacian Eigenmap generation (k_eigsh={k_eigsh_new}).")
    except Exception as e_lap_new: print(f"    Error generating Laplacian Eigenmaps for new data: {e_lap_new}.")
else: print(f"    Warning: Not enough data points ({len(coords_new)}) to generate Laplacian Eigenmaps with k={K_FOR_LAPLACIAN_SWM}.")

expected_lap_cols_names = [f'LAPEIG_{i+1}' for i in range(N_LAPLACIAN_EIGENMAPS)]
for col_name in expected_lap_cols_names:
    if col_name not in new_data_gdf.columns: new_data_gdf[col_name] = 0.0
LAPLACIAN_COLS = expected_lap_cols_names
print(f"    New data preprocessed. Shape: {new_data_gdf.shape}")

# --- Stage 3: Generate Base Model Predictions (with save-as-you-go) ---
print("\n--- [3] Generating Base Model Predictions ---")
X_new_scaled_df = pd.DataFrame(scaler_base.transform(new_data_gdf[BASE_MODEL_PREDICTORS]), columns=BASE_MODEL_PREDICTORS, index=new_data_gdf.index)

# --- [3.1] GWR Predictions ---
print("  --- [3.1] GWR Predictions ---")
gwr_pred_col_name = 'gwr_full_pred'
if not OVERWRITE_SCENARIO_BASE_PREDS and os.path.exists(NEW_DATA_GWR_PREDS_CSV):
    print(f"    Loading existing GWR predictions for new scenario from {NEW_DATA_GWR_PREDS_CSV}")
    gwr_preds_loaded_df = pd.read_csv(NEW_DATA_GWR_PREDS_CSV)
    if IDENTIFIER_COL in gwr_preds_loaded_df.columns and gwr_pred_col_name in gwr_preds_loaded_df.columns:
        try:
            gwr_preds_loaded_df[IDENTIFIER_COL] = gwr_preds_loaded_df[IDENTIFIER_COL].astype(new_data_gdf[IDENTIFIER_COL].dtype)
            new_data_gdf = new_data_gdf.merge(gwr_preds_loaded_df[[IDENTIFIER_COL, gwr_pred_col_name]], on=IDENTIFIER_COL, how='left')
            print(f"    Successfully merged {new_data_gdf[gwr_pred_col_name].notna().sum()} pre-existing GWR predictions.")
        except Exception as e:
            print(f"    WARNING: Failed to merge GWR predictions due to an error: {e}. Will regenerate.")
            if gwr_pred_col_name in new_data_gdf.columns: new_data_gdf.drop(columns=[gwr_pred_col_name], inplace=True)
    else:
        print(f"    WARNING: GWR predictions CSV {NEW_DATA_GWR_PREDS_CSV} is missing required columns. Will regenerate.")

if OVERWRITE_SCENARIO_BASE_PREDS or gwr_pred_col_name not in new_data_gdf.columns or new_data_gdf[gwr_pred_col_name].isnull().any():
    print("    Generating GWR predictions for new scenario...")
    # (GWR prediction logic is unchanged)
    X_train_gwr_original_scaled = scaler_base.transform(original_training_data_gdf[BASE_MODEL_PREDICTORS])
    coords_train_original = np.array(list(zip(original_training_data_gdf.geometry.x, original_training_data_gdf.geometry.y)))
    y_train_gwr_original = original_training_data_gdf[TARGET].values.reshape(-1,1)
    gwr_pred_model = GWR(coords_train_original, y_train_gwr_original, X_train_gwr_original_scaled, bw=gwr_bandwidth_k, kernel='gaussian', fixed=False, spherical=False)
    try:
        gwr_pred_results = gwr_pred_model.predict(coords_new, X_new_scaled_df.values)
        new_data_gdf[gwr_pred_col_name] = gwr_pred_results.predictions.flatten()
        pd.DataFrame({IDENTIFIER_COL: new_data_gdf[IDENTIFIER_COL], gwr_pred_col_name: new_data_gdf[gwr_pred_col_name]}).to_csv(NEW_DATA_GWR_PREDS_CSV, index=False)
        print(f"    GWR predictions for new scenario generated and saved to {NEW_DATA_GWR_PREDS_CSV}.")
    except Exception as e_gwr_pred: print(f"    ERROR during GWR prediction: {e_gwr_pred}"); new_data_gdf[gwr_pred_col_name] = np.nan

# --- [3.2] GWRF Predictions ---
gwrf_pred_col_name = 'gwrf_full_pred'
def predict_gwrf_new_scenario(train_data_for_gwrf, points_to_predict_gdf_current, predictors, target_col_name_in_train, scaler_obj_fitted, bandwidth_k_val, adaptive_val, n_estimators_val, min_samples_leaf_val, max_features_val, n_jobs_val, kernel_type_val='gaussian'):
    # (GWRF function definition is unchanged)
    train_coords = train_data_for_gwrf[['projected_X', 'projected_Y']].values
    predict_coords = points_to_predict_gdf_current[['projected_X', 'projected_Y']].values
    X_train_scaled_df = pd.DataFrame(scaler_obj_fitted.transform(train_data_for_gwrf[predictors]), columns=predictors, index=train_data_for_gwrf.index)
    y_train = train_data_for_gwrf[target_col_name_in_train]
    X_predict_scaled_df = pd.DataFrame(scaler_obj_fitted.transform(points_to_predict_gdf_current[predictors]), columns=predictors, index=points_to_predict_gdf_current.index)
    ball_tree = BallTree(train_coords, metric='euclidean')
    final_preds = pd.Series(np.nan, index=points_to_predict_gdf_current.index)
    print(f"    GWRF (Predict): for {len(points_to_predict_gdf_current)} points (k={bandwidth_k_val}, kernel={kernel_type_val})...")
    for i in range(len(points_to_predict_gdf_current)):
        current_predict_index = points_to_predict_gdf_current.index[i]; point_coords = predict_coords[i].reshape(1, -1)
        distances, indices_from_query = ball_tree.query(point_coords, k=len(train_coords)); distances = distances[0]; indices_from_query = indices_from_query[0]
        local_distances = distances
        local_bw_dist = 0.0
        if adaptive_val:
            k_for_bw = int(min(bandwidth_k_val, len(local_distances) -1)); k_for_bw = max(0, k_for_bw)
            sorted_local_distances = np.sort(local_distances)
            if k_for_bw < len(sorted_local_distances): local_bw_dist = sorted_local_distances[k_for_bw]
            elif sorted_local_distances.size > 0: local_bw_dist = sorted_local_distances[-1]
        else: local_bw_dist = bandwidth_k_val
        if local_bw_dist == 0:
            weights = np.zeros_like(local_distances)
            if local_distances.size > 0:
                zero_dist_mask_local = (local_distances == 0)
                if np.any(zero_dist_mask_local): weights[zero_dist_mask_local] = 1.0 / np.sum(zero_dist_mask_local)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                if kernel_type_val == 'gaussian': weights = np.exp(-(local_distances**2) / (2 * (local_bw_dist**2))); weights[local_distances >= local_bw_dist * 3] = 0.0
                elif kernel_type_val == 'bi-square': weights = (1 - (local_distances / local_bw_dist)**2)**2; weights[local_distances >= local_bw_dist] = 0.0
                else: weights = (1 - (local_distances / local_bw_dist)**2)**2; weights[local_distances >= local_bw_dist] = 0.0
        weights[np.isnan(weights)] = 0.0
        sum_weights = np.sum(weights); nz_weights = np.sum(weights > 1e-9)
        if i % 1000 == 0 and i > 0: print(f"      GWRF processed point {i}/{len(points_to_predict_gdf_current)}")
        if sum_weights < 1e-6 or nz_weights < min_samples_leaf_val: final_preds.loc[current_predict_index] = np.nan; continue
        try:
            rf = RandomForestRegressor(n_estimators=n_estimators_val, min_samples_leaf=min_samples_leaf_val, max_features=max_features_val, random_state=42, n_jobs=n_jobs_val)
            rf.fit(X_train_scaled_df, y_train, sample_weight=weights)
            pred = rf.predict(X_predict_scaled_df.loc[[current_predict_index]])[0]
            final_preds.loc[current_predict_index] = pred
        except Exception as e_rf: print(f"        Error GWRF predict point {current_predict_index}: {e_rf}"); final_preds.loc[current_predict_index] = np.nan
    return final_preds

if ENSEMBLE_INCLUDED_GWRF:
    print("  --- [3.2] GWRF Predictions ---")
    if not OVERWRITE_SCENARIO_BASE_PREDS and os.path.exists(NEW_DATA_GWRF_PREDS_CSV):
        print(f"    Loading existing GWRF predictions for new scenario from {NEW_DATA_GWRF_PREDS_CSV}")
        gwrf_preds_loaded_df = pd.read_csv(NEW_DATA_GWRF_PREDS_CSV)
        if IDENTIFIER_COL in gwrf_preds_loaded_df.columns and gwrf_pred_col_name in gwrf_preds_loaded_df.columns:
            try:
                gwrf_preds_loaded_df[IDENTIFIER_COL] = gwrf_preds_loaded_df[IDENTIFIER_COL].astype(new_data_gdf[IDENTIFIER_COL].dtype)
                new_data_gdf = new_data_gdf.merge(gwrf_preds_loaded_df[[IDENTIFIER_COL, gwrf_pred_col_name]], on=IDENTIFIER_COL, how='left')
                print(f"    Successfully merged {new_data_gdf[gwrf_pred_col_name].notna().sum()} pre-existing GWRF predictions.")
            except Exception as e:
                print(f"    WARNING: Failed to merge GWRF predictions due to an error: {e}. Will regenerate.")
                if gwrf_pred_col_name in new_data_gdf.columns: new_data_gdf.drop(columns=[gwrf_pred_col_name], inplace=True)
        else:
            print(f"    WARNING: GWRF predictions CSV {NEW_DATA_GWRF_PREDS_CSV} is missing required columns. Will regenerate.")
    
    if OVERWRITE_SCENARIO_BASE_PREDS or gwrf_pred_col_name not in new_data_gdf.columns or new_data_gdf[gwrf_pred_col_name].isnull().any():
        print(f"    Generating GWRF predictions for new scenario...")
        new_data_gdf[gwrf_pred_col_name] = predict_gwrf_new_scenario(
            original_training_data_gdf, new_data_gdf, BASE_MODEL_PREDICTORS, TARGET, scaler_base,
            bandwidth_k_val=GWRF_ADAPTIVE_K_PRED, adaptive_val=True,
            n_estimators_val=GWRF_N_ESTIMATORS_PRED, min_samples_leaf_val=GWRF_MIN_LEAF_PRED,
            max_features_val=GWRF_MAX_FEATURES_PRED, n_jobs_val=-1, kernel_type_val=GWRF_KERNEL_TYPE
        )
        pd.DataFrame({IDENTIFIER_COL: new_data_gdf[IDENTIFIER_COL], gwrf_pred_col_name: new_data_gdf[gwrf_pred_col_name]}).to_csv(NEW_DATA_GWRF_PREDS_CSV, index=False)
        print(f"    GWRF predictions for new scenario generated and saved to {NEW_DATA_GWRF_PREDS_CSV}.")
else:
    print("    Skipping GWRF predictions as ENSEMBLE_INCLUDED_GWRF is False.")
    if gwrf_pred_col_name not in new_data_gdf.columns : new_data_gdf[gwrf_pred_col_name] = 0.0

# --- [3.3] GGP-GAM-SVC Predictions (via R) ---
print("  --- [3.3] GGP-GAM-SVC Predictions ---")
ggpgamsvc_pred_col_name = 'ggpgamsvc_full_pred'
if not OVERWRITE_SCENARIO_BASE_PREDS and os.path.exists(NEW_DATA_GGPGAMSVC_PREDS_CSV):
    print(f"    Loading existing GGP-GAM-SVC predictions for new scenario from {NEW_DATA_GGPGAMSVC_PREDS_CSV}")
    ggpgamsvc_preds_loaded_df = pd.read_csv(NEW_DATA_GGPGAMSVC_PREDS_CSV)
    if IDENTIFIER_COL in ggpgamsvc_preds_loaded_df.columns and ggpgamsvc_pred_col_name in ggpgamsvc_preds_loaded_df.columns:
        try:
            ggpgamsvc_preds_loaded_df[IDENTIFIER_COL] = ggpgamsvc_preds_loaded_df[IDENTIFIER_COL].astype(new_data_gdf[IDENTIFIER_COL].dtype)
            new_data_gdf = new_data_gdf.merge(ggpgamsvc_preds_loaded_df[[IDENTIFIER_COL, ggpgamsvc_pred_col_name]], on=IDENTIFIER_COL, how='left')
            print(f"    Successfully merged {new_data_gdf[ggpgamsvc_pred_col_name].notna().sum()} pre-existing GGP-GAM-SVC predictions.")
        except Exception as e:
            print(f"    WARNING: Failed to merge GGP-GAM-SVC predictions due to an error: {e}. Will regenerate.")
            if ggpgamsvc_pred_col_name in new_data_gdf.columns: new_data_gdf.drop(columns=[ggpgamsvc_pred_col_name], inplace=True)
    else:
        print(f"    WARNING: GGP-GAM-SVC predictions CSV {NEW_DATA_GGPGAMSVC_PREDS_CSV} is missing required columns. Will regenerate.")

if OVERWRITE_SCENARIO_BASE_PREDS or ggpgamsvc_pred_col_name not in new_data_gdf.columns or new_data_gdf[ggpgamsvc_pred_col_name].isnull().any():
    print(f"    Generating GGP-GAM-SVC predictions for new scenario via R...")
    temp_new_data_r_path = os.path.join(PREDICTION_RUN_OUTPUT_DIR, "temp_new_data_for_r_predict.csv")
    temp_gam_preds_r_path = os.path.join(PREDICTION_RUN_OUTPUT_DIR, "temp_gam_preds_from_r_predict.csv")
    cols_for_r_pred = list(set([IDENTIFIER_COL, 'projected_X', 'projected_Y'] + BASE_MODEL_PREDICTORS))
    new_data_for_r = new_data_gdf[[col for col in cols_for_r_pred if col in new_data_gdf.columns]].copy()
    new_data_for_r.to_csv(temp_new_data_r_path, index=False)
    r_script_command_predict = [ R_EXECUTABLE, "--vanilla", "-f", PATH_TO_GGPGAMSVC_PREDICT_R_SCRIPT, "--args", R_MODEL_OBJECT_LOAD_PATH, temp_new_data_r_path, temp_gam_preds_r_path, IDENTIFIER_COL]
    try:
        process = subprocess.run(r_script_command_predict, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if os.path.exists(temp_gam_preds_r_path):
            gam_preds_df = pd.read_csv(temp_gam_preds_r_path)
            pred_col_r_output = 'ggpgamsvc_pred'
            if IDENTIFIER_COL in gam_preds_df.columns and pred_col_r_output in gam_preds_df.columns:
                gam_preds_df.rename(columns={pred_col_r_output: ggpgamsvc_pred_col_name}, inplace=True)
                gam_preds_df[IDENTIFIER_COL] = gam_preds_df[IDENTIFIER_COL].astype(new_data_gdf[IDENTIFIER_COL].dtype)
                if ggpgamsvc_pred_col_name in new_data_gdf.columns:
                    new_data_gdf.drop(columns=[ggpgamsvc_pred_col_name], inplace=True)
                new_data_gdf = new_data_gdf.merge(gam_preds_df[[IDENTIFIER_COL, ggpgamsvc_pred_col_name]], on=IDENTIFIER_COL, how='left')
                new_data_gdf[[IDENTIFIER_COL, ggpgamsvc_pred_col_name]].to_csv(NEW_DATA_GGPGAMSVC_PREDS_CSV, index=False)
                print(f"    GGP-GAM-SVC predictions for new scenario generated and saved to {NEW_DATA_GGPGAMSVC_PREDS_CSV}.")
            else:
                print(f"    ERROR: GAM Preds CSV from R missing columns. Found: {gam_preds_df.columns}"); new_data_gdf[ggpgamsvc_pred_col_name] = np.nan
        else:
            print(f"    ERROR: GAM Preds CSV from R not found."); new_data_gdf[ggpgamsvc_pred_col_name] = np.nan
    except Exception as e_r_predict:
        print(f"    ERROR: R prediction script for GGP-GAM-SVC failed: {e_r_predict}"); new_data_gdf[ggpgamsvc_pred_col_name] = np.nan
    finally:
        if os.path.exists(temp_new_data_r_path): os.remove(temp_new_data_r_path)
        if os.path.exists(temp_gam_preds_r_path): os.remove(temp_gam_preds_r_path)

# --- Stage 4: Generate Final Ensemble Prediction ---
print("\n--- [4] Generating Final Meta-Learner (LGBM) Prediction ---")
available_cols_for_meta = [col for col in expected_meta_cols_ordered if col in new_data_gdf.columns]
meta_features_new_df = new_data_gdf[available_cols_for_meta].copy()

for col in meta_features_new_df.columns:
    if meta_features_new_df[col].isnull().any():
        mean_val = meta_features_new_df[col].mean()
        meta_features_new_df[col].fillna(mean_val, inplace=True)
        print(f"    Imputed NaNs in new scenario meta-feature '{col}' with mean value {mean_val:.4f}")

if expected_meta_cols_ordered:
    for col in expected_meta_cols_ordered:
        if col not in meta_features_new_df.columns:
            print(f"    Warning: Expected meta-feature '{col}' not found. Filling with 0.")
            meta_features_new_df[col] = 0.0
    meta_features_new_df = meta_features_new_df[expected_meta_cols_ordered]
else:
    raise ValueError("CRITICAL: `expected_meta_cols_ordered` was not loaded.")

if len(meta_features_new_df.columns) != len(expected_meta_cols_ordered):
    raise ValueError(f"CRITICAL: Meta feature mismatch for LGBM. Model expects {len(expected_meta_cols_ordered)}, data has {len(meta_features_new_df.columns)}")

new_data_gdf['lgbm_ensemble_pred'] = model_lgbm.predict(meta_features_new_df)
print("    Ensemble predictions generated.")

# --- Stage 5: Generate Final Residual Correction ---
print("\n--- [5] Generating Final Residual (Deep Kriging) Correction ---")
dk_input_feature_names_pred = ['projected_X', 'projected_Y'] + LAPLACIAN_COLS
dk_inputs_new_df = new_data_gdf[dk_input_feature_names_pred].copy()
for col in dk_input_feature_names_pred:
    if dk_inputs_new_df[col].isnull().any(): dk_inputs_new_df[col].fillna(dk_inputs_new_df[col].mean(), inplace=True)

dk_inputs_new_scaled = scaler_dk.transform(dk_inputs_new_df)
new_data_gdf['deep_kriging_correction'] = model_dk.predict(dk_inputs_new_scaled).flatten()
print("    Residual corrections generated.")

# --- Stage 6: Combine and Save Final Output ---
print("\n--- [6] Combining Predictions and Saving Final Output ---")
new_data_gdf['final_combined_prediction'] = new_data_gdf['lgbm_ensemble_pred'] + new_data_gdf['deep_kriging_correction']

output_cols_final = [IDENTIFIER_COL, 'geometry', 'final_combined_prediction',
                     'lgbm_ensemble_pred', 'deep_kriging_correction',
                     'gwr_full_pred', 'ggpgamsvc_full_pred']

if ENSEMBLE_INCLUDED_GWRF and 'gwrf_full_pred' in new_data_gdf.columns:
    output_cols_final.append('gwrf_full_pred')

final_output_gdf_cols = [col for col in output_cols_final if col in new_data_gdf.columns]
final_output_gdf = new_data_gdf[final_output_gdf_cols].copy()

final_output_gdf.to_file(FINAL_PREDICTIONS_GPKG_PATH, driver='GPKG', layer=f"predictions_{os.path.basename(NEW_SCENARIO_CSV_PATH).split('.')[0]}")

print("\n--- Phase 3 Prediction Script Complete ---")
print(f"SUCCESS: Final predictions for the new scenario have been saved to:")
print(FINAL_PREDICTIONS_GPKG_PATH)