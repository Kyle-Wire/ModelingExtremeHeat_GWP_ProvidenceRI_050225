# Comprehensive_Spatial_ML_Workflow_Definitive_v3.py

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import joblib  # For saving scalers and models
import tempfile
import subprocess
import time
import matplotlib.pyplot as plt  # For saving SHAP plot
import shutil

# Preprocessing & Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from libpysal.weights import KNN as PysalKNN
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

# Base Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import statsmodels.api as sm

# Meta-Learner & Interpretation
from sklearn.model_selection import GroupKFold, train_test_split # For Optuna
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import shap # For SHAP interpretation
import optuna # For hyperparameter tuning

# Deep Kriging (Neural Network)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping

# Moran's I for residuals
from esda.moran import Moran


# --- Configuration ---
CSV_PATH = 'with_two_form_indices.csv' # Make sure this path is correct
TARGET = 'AAT_z'
COORDS_COLS = ['POINT_X', 'POINT_Y']
IDENTIFIER_COL = 'OBJECTID'
INITIAL_CRS = "EPSG:3438"
TARGET_PROJECTED_CRS = "EPSG:26919"

BASE_MODEL_PREDICTORS = ['Distance_from_water_m', 'Pct_Impervious', 'Pct_Canopy', 'Pct_GreenSpace', 'Elevation_m']
print(f"Using predefined base model predictors: {BASE_MODEL_PREDICTORS}")

# --- CONFIGURATION FOR THIS RUN ---
INCLUDE_GWRF_IN_ENSEMBLE = True # Set to False to exclude GWRF
ABLATION_SUFFIX = "" if INCLUDE_GWRF_IN_ENSEMBLE else "_no_gwrf"

OUTPUT_DIR = f"Comprehensive_Spatial_ML_Workflow_Final"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"Output files will be saved in: {os.path.join(os.getcwd(), OUTPUT_DIR)}")

N_SPATIAL_FOLDS = 5
OVERWRITE_EXISTING_OUTPUTS = False # Set to True to force re-running base models

N_LAPLACIAN_EIGENMAPS = 150
K_FOR_LAPLACIAN_SWM = 10
PREPPED_DATA_WITH_BASIS_PATH = os.path.join(OUTPUT_DIR, "data_prepped_with_laplacian_basis.gpkg")

PATH_TO_GGPGAMSVC_R_SCRIPT = "Final_GGPGAM_Fold.r"
# Attempt to find the R executable automatically for cross-platform support
R_EXECUTABLE = shutil.which("R") or r"C:\Users\KWire\AppData\Local\Programs\R\R-4.5.0\bin\x64\R.exe"
KEEP_TEMP_PREDS_CSV_FOR_FIRST_R_FOLD = False

gwr_oof_file = os.path.join(OUTPUT_DIR, "gwr_oof_predictions.csv")
gwr_performance_file = os.path.join(OUTPUT_DIR, "gwr_oof_performance.csv")
gwrf_oof_file = os.path.join(OUTPUT_DIR, f"gwrf_oof_predictions{ABLATION_SUFFIX}.csv")
gwrf_performance_file = os.path.join(OUTPUT_DIR, f"gwrf_oof_performance{ABLATION_SUFFIX}.csv")
ggpgamsvc_oof_file = os.path.join(OUTPUT_DIR, "ggpgamsvc_oof_predictions.csv")
ggpgamsvc_performance_file = os.path.join(OUTPUT_DIR, "ggpgamsvc_oof_performance.csv")

lgbm_ensemble_oof_output_path = os.path.join(OUTPUT_DIR, f"lgbm_ensemble_oof_predictions{ABLATION_SUFFIX}.csv")
lgbm_ensemble_performance_path = os.path.join(OUTPUT_DIR, f"lgbm_ensemble_performance{ABLATION_SUFFIX}.csv")
final_lgbm_model_path = os.path.join(OUTPUT_DIR, f'final_lgbm_meta_model{ABLATION_SUFFIX}.pkl')
tuned_lgbm_params_path = os.path.join(OUTPUT_DIR, f'tuned_lgbm_params{ABLATION_SUFFIX}.json')
shap_summary_plot_path = os.path.join(OUTPUT_DIR, f'shap_summary_plot{ABLATION_SUFFIX}.png')
shap_dependence_plot_dir = os.path.join(OUTPUT_DIR, f'shap_dependence_plots{ABLATION_SUFFIX}')
if not os.path.exists(shap_dependence_plot_dir): os.makedirs(shap_dependence_plot_dir)

dk_scaler_path = os.path.join(OUTPUT_DIR, f'dk_scaler{ABLATION_SUFFIX}.pkl')
deep_kriging_model_path = os.path.join(OUTPUT_DIR, f'deep_kriging_residual_model{ABLATION_SUFFIX}.keras')
final_prediction_output_path = os.path.join(OUTPUT_DIR, f"final_combined_predictions{ABLATION_SUFFIX}.csv")
final_performance_path = os.path.join(OUTPUT_DIR, f"final_combined_performance{ABLATION_SUFFIX}.csv")
final_residual_morans_path = os.path.join(OUTPUT_DIR, f"final_combined_residual_morans_i{ABLATION_SUFFIX}.csv")
K_FOR_MORAN_SWM = 100

N_OPTUNA_TRIALS_LGBM = 25

print(f"--- Comprehensive Spatial ML Workflow Script (Definitive v3) ---")

# --- Stage 1: Data Preprocessing and Laplacian Eigenmaps ---
print(f"\n--- Stage 1: Data Preprocessing and Laplacian Eigenmaps ---")
data_for_modeling = None
LAPLACIAN_COLS = []

print(f"\n  --- [1.1] Data Preprocessing ---")
if not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(PREPPED_DATA_WITH_BASIS_PATH):
    print(f"    Loading existing preprocessed data with basis from {PREPPED_DATA_WITH_BASIS_PATH}")
    data_for_modeling = gpd.read_file(PREPPED_DATA_WITH_BASIS_PATH)
    LAPLACIAN_COLS = [col for col in data_for_modeling.columns if col.startswith('LAPEIG_')]
    print(f"      Loaded data shape: {data_for_modeling.shape}")
    print(f"      Found {len(LAPLACIAN_COLS)} Laplacian Eigenmap columns.")
else:
    try:
        data = pd.read_csv(CSV_PATH, low_memory=False)
        print(f"    Successfully loaded data from {CSV_PATH}. Shape: {data.shape}")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {CSV_PATH}."); exit()

    if IDENTIFIER_COL not in data.columns:
        print(f"ERROR: IDENTIFIER_COL '{IDENTIFIER_COL}' not found."); exit()

    for p_col in BASE_MODEL_PREDICTORS + [TARGET] + COORDS_COLS:
        if p_col in data.columns: data[p_col] = pd.to_numeric(data[p_col], errors='coerce')
        else: print(f"ERROR: Expected column '{p_col}' not found in CSV."); exit()

    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                temp_converted_col = pd.to_numeric(data[col], errors='coerce')
                if temp_converted_col.notna().sum() > 0.5 * len(temp_converted_col): data[col] = temp_converted_col
            except ValueError: pass
        if pd.api.types.is_numeric_dtype(data[col].dtype):
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)

    essential_cols_for_dropna = COORDS_COLS + BASE_MODEL_PREDICTORS + [TARGET]
    if IDENTIFIER_COL not in essential_cols_for_dropna: essential_cols_for_dropna.append(IDENTIFIER_COL)
    original_rows = len(data); data.dropna(subset=essential_cols_for_dropna, inplace=True)
    print(f"      Shape after dropping NaNs based on essential columns: {data.shape} (dropped {original_rows - len(data)} rows)")
    if data.empty: print("ERROR: DataFrame empty after NaNs drop."); exit()

    cols_to_keep_in_gdf = list(set([IDENTIFIER_COL] + COORDS_COLS + BASE_MODEL_PREDICTORS + [TARGET]))
    cols_to_keep_in_gdf = [col for col in cols_to_keep_in_gdf if col in data.columns]
    gdf = gpd.GeoDataFrame(data[cols_to_keep_in_gdf].copy(), geometry=gpd.points_from_xy(data[COORDS_COLS[0]], data[COORDS_COLS[1]]), crs=INITIAL_CRS)

    if str(gdf.crs).upper() != str(TARGET_PROJECTED_CRS).upper():
        print(f"      Projecting GeoDataFrame from {gdf.crs} to {TARGET_PROJECTED_CRS}...")
        gdf = gdf.to_crs(TARGET_PROJECTED_CRS)
    gdf['projected_X'] = gdf.geometry.x; gdf['projected_Y'] = gdf.geometry.y
    data_for_modeling = gdf.copy()
    print(f"      Data preparation complete. Shape of data_for_modeling: {data_for_modeling.shape}.")

    print(f"\n  --- [1.2] Generating Graph Laplacian Eigenmaps ---")
    print(f"    Generating {N_LAPLACIAN_EIGENMAPS} Laplacian Eigenmaps using k={K_FOR_LAPLACIAN_SWM} for SWM...")
    coords_for_laplacian = np.array(list(zip(data_for_modeling.geometry.x, data_for_modeling.geometry.y)))
    try:
        w_laplacian = PysalKNN.from_array(coords_for_laplacian, k=K_FOR_LAPLACIAN_SWM)
        L = laplacian(w_laplacian.sparse, normed=True)
        k_eigsh = min(N_LAPLACIAN_EIGENMAPS + 1, L.shape[0] - 2)
        LAPLACIAN_COLS = []
        if k_eigsh < 2:
            print("    Warning: Not enough samples or N_LAPLACIAN_EIGENMAPS too low for eigsh. Skipping Laplacian Eigenmaps.")
        else:
            vals, vecs = eigsh(L, k=k_eigsh, which='SM')
            eigenmaps_to_add = vecs[:, 1:min(vecs.shape[1], N_LAPLACIAN_EIGENMAPS + 1)]
            eigenmap_df_cols = {f'LAPEIG_{i+1}': eigenmaps_to_add[:, i] for i in range(eigenmaps_to_add.shape[1])}
            eigenmap_df = pd.DataFrame(eigenmap_df_cols, index=data_for_modeling.index)
            data_for_modeling = data_for_modeling.join(eigenmap_df)
            LAPLACIAN_COLS = list(eigenmap_df.columns)
            print(f"    Generated and added {len(LAPLACIAN_COLS)} Laplacian Eigenmaps.")
    except Exception as e_lap:
        print(f"    Error generating Laplacian Eigenmaps: {e_lap}. Proceeding without them.")
        LAPLACIAN_COLS = []
            
    print(f"    Saving data with Laplacian Eigenmaps to: {PREPPED_DATA_WITH_BASIS_PATH}")
    df_to_save_gpkg_basis = data_for_modeling.copy()
    if df_to_save_gpkg_basis.index.name == IDENTIFIER_COL: df_to_save_gpkg_basis = df_to_save_gpkg_basis.reset_index()
    elif IDENTIFIER_COL not in df_to_save_gpkg_basis.columns : df_to_save_gpkg_basis[IDENTIFIER_COL] = df_to_save_gpkg_basis.index
    df_to_save_gpkg_basis.to_file(PREPPED_DATA_WITH_BASIS_PATH, driver="GPKG", layer="data")
print(f"--- Stage 1 Processing Complete ---")


# --- Stage 2: Base Model OOF Prediction Generation ---
print(f"\n--- Stage 2: Base Model OOF Prediction Generation ---")

print(f"\n  --- [2.1] Generating Spatial CV Folds ---")
spatial_cv_folds = []
gkf_cv = GroupKFold(n_splits=N_SPATIAL_FOLDS)
group_ids_for_cv = data_for_modeling[IDENTIFIER_COL] if IDENTIFIER_COL in data_for_modeling.columns else data_for_modeling.index
for fold_num_counter_cv, (train_pos_idx_cv, val_pos_idx_cv) in enumerate(gkf_cv.split(X=data_for_modeling, y=data_for_modeling[TARGET], groups=group_ids_for_cv)):
    spatial_cv_folds.append({
        'fold': fold_num_counter_cv + 1,
        'train_indices': data_for_modeling.index[train_pos_idx_cv],
        'val_indices': data_for_modeling.index[val_pos_idx_cv]
    })
print(f"    Generated {len(spatial_cv_folds)} spatial CV folds.")

gwr_oof_predictions = pd.Series(np.nan, index=data_for_modeling.index)
gwrf_oof_predictions = pd.Series(np.nan, index=data_for_modeling.index)
ggpgamsvc_oof_predictions = pd.Series(np.nan, index=data_for_modeling.index)

# --- Function Definitions ---
def get_gwr_oof_predictions_spatial_cv(data_gdf_full, predictors_list_for_gwr, target_col, spatial_cv_folds_list):
    oof_preds = pd.Series(np.nan, index=data_gdf_full.index)
    print(f"    Generating GWR OOF predictions with predictors: {predictors_list_for_gwr}...")
    for fold_info in spatial_cv_folds_list:
        fold_num = fold_info['fold']; train_idx_orig = fold_info['train_indices']; val_idx_orig = fold_info['val_indices']
        train_gdf = data_gdf_full.loc[train_idx_orig]; val_gdf = data_gdf_full.loc[val_idx_orig]
        print(f"      GWR Fold {fold_num}: Train {len(train_gdf)}, Val {len(val_gdf)}")
        min_obs_for_gwr = len(predictors_list_for_gwr) + 5
        if len(train_gdf) < min_obs_for_gwr or len(val_gdf) == 0:
            print(f"        Skipping GWR Fold {fold_num} (insufficient data)."); oof_preds.loc[val_idx_orig] = np.nan; continue
        scaler_fold = StandardScaler()
        X_train_fold_scaled_np = scaler_fold.fit_transform(train_gdf[predictors_list_for_gwr])
        X_val_fold_scaled_np = scaler_fold.transform(val_gdf[predictors_list_for_gwr])
        y_train_fold_np = train_gdf[target_col].values.reshape(-1,1)
        coords_train_fold = np.array(list(zip(train_gdf.geometry.x, train_gdf.geometry.y)))
        coords_val_fold = np.array(list(zip(val_gdf.geometry.x, val_gdf.geometry.y)))
        try:
            if len(np.unique(coords_train_fold, axis=0)) < X_train_fold_scaled_np.shape[1] + 2:
                raise ValueError("Not enough unique locations for GWR Sel_BW.")
            gwr_selector_fold = Sel_BW(coords_train_fold, y_train_fold_np, X_train_fold_scaled_np, kernel='gaussian', fixed=False, spherical=False)
            bw_fold = gwr_selector_fold.search(search_method='golden_section', criterion='AICc', tol=1e-5)
            min_adaptive_k = 500
            bw_fold_int = max(min_adaptive_k, X_train_fold_scaled_np.shape[1] + 2, int(bw_fold)) # Added check against num predictors
            bw_fold_int = min(bw_fold_int, len(train_gdf) - 1 if len(train_gdf) > 1 else 1)
            bw_fold_int = max(1, bw_fold_int)
            print(f"        Fold {fold_num} GWR: Selected adaptive k = {bw_fold_int}")
            gwr_model_fold = GWR(coords_train_fold, y_train_fold_np, X_train_fold_scaled_np, bw=bw_fold_int, kernel='gaussian', fixed=False, spherical=False)
            gwr_model_fold.fit()
            if len(coords_val_fold) > 0:
                val_preds_struct = gwr_model_fold.predict(coords_val_fold, X_val_fold_scaled_np)
                oof_preds.loc[val_idx_orig] = val_preds_struct.predictions.flatten()
            print(f"        Fold {fold_num} GWR processed.")
        except Exception as e_gwr_fold:
            print(f"        Error GWR Fold {fold_num}: {e_gwr_fold}. Attempting OLS fallback...")
            try:
                ols_model_fallback = sm.OLS(y_train_fold_np, sm.add_constant(X_train_fold_scaled_np)).fit()
                ols_predictions_fallback = ols_model_fallback.predict(sm.add_constant(X_val_fold_scaled_np))
                oof_preds.loc[val_idx_orig] = np.array(ols_predictions_fallback).flatten()
                print(f"        Fold {fold_num}: GWR failed, used OLS fallback.")
            except Exception as e_ols_fallback:
                print(f"        Fold {fold_num}: GWR & OLS fallback failed: {e_ols_fallback}"); oof_preds.loc[val_idx_orig] = np.nan
    return oof_preds

def get_gwrf_oof_predictions_cv_with_lap_pca(
    data_gdf_full, predictors_list, target_col, spatial_cv_folds_list,
    laplacian_cols=None,           # List of Laplacian column names
    n_lap_pca=10,                  # Number of Laplacian PCs to use
    identifier_col_name='OBJECTID',
    bandwidth=100,
    adaptive_bandwidth=True,
    rf_n_estimators=100,
    rf_min_samples_leaf=10,
    rf_max_features='sqrt',
    n_jobs_rf=1
):
    """
    Returns: oof_preds (pd.Series), oof_diagnostics (pd.DataFrame)
    """
    oof_preds = pd.Series(np.nan, index=data_gdf_full.index)
    diag_cols = [f"imp_{p}" for p in predictors_list]  # For base predictors
    if laplacian_cols:
        diag_cols += [f"imp_lap_pca_{i+1}" for i in range(n_lap_pca)]
    diag_cols += ['local_r2']
    oof_diagnostics = pd.DataFrame(np.nan, index=data_gdf_full.index, columns=diag_cols)
    print(f"\n  Generating GWRF OOF predictions (Bi-square, LapPCA, Bandwidth: {'adaptive k=' if adaptive_bandwidth else 'fixed m='}{bandwidth}, n_est={rf_n_estimators}, min_leaf={rf_min_samples_leaf})...")

    # Check for missing predictors
    if not all(p in data_gdf_full.columns for p in predictors_list):
        missing_p = [p for p in predictors_list if p not in data_gdf_full.columns]
        print(f"    ERROR: Missing predictors for GWRF: {missing_p}. Halting.")
        return oof_preds, oof_diagnostics
    if laplacian_cols and not all(l in data_gdf_full.columns for l in laplacian_cols):
        missing_lap = [l for l in laplacian_cols if l not in data_gdf_full.columns]
        print(f"    ERROR: Missing Laplacian columns for GWRF: {missing_lap}. Halting.")
        return oof_preds, oof_diagnostics

    for fold_num_idx, fold_info in enumerate(spatial_cv_folds_list):
        fold_num = fold_info['fold']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        train_gdf = data_gdf_full.loc[train_idx].copy()
        val_gdf = data_gdf_full.loc[val_idx].copy()
        print(f"    GWRF Fold {fold_num}: Train {len(train_gdf)}, Val {len(val_gdf)}")

        if len(train_gdf) < 20 or len(val_gdf) == 0:
            print(f"      Skipping GWRF Fold {fold_num} due to insufficient data.")
            continue

        # Scale predictors
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(train_gdf[predictors_list]), columns=predictors_list, index=train_gdf.index)
        X_val = pd.DataFrame(scaler.transform(val_gdf[predictors_list]), columns=predictors_list, index=val_gdf.index)

        # Foldwise Laplacian PCA
        if laplacian_cols and n_lap_pca > 0:
            # Replace NaNs with col mean for PCA stability
            train_laps = train_gdf[laplacian_cols].fillna(train_gdf[laplacian_cols].mean())
            val_laps = val_gdf[laplacian_cols].fillna(train_gdf[laplacian_cols].mean())  # Use train means for val

            lap_scaler = StandardScaler()
            train_laps_scaled = lap_scaler.fit_transform(train_laps)
            val_laps_scaled = lap_scaler.transform(val_laps)

            pca = PCA(n_components=n_lap_pca, random_state=42)
            train_lap_pca = pca.fit_transform(train_laps_scaled)
            val_lap_pca = pca.transform(val_laps_scaled)

            # Turn into DataFrames with nice col names
            lap_pca_names = [f'lap_pca_{i+1}' for i in range(n_lap_pca)]
            train_lap_pca_df = pd.DataFrame(train_lap_pca, columns=lap_pca_names, index=train_gdf.index)
            val_lap_pca_df = pd.DataFrame(val_lap_pca, columns=lap_pca_names, index=val_gdf.index)

            # Concatenate with scaled predictors
            X_train = pd.concat([X_train, train_lap_pca_df], axis=1)
            X_val = pd.concat([X_val, val_lap_pca_df], axis=1)

        y_train = train_gdf[target_col]

        train_coords = train_gdf[['projected_X', 'projected_Y']].values
        val_coords = val_gdf[['projected_X', 'projected_Y']].values
        if len(train_coords) == 0:
            print(f"      Skipping GWRF Fold {fold_num}, no training coordinates.")
            continue
        ball_tree = BallTree(train_coords, metric='euclidean')

        fold_preds = []
        fold_diags = []
        for i in range(len(val_gdf)):
            val_point_coord = val_coords[i].reshape(1, -1)
            distances, _ = ball_tree.query(val_point_coord, k=len(train_gdf))
            distances = distances[0]

            if adaptive_bandwidth:
                k_adaptive = int(min(bandwidth, len(train_gdf) - 1))
                local_bw = distances[k_adaptive] if k_adaptive < len(distances) else distances[-1]
            else:
                local_bw = bandwidth

            # Bi-square kernel
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = (1 - (distances / local_bw) ** 2) ** 2
                weights[distances >= local_bw] = 0.0

            if np.sum(weights) < 1e-6:
                fold_preds.append(np.nan)
                fold_diags.append([np.nan] * len(diag_cols))
                continue

            try:
                local_rf = RandomForestRegressor(
                    n_estimators=rf_n_estimators,
                    min_samples_leaf=rf_min_samples_leaf,
                    max_features=rf_max_features,
                    random_state=42,
                    n_jobs=n_jobs_rf
                )
                local_rf.fit(X_train, y_train, sample_weight=weights)
                pred = local_rf.predict(X_val.iloc[[i]])[0]
                # Diagnostics
                local_r2 = r2_score(y_train, local_rf.predict(X_train), sample_weight=weights)
                feat_imps = local_rf.feature_importances_
                # Feature importances may not line up with diag_cols order; pad or slice
                n_feats = len(diag_cols) - 1
                imp_vec = list(feat_imps) + [np.nan] * (n_feats - len(feat_imps))
                imp_vec = imp_vec[:n_feats]  # Truncate if too many
                fold_diags.append(imp_vec + [local_r2])
                fold_preds.append(pred)
            except Exception as e_rf:
                print(f"      RF error on val idx {i}: {e_rf}")
                fold_preds.append(np.nan)
                fold_diags.append([np.nan] * len(diag_cols))

        oof_preds.loc[val_idx] = fold_preds
        if fold_diags:
            oof_diagnostics.loc[val_idx] = np.array(fold_diags)
        valid_preds = len(fold_preds) - pd.Series(fold_preds).isnull().sum()
        print(f"      GWRF Fold {fold_num} processed. {valid_preds}/{len(val_gdf)} valid predictions.")

    print("  GWRF OOF (LapPCA) generation complete.")
    return oof_preds, oof_diagnostics

# [2.2] GWR OOF
print(f"\n  --- [2.2] GWR OOF Prediction ---")
if not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(gwr_oof_file):
    print(f"    Loading existing GWR OOF from {gwr_oof_file}")
    gwr_df_loaded = pd.read_csv(gwr_oof_file)
    gwr_oof_predictions = data_for_modeling[IDENTIFIER_COL].map(pd.Series(gwr_df_loaded['gwr_oof_prediction'].values, index=gwr_df_loaded[IDENTIFIER_COL])).reindex(data_for_modeling.index)
else:
    gwr_oof_predictions = get_gwr_oof_predictions_spatial_cv(data_for_modeling, BASE_MODEL_PREDICTORS, TARGET, spatial_cv_folds)
    pd.DataFrame({IDENTIFIER_COL: data_for_modeling[IDENTIFIER_COL], TARGET: data_for_modeling[TARGET], 'gwr_oof_prediction': gwr_oof_predictions}).to_csv(gwr_oof_file, index=False)
if gwr_oof_predictions.notna().sum() >= 2:
    valid_preds = gwr_oof_predictions.dropna()
    r2 = r2_score(data_for_modeling.loc[valid_preds.index, TARGET], valid_preds); rmse = np.sqrt(mean_squared_error(data_for_modeling.loc[valid_preds.index, TARGET], valid_preds))
    print(f"      GWR OOF Performance: R2 = {r2:.4f}, RMSE = {rmse:.4f}"); pd.DataFrame([{'model':'GWR_OOF', 'R2':r2, 'RMSE':rmse}]).to_csv(gwr_performance_file, index=False)

# [2.3] GWRF OOF
# [2.3] GWRF OOF
print(f"\n  --- [2.3] GWRF OOF Prediction ---")
if INCLUDE_GWRF_IN_ENSEMBLE:
    if not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(gwrf_oof_file):
        print(f"    Loading existing GWRF OOF from {gwrf_oof_file}")
        gwrf_df_loaded = pd.read_csv(gwrf_oof_file)
        gwrf_oof_predictions = data_for_modeling[IDENTIFIER_COL].map(pd.Series(gwrf_df_loaded['gwrf_oof_prediction'].values, index=gwrf_df_loaded[IDENTIFIER_COL])).reindex(data_for_modeling.index)
    else:
        gwrf_oof_predictions, _ = get_gwrf_oof_predictions_cv_with_lap_pca(
            data_gdf_full=data_for_modeling,
            predictors_list=BASE_MODEL_PREDICTORS,
            target_col=TARGET,
            spatial_cv_folds_list=spatial_cv_folds,
            laplacian_cols=LAPLACIAN_COLS,
            n_lap_pca=10
        )
        pd.DataFrame({IDENTIFIER_COL: data_for_modeling[IDENTIFIER_COL], TARGET: data_for_modeling[TARGET], 'gwrf_oof_prediction': gwrf_oof_predictions}).to_csv(gwrf_oof_file, index=False)
    if gwrf_oof_predictions.notna().sum() >= 2:
        valid_preds = gwrf_oof_predictions.dropna()
        r2 = r2_score(data_for_modeling.loc[valid_preds.index, TARGET], valid_preds); rmse = np.sqrt(mean_squared_error(data_for_modeling.loc[valid_preds.index, TARGET], valid_preds))
        print(f"      GWRF OOF Performance: R2 = {r2:.4f}, RMSE = {rmse:.4f}"); pd.DataFrame([{'model':f'GWRF_OOF{ABLATION_SUFFIX}', 'R2':r2, 'RMSE':rmse}]).to_csv(gwrf_performance_file, index=False)
else: print(f"    Skipping GWRF OOF generation as INCLUDE_GWRF_IN_ENSEMBLE is False.")

# [2.4] GGP-GAM-SVC OOF (via R)
print(f"\n  --- [2.4] GGP-GAM-SVC OOF Prediction (via R) ---")
if not OVERWRITE_EXISTING_OUTPUTS and os.path.exists(ggpgamsvc_oof_file):
    print(f"    Loading existing GGP-GAM-SVC OOF from {ggpgamsvc_oof_file}")
    ggpgamsvc_df_loaded = pd.read_csv(ggpgamsvc_oof_file)
    ggpgamsvc_oof_predictions = data_for_modeling[IDENTIFIER_COL].map(pd.Series(ggpgamsvc_df_loaded['ggpgamsvc_oof_prediction'].values, index=ggpgamsvc_df_loaded[IDENTIFIER_COL])).reindex(data_for_modeling.index)
else:
    print(f"    Generating GGP-GAM-SVC OOF predictions via R...")
    ggpgamsvc_oof_predictions.loc[:] = np.nan 
    all_folds_successful_gam = True
    for fold_info in spatial_cv_folds:
        fold_num = fold_info['fold']; train_idx_orig = fold_info['train_indices']; val_idx_orig = fold_info['val_indices']
        train_data_fold = data_for_modeling.loc[train_idx_orig].copy(); val_data_fold = data_for_modeling.loc[val_idx_orig].copy()
        print(f"      GGP-GAM-SVC Fold {fold_num}: Train {len(train_data_fold)}, Val {len(val_data_fold)}")
        if len(train_data_fold) < 50 or len(val_data_fold) == 0:
            print(f"        Skipping GGP-GAM-SVC Fold {fold_num} (insufficient data)."); ggpgamsvc_oof_predictions.loc[val_idx_orig] = np.nan; continue
        cols_for_r_train = list(set([IDENTIFIER_COL, TARGET, 'projected_X', 'projected_Y'] + BASE_MODEL_PREDICTORS))
        actual_cols_for_r_train = [col for col in cols_for_r_train if col in train_data_fold.columns]
        cols_for_r_val = list(set([IDENTIFIER_COL, 'projected_X', 'projected_Y'] + BASE_MODEL_PREDICTORS))
        actual_cols_for_r_val = [col for col in cols_for_r_val if col in val_data_fold.columns]
        train_data_to_save = train_data_fold[actual_cols_for_r_train].copy(); val_data_to_save = val_data_fold[actual_cols_for_r_val].copy()
        temp_train_path, temp_val_path, temp_preds_path = "", "", ""
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', dir=OUTPUT_DIR, newline='', encoding='utf-8') as tf_train, \
                 tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', dir=OUTPUT_DIR, newline='', encoding='utf-8') as tf_val, \
                 tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', dir=OUTPUT_DIR, newline='', encoding='utf-8') as tf_preds:
                temp_train_path = tf_train.name; temp_val_path = tf_val.name; temp_preds_path = tf_preds.name
            train_data_to_save.to_csv(temp_train_path, index=False); val_data_to_save.to_csv(temp_val_path, index=False)
            r_script_command = [R_EXECUTABLE, "--vanilla", "-f", PATH_TO_GGPGAMSVC_R_SCRIPT, "--args", temp_train_path, temp_val_path, IDENTIFIER_COL, temp_preds_path]
            print(f"        Executing R for GGP-GAM-SVC Fold {fold_num}...")
            process = subprocess.run(r_script_command, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            fold_predictions_df = pd.read_csv(temp_preds_path)
            pred_col_from_r = 'ggpgam_oof_prediction'
            if 'ggpgamsvc_oof_prediction' in fold_predictions_df.columns: pred_col_from_r = 'ggpgamsvc_oof_prediction'
            if pred_col_from_r not in fold_predictions_df.columns:
                print(f"        ERROR: Pred col not found in R output Fold {fold_num}. Found: {fold_predictions_df.columns}"); ggpgamsvc_oof_predictions.loc[val_idx_orig] = np.nan; continue
            id_to_pred_map_fold = pd.Series(fold_predictions_df[pred_col_from_r].values, index=fold_predictions_df[IDENTIFIER_COL].astype(data_for_modeling[IDENTIFIER_COL].dtype))
            ggpgamsvc_oof_predictions.loc[val_idx_orig] = val_data_fold[IDENTIFIER_COL].map(id_to_pred_map_fold).values
        except subprocess.CalledProcessError as e:
            print("\n" + "="*80)
            print(f"CRITICAL ERROR: R script failed for GGP-GAM-SVC Fold {fold_num}. Halting GGP-GAM-SVC.")
            print(f"  R Stderr:\n{e.stderr}")
            print(f"  R Stdout:\n{e.stdout}")
            print("="*80 + "\n"); all_folds_successful_gam = False; break
        finally:
            for f_path in [temp_train_path, temp_val_path, temp_preds_path]:
                if os.path.exists(f_path) and not (KEEP_TEMP_PREDS_CSV_FOR_FIRST_R_FOLD and fold_num == 1 and f_path == temp_preds_path):
                    try: os.remove(f_path)
                    except Exception: pass
    if all_folds_successful_gam:
        pd.DataFrame({IDENTIFIER_COL: data_for_modeling[IDENTIFIER_COL], TARGET: data_for_modeling[TARGET], 'ggpgamsvc_oof_prediction': ggpgamsvc_oof_predictions}).to_csv(ggpgamsvc_oof_file, index=False)
        if ggpgamsvc_oof_predictions.notna().sum() >= 2:
            valid_preds = ggpgamsvc_oof_predictions.dropna()
            r2 = r2_score(data_for_modeling.loc[valid_preds.index, TARGET], valid_preds); rmse = np.sqrt(mean_squared_error(data_for_modeling.loc[valid_preds.index, TARGET], valid_preds))
            print(f"      GGP-GAM-SVC OOF Performance: R2 = {r2:.4f}, RMSE = {rmse:.4f}"); pd.DataFrame([{'model':'GGPGAMSVC_OOF', 'R2':r2, 'RMSE':rmse}]).to_csv(ggpgamsvc_performance_file, index=False)
print(f"--- Stage 2 Processing Complete ---")


# --- Stage 2.5: OOF Prediction Sanity Check ---
print(f"\n--- Stage 2.5: OOF Prediction Sanity Check ---")
oof_model_files_map = {
    'gwr': gwr_oof_file,
    'ggpgamsvc': ggpgamsvc_oof_file
}
if INCLUDE_GWRF_IN_ENSEMBLE:
    oof_model_files_map['gwrf'] = gwrf_oof_file

any_base_model_produced_all_nans = False
for model_key, file_path in oof_model_files_map.items():
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if IDENTIFIER_COL in df.columns and len(df.columns) > 1:
                # Attempt to find a prediction column dynamically
                potential_pred_cols = [col for col in df.columns if col != IDENTIFIER_COL and col != TARGET]
                if not potential_pred_cols:
                     print(f"    DEBUG [{model_key.upper()}]: File '{file_path}' does not have an obvious prediction column besides ID and Target.")
                     any_base_model_produced_all_nans = True
                     continue
                pred_col_name = potential_pred_cols[0] # Take the first one found

                total_rows = len(df)
                valid_preds = df[pred_col_name].notna().sum()
                print(f"    DEBUG [{model_key.upper()}]: File '{file_path}' has {valid_preds} valid predictions out of {total_rows} rows in column '{pred_col_name}'.")
                if valid_preds == 0: any_base_model_produced_all_nans = True
            else:
                print(f"    DEBUG [{model_key.upper()}]: File '{file_path}' does not have expected columns or only one column.")
                any_base_model_produced_all_nans = True
        except Exception as e:
            print(f"    DEBUG [{model_key.upper()}]: Could not read or parse file '{file_path}'. Error: {e}")
            any_base_model_produced_all_nans = True
    else:
        print(f"    DEBUG [{model_key.upper()}]: OOF prediction file '{file_path}' not found.")
        any_base_model_produced_all_nans = True
# --- End OOF Sanity Check ---


# --- Stage 3: Meta-Ensemble, SHAP, Deep Kriging, Final Prediction ---
print(f"\n--- Stage 3: Meta-Ensemble, SHAP, Deep Kriging, and Final Prediction ---")

print(f"\n  --- [3.1] Loading Data and OOFs for Meta-Learner ---")
meta_features_df = data_for_modeling[[IDENTIFIER_COL, TARGET, 'projected_X', 'projected_Y', 'geometry'] + LAPLACIAN_COLS].copy()
base_model_oof_cols_for_meta = []

for model_key, file_path in oof_model_files_map.items():
    if os.path.exists(file_path):
        try:
            oof_df = pd.read_csv(file_path)
            if IDENTIFIER_COL in oof_df.columns and len(oof_df.columns) > 1:
                potential_pred_cols = [col for col in oof_df.columns if col != IDENTIFIER_COL and col != TARGET]
                if not potential_pred_cols:
                    print(f"    WARNING: No prediction column found for '{model_key}' in '{file_path}'. It will be EXCLUDED.")
                    continue
                pred_col_to_merge = potential_pred_cols[0]

                if oof_df[pred_col_to_merge].isnull().all():
                    print(f"    WARNING: OOF predictions for '{model_key}' (column '{pred_col_to_merge}') are all NaN. It will be EXCLUDED.")
                    continue
                
                # Explicitly select only the ID and the determined prediction column for the merge
                meta_features_df = pd.merge(meta_features_df, oof_df[[IDENTIFIER_COL, pred_col_to_merge]], on=IDENTIFIER_COL, how='left')
                base_model_oof_cols_for_meta.append(pred_col_to_merge) # Add the actual merged column name
                print(f"    Successfully loaded and merged OOF predictions for '{model_key}' (using column '{pred_col_to_merge}').")
            else:
                print(f"    WARNING: OOF file for '{model_key}' does not contain expected ID and prediction columns. It will be EXCLUDED.")
        except Exception as e_merge:
            print(f"    WARNING: Could not load or merge OOF file for '{model_key}' from {file_path}. It will be EXCLUDED. Error: {e_merge}")

# Ensure no accidental inclusion of TARGET in the feature list for X
all_meta_features_list = [col for col in (base_model_oof_cols_for_meta + ['projected_X', 'projected_Y'] + LAPLACIAN_COLS) if col != TARGET]
# Drop rows where essential OOFs or TARGET might be NaN *before* imputation
cols_to_check_for_na = [TARGET] + base_model_oof_cols_for_meta
meta_features_df.dropna(subset=[col for col in cols_to_check_for_na if col in meta_features_df.columns], inplace=True)

# Impute any remaining NaNs (e.g. in Laplacians if some points had no neighbors)
for col in all_meta_features_list:
    if col in meta_features_df.columns and meta_features_df[col].isnull().any():
        meta_features_df[col].fillna(meta_features_df[col].mean(), inplace=True)

# Create X_meta_aligned using only the columns present in meta_features_df
X_meta_aligned = meta_features_df[[col for col in all_meta_features_list if col in meta_features_df.columns]].copy()
y_meta_aligned = meta_features_df[TARGET].copy()
print(f"    Meta-features prepared. Shape of X_meta_aligned: {X_meta_aligned.shape}, y_meta_aligned: {y_meta_aligned.shape}")

if X_meta_aligned.empty or y_meta_aligned.empty or X_meta_aligned.shape[0] < N_SPATIAL_FOLDS :
    print("\n" + "="*80)
    print("FATAL ERROR: Meta-features DataFrame is empty or has too few samples after loading and cleaning base model predictions.")
    print("This means there are no data points where all successful base models have a valid overlapping prediction, or too few for CV.")
    print("Please check the OOF Sanity Check logs in Stage [2.5] to diagnose the issue.")
    print("="*80 + "\n")
else:
    print(f"\n  --- [3.2] Generating Spatial CV Folds for Meta-Learner ---")
    spatial_cv_folds_meta = []
    gkf_meta = GroupKFold(n_splits=N_SPATIAL_FOLDS)
    for fold_num_counter_meta, (train_pos_idx_meta, val_pos_idx_meta) in enumerate(gkf_meta.split(X=X_meta_aligned, y=y_meta_aligned, groups=meta_features_df.loc[X_meta_aligned.index, IDENTIFIER_COL])):
        spatial_cv_folds_meta.append({
            'fold': fold_num_counter_meta + 1,
            'train_indices': X_meta_aligned.index[train_pos_idx_meta],
            'val_indices': X_meta_aligned.index[val_pos_idx_meta]
        })
    print(f"    Generated {len(spatial_cv_folds_meta)} spatial CV folds for meta-learner.")

    def lgbm_objective(trial, X_train, y_train, X_val, y_val):
        params = {
            'objective': 'regression_l2', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60), 'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    print(f"\n  --- [3.3] Training LightGBM Meta-Ensemble ---")
    LGBM_OOF_PARAMS = {'random_state': 42, 'n_jobs': -1, 'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31}
    lgbm_oof_predictions = pd.Series(np.nan, index=X_meta_aligned.index)
    for fold_info in spatial_cv_folds_meta:
        fold_num = fold_info['fold']; train_idx = fold_info['train_indices']; val_idx = fold_info['val_indices']
        X_train_fold, y_train_fold = X_meta_aligned.loc[train_idx], y_meta_aligned.loc[train_idx]
        X_val_fold, y_val_fold = X_meta_aligned.loc[val_idx], y_meta_aligned.loc[val_idx]
        lgbm_meta_model_fold = lgb.LGBMRegressor(**LGBM_OOF_PARAMS)
        lgbm_meta_model_fold.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(10, verbose=-1)])
        lgbm_oof_predictions.loc[val_idx] = lgbm_meta_model_fold.predict(X_val_fold)
    
    lgbm_oof_df_to_save = pd.DataFrame({IDENTIFIER_COL: meta_features_df.loc[X_meta_aligned.index, IDENTIFIER_COL], TARGET: y_meta_aligned, 'lgbm_ensemble_oof_prediction': lgbm_oof_predictions})
    lgbm_oof_df_to_save.to_csv(lgbm_ensemble_oof_output_path, index=False)
    r2 = r2_score(y_meta_aligned, lgbm_oof_predictions); rmse = np.sqrt(mean_squared_error(y_meta_aligned, lgbm_oof_predictions))
    print(f"    LGBM Ensemble OOF Performance (fixed params): R2 = {r2:.4f}, RMSE = {rmse:.4f}")
    pd.DataFrame([{'model':f'LGBM_ENSEMBLE_OOF{ABLATION_SUFFIX}', 'R2':r2, 'RMSE':rmse}]).to_csv(lgbm_ensemble_performance_path, index=False)

    print(f"\n  Tuning final LGBM Meta-Learner on all available data with Optuna...")
    X_meta_train_opt, X_meta_val_opt, y_meta_train_opt, y_meta_val_opt = train_test_split(X_meta_aligned, y_meta_aligned, test_size=0.2, random_state=42)
    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(lambda trial: lgbm_objective(trial, X_meta_train_opt, y_meta_train_opt, X_meta_val_opt, y_meta_val_opt), n_trials=N_OPTUNA_TRIALS_LGBM, n_jobs=1)
    best_lgbm_params = study_lgbm.best_trial.params
    import json # Ensure json is imported
    with open(tuned_lgbm_params_path, 'w') as f: json.dump(best_lgbm_params, f, indent=4)
    final_lgbm_meta_model = lgb.LGBMRegressor(**best_lgbm_params, random_state=42, n_jobs=-1, verbose=-1).fit(X_meta_aligned, y_meta_aligned)
    joblib.dump(final_lgbm_meta_model, final_lgbm_model_path)
    print(f"    Saved final tuned LGBM meta-model to: {final_lgbm_model_path}")

    print(f"\n  --- [3.4] SHAP Interpretation for LightGBM Meta-Model ---")
    try:
        explainer = shap.TreeExplainer(final_lgbm_meta_model)
        shap_values = explainer.shap_values(X_meta_aligned) # Make sure X_meta_aligned is not empty
        if X_meta_aligned.shape[0] > 0: # Check if X_meta_aligned has rows
            shap.summary_plot(shap_values, X_meta_aligned, show=False, plot_size=(12, max(8, int(X_meta_aligned.shape[1]*0.35)))); plt.savefig(shap_summary_plot_path, bbox_inches='tight'); plt.close()
            print(f"    SHAP summary plot saved to: {shap_summary_plot_path}")
            df_feature_importance = pd.DataFrame({'feature': X_meta_aligned.columns, 'importance': final_lgbm_meta_model.feature_importances_}).sort_values(by='importance', ascending=False)
            top_n_features_for_shap_dependence = min(5, len(df_feature_importance))
            for i in range(top_n_features_for_shap_dependence):
                if top_n_features_for_shap_dependence == 0 : break # No features to plot
                top_feature = df_feature_importance.iloc[i]['feature']
                shap.dependence_plot(top_feature, shap_values, X_meta_aligned, show=False, interaction_index="auto")
                dep_plot_path = os.path.join(shap_dependence_plot_dir, f'shap_dependence_{top_feature}.png')
                plt.savefig(dep_plot_path, bbox_inches='tight'); plt.close()
                print(f"      Saved SHAP dependence plot for '{top_feature}'.")
        else:
            print("    Skipping SHAP plots as X_meta_aligned is empty.")
    except Exception as e_shap: print(f"    Error during SHAP interpretation: {e_shap}. Skipping SHAP analysis.")

    # In Final_Spatial_Model_FOLD.py, after X_meta_aligned is ready:
    X_meta_aligned_output_path = os.path.join(OUTPUT_DIR, f'X_meta_aligned_for_shap{ABLATION_SUFFIX}.csv')
    X_meta_aligned.to_csv(X_meta_aligned_output_path, index=False)
    print(f"    Saved X_meta_aligned for SHAP analysis to: {X_meta_aligned_output_path}")

    print(f"\n  --- [3.5] Training Deep Kriging NN on Ensemble Residuals ---")
    lgbm_oof_aligned_for_resid = lgbm_oof_predictions.reindex(y_meta_aligned.index).dropna()
    ensemble_residuals_for_dk = y_meta_aligned.loc[lgbm_oof_aligned_for_resid.index] - lgbm_oof_aligned_for_resid
    valid_dk_indices = ensemble_residuals_for_dk.index
    dk_input_features_df = meta_features_df.loc[valid_dk_indices, ['projected_X', 'projected_Y'] + LAPLACIAN_COLS].copy()
    dk_target_series = ensemble_residuals_for_dk.copy()
    deep_kriging_corrections_on_meta_aligned = pd.Series(0, index=X_meta_aligned.index) 

    if not dk_input_features_df.empty and not dk_target_series.empty:
        scaler_dk = StandardScaler(); dk_input_features_scaled = scaler_dk.fit_transform(dk_input_features_df)
        joblib.dump(scaler_dk, dk_scaler_path)
        dk_model = Sequential([Input(shape=(dk_input_features_scaled.shape[1],)), Dense(128, activation='relu'), Dropout(0.2),Dense(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])
        dk_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        early_stopping_dk = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        if len(dk_target_series) >= N_SPATIAL_FOLDS * 2 :
            X_dk_train_full, X_dk_val_full, y_dk_train_full, y_dk_val_full = train_test_split(dk_input_features_scaled, dk_target_series, test_size=0.2, random_state=42)
            if len(X_dk_train_full) > 0 and len(X_dk_val_full) > 0:
                print(f"    DK Model training: Train {len(X_dk_train_full)}, Val {len(X_dk_val_full)}")
                dk_model.fit(X_dk_train_full, y_dk_train_full, epochs=100, batch_size=256, verbose=0, validation_data=(X_dk_val_full, y_dk_val_full), callbacks=[early_stopping_dk])
                dk_pred_features_for_all_meta = meta_features_df.loc[X_meta_aligned.index, ['projected_X', 'projected_Y'] + LAPLACIAN_COLS].copy()
                # Ensure scaler_dk is used for transforming these features
                dk_pred_features_for_all_meta_scaled = scaler_dk.transform(dk_pred_features_for_all_meta)
                deep_kriging_corrections_on_meta_aligned = pd.Series(dk_model.predict(dk_pred_features_for_all_meta_scaled).flatten(), index=X_meta_aligned.index)
                dk_model.save(deep_kriging_model_path); print(f"    Deep Kriging model trained and saved to: {deep_kriging_model_path}")
            else: print("    Skipping DK training: Not enough data after train/val split for DK.")
        else: print("    Skipping DK training: Not enough target series data for DK.")
    else: print("    Skipping DK training: Empty input features or target series for DK.")
    
    print(f"\n  --- [3.6] Final Prediction and Evaluation ---")
    final_predictions_df = pd.DataFrame(index=X_meta_aligned.index)
    final_predictions_df[IDENTIFIER_COL] = meta_features_df.loc[X_meta_aligned.index, IDENTIFIER_COL]
    final_predictions_df[TARGET] = y_meta_aligned
    final_predictions_df['lgbm_ensemble_pred'] = lgbm_oof_predictions
    final_predictions_df['deep_kriging_correction'] = deep_kriging_corrections_on_meta_aligned.reindex(final_predictions_df.index).fillna(0)
    final_predictions_df['final_combined_prediction'] = final_predictions_df['lgbm_ensemble_pred'] + final_predictions_df['deep_kriging_correction']
    final_predictions_df.to_csv(final_prediction_output_path, index=False)
    print(f"    Final combined predictions saved to: {final_prediction_output_path}")
    valid_final_preds = final_predictions_df.dropna(subset=[TARGET, 'final_combined_prediction'])
    if not valid_final_preds.empty and len(valid_final_preds) >=2:
        final_r2 = r2_score(valid_final_preds[TARGET], valid_final_preds['final_combined_prediction']); final_rmse = np.sqrt(mean_squared_error(valid_final_preds[TARGET], valid_final_preds['final_combined_prediction']))
        print(f"    Final Combined Model Performance: R2 = {final_r2:.4f}, RMSE = {final_rmse:.4f}"); pd.DataFrame([{'model':f'FINAL_COMBINED_MODEL{ABLATION_SUFFIX}', 'R2':final_r2, 'RMSE':rmse}]).to_csv(final_performance_path, index=False)
        final_residuals_series = valid_final_preds[TARGET] - valid_final_preds['final_combined_prediction']
        gdf_for_moran_final = gpd.GeoDataFrame(meta_features_df.loc[final_residuals_series.index], geometry='geometry')
        if not gdf_for_moran_final.empty:
            coords_moran_final = np.array(list(zip(gdf_for_moran_final.geometry.x, gdf_for_moran_final.geometry.y)))
            w_moran_final = PysalKNN.from_array(coords_moran_final, k=K_FOR_MORAN_SWM); w_moran_final.transform = 'R'
            mi_final = Moran(final_residuals_series.values, w_moran_final, permutations=999)
            print(f"    Moran's I for Final Combined Residuals: {mi_final.I:.4f} (p-value: {mi_final.p_sim:.4f})"); pd.DataFrame([{'morans_I': mi_final.I, 'p_value_sim': mi_final.p_sim}]).to_csv(final_residual_morans_path, index=False)

    print(f"\n--- Stage 4: Visualizing Laplacian Eigenmaps ---")
    if LAPLACIAN_COLS and len(LAPLACIAN_COLS) >= 3:
        print(f"  Visualizing the first 3 Laplacian Eigenmaps...")
        eig_map_1, eig_map_2, eig_map_3 = LAPLACIAN_COLS[0], LAPLACIAN_COLS[1], LAPLACIAN_COLS[2]
        if all(c in data_for_modeling.columns for c in [eig_map_1, eig_map_2, eig_map_3]): # Check if actual columns exist in the full data_for_modeling
            fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
            # Use data_for_modeling for visualization as meta_features_df might be smaller
            sample_size = min(5000, len(data_for_modeling)); sampled_data = data_for_modeling.sample(n=sample_size, random_state=42)
            scatter = ax.scatter(sampled_data[eig_map_1], sampled_data[eig_map_2], sampled_data[eig_map_3], c=sampled_data[TARGET], cmap='viridis', s=10)
            ax.set_xlabel(eig_map_1); ax.set_ylabel(eig_map_2); ax.set_zlabel(eig_map_3)
            ax.set_title('3D Manifold of First 3 Laplacian Eigenmaps (Colored by Target Variable)')
            cbar = fig.colorbar(scatter, shrink=0.5, aspect=10); cbar.set_label(TARGET)
            manifold_plot_path = os.path.join(OUTPUT_DIR, f'laplacian_eigenmaps_3d_manifold{ABLATION_SUFFIX}.png')
            plt.savefig(manifold_plot_path, bbox_inches='tight'); plt.close(fig)
            print(f"  3D Manifold plot saved to: {manifold_plot_path}")
    elif LAPLACIAN_COLS: print(f"  Found {len(LAPLACIAN_COLS)} Eigenmaps, need 3 for 3D plot.")
    else: print("  No Laplacian Eigenmaps. Skipping 3D plot.")

print(f"\n--- Comprehensive Spatial ML Workflow (Definitive v3) Complete ---")