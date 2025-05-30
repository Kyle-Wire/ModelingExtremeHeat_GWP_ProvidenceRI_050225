import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# —— Config ——
CSV_PATH = 'with_two_form_indices.csv'
TARGET = 'AAT_z'
COORDS_COLS = ['POINT_X', 'POINT_Y']
ALL_PREDICTORS = [
    'Elevation_m', 'Distance_from_water_m', 'Pct_Canopy', 'Pct_Impervious',
    'Pct_Building_Coverage', 'Pct_GreenSpace', 'Pct_Density',
    'Mean_Bldg_Height_m', 'mean_road_Width_m', 'Canyon_Ratio',
    'VertForm_z', 'HorizForm_z'
]
SAMPLE_SIZE = 5000
K_NEIGHBORS = 500  # number of neighbors for adaptive bandwidth

# —— Load & Sample Data ——
df = pd.read_csv(CSV_PATH)
# Ensure numeric & drop missing
df = df.dropna(subset=[TARGET] + ALL_PREDICTORS + COORDS_COLS)
for col in [TARGET] + ALL_PREDICTORS + COORDS_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=[TARGET] + ALL_PREDICTORS + COORDS_COLS)
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# —— Prepare arrays ——
X = df_sample[ALL_PREDICTORS].values
y = df_sample[TARGET].values
coords = df_sample[COORDS_COLS].values

# —— Tune ElasticNet globally ——
print("\n--- Tuning global ElasticNet hyperparameters ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
enet_cv = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, random_state=0)
enet_cv.fit(X_scaled, y)
alpha_opt = enet_cv.alpha_
l1_opt = enet_cv.l1_ratio_
print(f"Selected alpha={alpha_opt:.4f}, l1_ratio={l1_opt:.2f}\n")

# —— Adaptive bandwidth via k-NN ——
print(f"--- Computing adaptive bandwidths using {K_NEIGHBORS}-NN ---")
nn = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(coords)
distances, _ = nn.kneighbors(coords)
# bandwidth_i: distance to k-th neighbor for each i
bandwidths = distances[:, -1]

# —— Compute GWEN local coefficients ——
print("--- Computing local ElasticNet coefficients (GWEN) ---")
n, p = X_scaled.shape
local_coefs = np.zeros((n, p+1))
for i in range(n):
    bw_i = bandwidths[i]
    dist_i = np.linalg.norm(coords - coords[i], axis=1)
    w = np.exp(- (dist_i / bw_i)**2)
    sqrt_w = np.sqrt(w)
    Xw = X_scaled * sqrt_w[:, None]
    yw = y * sqrt_w
    enet = ElasticNet(alpha=alpha_opt, l1_ratio=l1_opt, fit_intercept=True)
    enet.fit(Xw, yw)
    local_coefs[i, 0] = enet.intercept_
    local_coefs[i, 1:] = enet.coef_
    if (i+1) % 500 == 0:
        print(f"Fitted local model {i+1}/{n}")

# —— Summarize variable importance ——
print("\n--- Summarizing GWEN variable importance ---")
importance = []
for j, var in enumerate(ALL_PREDICTORS):
    coefs_j = local_coefs[:, j+1]
    nonzero_pct = np.mean(coefs_j != 0)
    mean_abs = np.mean(np.abs(coefs_j[coefs_j != 0])) if nonzero_pct>0 else 0.0
    median_abs = np.median(np.abs(coefs_j[coefs_j != 0])) if nonzero_pct>0 else 0.0
    score = nonzero_pct * mean_abs
    importance.append({
        'predictor': var,
        'nonzero_pct': nonzero_pct,
        'mean_abs_coef': mean_abs,
        'median_abs_coef': median_abs,
        'importance_score': score
    })
imp_df = pd.DataFrame(importance).sort_values('importance_score', ascending=False)
imp_df.to_csv('gwen_variable_importance.csv', index=False)
print(imp_df)

# —— End of pipeline ——
print("\n✅ GWEN variable selection complete: results in 'gwen_variable_importance.csv'")
