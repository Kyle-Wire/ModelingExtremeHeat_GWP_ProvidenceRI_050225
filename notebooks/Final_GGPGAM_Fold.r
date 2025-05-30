# run_ggpgam_svc_fold.R (Version 4 - Enhanced for Large Sample Sizes)

# --- 0. Load Libraries ---
library(mgcv)
library(readr)
library(dplyr)

# --- 1. Get Command Line Arguments (Passed from Python) ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Four arguments must be supplied from Python: <train_csv_path> <val_csv_path> <identifier_col_name> <output_preds_csv_path>", call. = FALSE)
}
train_csv_path <- args[1]
val_csv_path <- args[2]
identifier_col_name <- args[3] 
output_preds_csv_path <- args[4]

cat("R SCRIPT: Starting GGP-GAM with SVCs for one fold.\n")
cat("R SCRIPT: Training data CSV:", train_csv_path, "\n")
cat("R SCRIPT: Output predictions CSV:", output_preds_csv_path, "\n")

# --- 2. Load Data ---
train_data_df <- readr::read_csv(train_csv_path, show_col_types = FALSE)
val_data_df <- readr::read_csv(val_csv_path, show_col_types = FALSE)

cat("R SCRIPT: Training data loaded. Rows:", nrow(train_data_df), "Cols:", ncol(train_data_df), "\n")
cat("R SCRIPT: Validation data loaded. Rows:", nrow(val_data_df), "Cols:", ncol(val_data_df), "\n")

# --- 3. Prepare Data ---
target_variable_name <- "AAT_z" 
base_predictor_names <- c("Distance_from_water_m", "Pct_Impervious", "Pct_Canopy", "Pct_GreenSpace", "Elevation_m")
base_predictor_names <- base_predictor_names[base_predictor_names %in% names(train_data_df)]
if (length(base_predictor_names) == 0) stop("No valid base predictors found.")
coord_x_name <- "projected_X"
coord_y_name <- "projected_Y"

# --- 4. Construct GAM Formula ---
n_obs <- nrow(train_data_df)
k_main_spatial <- max(150, floor(n_obs / 300))
k_svc_interaction <- max(40, floor(n_obs / 1000))

linear_terms <- paste(base_predictor_names, collapse = " + ")
main_spatial_term <- paste0("s(", coord_x_name, ", ", coord_y_name, ", bs=\"tp\", k=", k_main_spatial, ")")
svc_terms <- sapply(base_predictor_names, function(pred_name) {
  paste0("s(", coord_x_name, ", ", coord_y_name, ", by=", pred_name, ", bs=\"tp\", k=", k_svc_interaction, ")")
})
svc_terms_str <- paste(svc_terms, collapse = " + ")
formula_str <- paste(target_variable_name, "~", linear_terms, "+", main_spatial_term, "+", svc_terms_str)
gam_formula <- as.formula(formula_str)

cat("R SCRIPT: Using GAM formula:\n", formula_str, "\n")

# --- 5. Fit Model ---
cat("R SCRIPT: Fitting mgcv::bam model...\n")
ggpgam_model <- NULL
tryCatch({
  ggpgam_model <- mgcv::bam(
    formula = gam_formula,
    data = train_data_df,
    method = "fREML",
    select = TRUE
  )
  cat("R SCRIPT: Model fitting complete.\n")
}, error = function(e) {
  cat("R SCRIPT: Model fitting error:", conditionMessage(e), "\n")
})

# Optional: Print model summary to file
sink("gam_summary.txt")
if (!is.null(ggpgam_model)) print(summary(ggpgam_model))
sink()

# Optional: Log R^2
if (!is.null(ggpgam_model)) {
  r2_train <- 1 - sum(residuals(ggpgam_model)^2) / sum((train_data_df[[target_variable_name]] - mean(train_data_df[[target_variable_name]]))^2)
  cat("R SCRIPT: R^2 on training data:", round(r2_train, 4), "\n")
}

# --- 6. Predict on Validation Set ---
val_predictions <- rep(NA, nrow(val_data_df))
if (!is.null(ggpgam_model) && nrow(val_data_df) > 0) {
  tryCatch({
    pred_object <- predict(ggpgam_model, newdata = val_data_df, type = "response")
    if (is.numeric(pred_object) && length(pred_object) == nrow(val_data_df)) {
      val_predictions <- pred_object
    }
  }, error = function(e) {
    cat("R SCRIPT: Prediction error:", conditionMessage(e), "\n")
  })
} else {
  cat("R SCRIPT: Skipping prediction.\n")
}

# --- 7. Save Predictions ---
output_df <- data.frame(
  val_data_df[[identifier_col_name]],
  ggpgam_oof_prediction = val_predictions
)
colnames(output_df)[1] <- identifier_col_name
readr::write_csv(output_df, output_preds_csv_path)
cat("R SCRIPT: Predictions saved to:", output_preds_csv_path, "\n")