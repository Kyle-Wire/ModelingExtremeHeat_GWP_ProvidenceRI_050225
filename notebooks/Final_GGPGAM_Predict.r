# GGPGAM_Predict.r
#
# PURPOSE:
# 1. Load a pre-trained GAM model object (.rds).
# 2. Load a new dataset to predict on.
# 3. Generate predictions and save them to a CSV.

# --- 0. Load Libraries ---
library(mgcv)
library(readr)

# --- 1. Get Command Line Arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Four arguments required: <model_rds_path> <new_data_csv_path> <output_preds_csv_path> <identifier_col_name>", call. = FALSE)
}
model_path <- args[1]
new_data_path <- args[2]
output_path <- args[3]
id_col <- args[4]

cat("R SCRIPT (GAM Predict): Starting.\n")

# --- 2. Load Model and New Data ---
cat("R SCRIPT: Loading pre-trained model from:", model_path, "\n")
final_model <- readRDS(model_path)

cat("R SCRIPT: Loading new data from:", new_data_path, "\n")
new_data_df <- readr::read_csv(new_data_path, show_col_types = FALSE)

# --- 3. Generate Predictions ---
cat("R SCRIPT: Generating predictions on new data...\n")
predictions <- predict(final_model, newdata = new_data_df, type = "response")

# --- 4. Save Predictions ---
# Using a specific column name to be read by the Python script
output_df <- data.frame(
  ID = new_data_df[[id_col]],
  ggpgamsvc_pred = predictions # MODIFIED: Changed column name for consistency
)
# Rename the ID column to match the one from Python
names(output_df)[1] <- id_col

write.csv(output_df, output_path, row.names = FALSE)
cat("R SCRIPT: Predictions saved successfully to:", output_path, "\n")
cat("R SCRIPT (GAM Predict): Finished.\n")