# GGPGAM_Final_Analysis.r
#
# PURPOSE:
# 1. Load the full dataset passed from Python.
# 2. Train a ggpgam model.
# 3. Save predictions back to a CSV for Python.
# 4. Create and save diagnostic plots of the GAM component.

# --- Load Libraries ---
library(ggpgam)
library(mgcv)
library(ggplot2)
library(dplyr)

# --- Get Arguments from Python ---
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 6) {
  stop("Insufficient arguments. Required: input_csv, output_csv, plots_dir, id_col, target_col, predictor_cols...", call.=FALSE)
}

input_csv_path <- args[1]
output_csv_path <- args[2]
plots_dir_path <- args[3]
id_col <- args[4]
target_col <- args[5]
predictor_cols <- args[6:length(args)]

# --- Load and Prepare Data ---
message(paste("Loading data from:", input_csv_path))
full_data <- read.csv(input_csv_path)

# Ensure correct data types
full_data[[id_col]] <- as.character(full_data[[id_col]])
full_data[[target_col]] <- as.numeric(full_data[[target_col]])
for (col in predictor_cols) {
  full_data[[col]] <- as.numeric(full_data[[col]])
}

# --- Define the Model Formula ---
# Create the GAM smooth term formula part
# Example: s(projected_X, projected_Y) + s(Pred1) + s(Pred2) + ...
gam_smooth_terms <- paste0("s(", predictor_cols, ")", collapse = " + ")
spatial_smooth_term <- "s(projected_X, projected_Y)"
full_formula_str <- paste(target_col, "~", spatial_smooth_term, "+", gam_smooth_terms)
model_formula <- as.formula(full_formula_str)

message(paste("Using formula:", deparse(model_formula)))

# --- Train the GGP-GAM Model ---
message("Training the final GGP-GAM model on the full dataset...")
# Note: You may need to adjust st.nu, st.b, etc. based on your data/knowledge
final_model <- ggpgam(
  formula = model_formula,
  data = full_data,
  st.nu = 2,
  st.b = 10
)

# --- Generate and Save Predictions ---
message("Generating predictions on the full dataset...")
predictions <- predict(final_model)
pred_df <- data.frame(
  ID = full_data[[id_col]],
  ggpgamsvc_pred = predictions
)
# Rename ID column to match the identifier from Python
names(pred_df)[1] <- id_col

write.csv(pred_df, output_csv_path, row.names = FALSE)
message(paste("Predictions saved to:", output_csv_path))


# --- Generate and Save GAM Interaction Plots ---
message("Generating and saving GAM interaction plots...")
# The GAM component is stored in final_model$gam
gam_component <- final_model$gam

# Use plot.gam or vis.gam from the mgcv package to create plots
# Loop through each predictor to create an individual plot
for (pred in predictor_cols) {
  plot_path <- file.path(plots_dir_path, paste0("gam_effect_", pred, ".png"))
  png(plot_path, width = 800, height = 600)
  
  # plot.gam is simple and effective
  plot(gam_component, select = which(grepl(pred, names(coef(gam_component)))),
       shade = TRUE, seWithMean = TRUE, scale = 0, main = paste("Partial Effect of", pred))
  
  dev.off()
}
message(paste("GAM plots saved in:", plots_dir_path))

message("R script execution complete.")