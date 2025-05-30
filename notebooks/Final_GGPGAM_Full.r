# GGPGAM_Full.r (Revised)
#
# PURPOSE:
# 1. Train the final GAM (SVC) model on the ENTIRE dataset using best practices.
# 2. Save the final model object for future use.
# 3. Generate and save in-sample predictions for the Python meta-learner.
# 4. Generate and save diagnostic plots for each predictor's partial effect.

# --- 0. Load Libraries ---
library(mgcv)
library(readr)
library(dplyr)

# --- 1. Get Command Line Arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Four arguments required: <full_data_csv_path> <identifier_col_name> <output_dir_path> <predictions_csv_path>", call. = FALSE)
}
full_data_csv_path <- args[1]
identifier_col_name <- args[2]
output_dir_path <- args[3]
predictions_csv_path <- args[4]

cat("R SCRIPT (Final GAM-SVC Model): Starting.\n")

# Create output directory if it doesn't exist
if (!dir.exists(output_dir_path)) {
  dir.create(output_dir_path, recursive = TRUE)
}
plots_dir_path <- file.path(output_dir_path, "gam_interaction_plots")
if (!dir.exists(plots_dir_path)) {
  dir.create(plots_dir_path, recursive = TRUE)
}


# --- 2. Load Full Data ---
full_data_df <- readr::read_csv(full_data_csv_path, show_col_types = FALSE)
cat("R SCRIPT: Full data loaded. Rows:", nrow(full_data_df), "Cols:", ncol(full_data_df), "\n")

# --- 3. Prepare Data ---
target_variable_name <- "AAT_z"
base_predictor_names <- c("Distance_from_water_m", "Pct_Impervious", "Pct_Canopy", "Pct_GreenSpace", "Elevation_m")
base_predictor_names <- base_predictor_names[base_predictor_names %in% names(full_data_df)]
if(length(base_predictor_names) == 0) { stop("R SCRIPT: No valid base predictors found.")}
coord_x_name <- "projected_X"; coord_y_name <- "projected_Y"

# --- 4. Construct GAM Formula ---
n_obs <- nrow(full_data_df)
# SUGGESTION 4: Use dynamic k based on the full dataset size
k_main_spatial <- max(150, floor(n_obs / 300))
k_svc_interaction <- max(40, floor(n_obs / 1000))

linear_terms <- paste(base_predictor_names, collapse = " + ")
# SUGGESTION 1: Use bs="tp" for consistency with your validation script
main_spatial_term <- paste0("s(", coord_x_name, ", ", coord_y_name, ", bs=\"tp\", k=", k_main_spatial, ")")
svc_terms <- sapply(base_predictor_names, function(pred_name) {
  paste0("s(", coord_x_name, ", ", coord_y_name, ", by=", pred_name, ", bs=\"tp\", k=", k_svc_interaction, ")")
})
svc_terms_str <- paste(svc_terms, collapse = " + ")
formula_str <- paste(target_variable_name, "~", linear_terms, "+", main_spatial_term, "+", svc_terms_str)
gam_formula <- as.formula(formula_str)

cat("R SCRIPT: Using final GAM (SVC) formula:", formula_str, "\n")
cat("R SCRIPT: Fitting mgcv::bam model on full data...\n")

# --- 5. Fit Final Model ---
final_model <- NULL
tryCatch({
  # SUGGESTION 2 & 3: Use bam() for speed and add select=TRUE for regularization
  final_model <- mgcv::bam(
    formula = gam_formula,
    data = full_data_df,
    method = "fREML", # Use fREML for bam
    select = TRUE
  )
  cat("R SCRIPT: Final model fitting complete.\n")
}, error = function(e) {
  cat("R SCRIPT: Error during final model fitting: ", conditionMessage(e), "\n")
})


# --- 6. Save Outputs (Model, Predictions, Plots) ---
if (!is.null(final_model)) {
  # Save the final model object
  saveRDS(final_model, file = file.path(output_dir_path, "gam_svc_final_model.rds"))
  cat("  Saved final model object to gam_svc_final_model.rds\n")
  
  # --- Generate and Save In-Sample Predictions ---
  cat("R SCRIPT: Generating in-sample predictions...\n")
  model_predictions <- predict(final_model, newdata = full_data_df, type = "response")
  
  output_df <- data.frame(identifier_placeholder = full_data_df[[identifier_col_name]], prediction = model_predictions)
  names(output_df)[1] <- identifier_col_name
  names(output_df)[2] <- "ggpgamsvc_pred" # Use a consistent name
  readr::write_csv(output_df, predictions_csv_path)
  cat("R SCRIPT: In-sample predictions saved to:", predictions_csv_path, "\n")
  
  
  # --- SUGGESTION 5: Generate and Save Diagnostic Plots ---
  cat("R SCRIPT: Generating and saving diagnostic plots...\n")
  # This loop creates a plot for each smooth term in the model
  num_smooths <- length(final_model$smooth)
  for (i in 1:num_smooths) {
    term_label <- final_model$smooth[[i]]$label
    plot_path <- file.path(plots_dir_path, paste0("gam_effect_", term_label, ".png"))
    png(plot_path, width = 800, height = 600)
    
    # The plot() function for gam objects is very powerful
    plot(final_model, select = i, shade = TRUE, seWithMean = TRUE, scale = 0, main = term_label)
    
    dev.off()
  }
  cat("R SCRIPT: Diagnostic plots saved in:", plots_dir_path, "\n")

} else {
  cat("R SCRIPT: Model fitting failed. Skipping prediction and saving.\n")
}

cat("R SCRIPT (Final GAM-SVC Model): Finished.\n")