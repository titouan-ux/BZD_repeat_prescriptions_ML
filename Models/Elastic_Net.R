## ELASTIC NET FOR FEMALES

library(glmnet)
library(dplyr)
library(data.table)
library(pROC)
library(ggplot2)


X_train_fem <- readRDS("Datasets/X_train_fem_nobio.rds")
X_test_fem  <- readRDS("Datasets/X_test_fem_nobio.rds")
Y_train_fem <- readRDS("Datasets/Y_train_fem.rds")
Y_test_fem  <- readRDS("Datasets/Y_test_fem.rds")

#prdictors to matrix, outcome to characters
X_train_fem <- as.matrix(X_train_fem)
X_test_fem  <- as.matrix(X_test_fem)
Y_train_fem <- as.numeric(as.character(Y_train_fem))
Y_test_fem  <- as.numeric(as.character(Y_test_fem))


# CV error for a given alpha
cvm_enet <- function(alpha) {
  cv_model <- cv.glmnet(x = X_train_fem, y = Y_train_fem, 
                        alpha = alpha, family = "binomial", nfolds = 10)
  cv_error <- cv_model$cvm[which.min(abs(cv_model$lambda - cv_model$lambda.1se))]
  return(cv_error)
}

# optimise alpha
alpha_opt <- optimise(cvm_enet, c(0, 1))
optimal_alpha <- alpha_opt$minimum
cat("Optimal alpha:", optimal_alpha, "\n")

# elastic net
enet_fem_cv <- cv.glmnet(x = X_train_fem, y = Y_train_fem, 
                         alpha = optimal_alpha, family = "binomial", nfolds = 10)
plot(enet_fem_cv, main = "Elastic Net CV Curve (Females)")

# Extract the lambda value (lambda.1se)
lambda_val <- enet_fem_cv$lambda.1se
cat("Lambda.1se:", lambda_val, "\n")

# extract coeff
coef_enet <- coef(enet_fem_cv, s = "lambda.1se")
betas <- coef_enet[-1]  # Remove the intercept for plotting
names(betas) <- rownames(coef_enet)[-1]

#table of non-zero coefficients
non_zero_coefs <- betas[betas != 0]
coef_table_enef <- data.frame(Variable = names(non_zero_coefs), 
                         Coefficient = as.numeric(non_zero_coefs))
# Stability selection for the female dataset

t0 <- Sys.time()
female_stability_final <- VariableSelection(xdata = X_train_fem, ydata = Y_train_fem, 
                                            family = "binomial", n_cat = 3)
t1 <- Sys.time()
cat("Stability selection runtime (Females):", t1 - t0, "\n")

# Plot the calibration curve for stability selection
CalibrationPlot(female_stability_final)
# Save current par settings
old_par <- par(no.readonly = TRUE)

# Increase margins: c(bottom, left, top, right)
par(mar = c(7, 5, 5, 2) + 0.1)  # adjust these numbers as needed

# Call CalibrationPlot() with the new margin settings
CalibrationPlot(female_stability_final)

# Restore original par settings
par(old_par)


# Extract selection proportions for each variable
selprop_female <- SelectionProportions(female_stability_final)
print(selprop_female)

# Obtain the optimal parameters (e.g., threshold) from the stability selection results
hat_params_female <- Argmax(female_stability_final)
print(hat_params_female)

# Identify stable variables, e.g., those with a selection proportion >= 0.97
stable_vars_female <- selprop_female[selprop_female >= 0.97]
print(stable_vars_female)

# Plot the selection proportions
par(mar = c(10, 5, 1, 1))
plot(selprop_female, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", 
     col = ifelse(selprop_female >= hat_params_female[2], "red", "grey"),
     cex.lab = 1.5)
abline(h = hat_params_female[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_female)) {
  axis(side = 1, at = i, labels = names(selprop_female)[i],
       las = 2, 
       col = ifelse(selprop_female[i] >= hat_params_female[2], "red", "grey"), 
       col.axis = ifelse(selprop_female[i] >= hat_params_female[2], "red", "grey"))
}


# Create a mapping of long variable names to short labels
rename_vars <- c(
  "owner_or_rent.0.0_Rent\n" = "Rent",
  "owner_or_rent.0.0_Others" = "Other Housing",
  "vigorous_phys_act.0.0_7" = "Vigorous Phys. Act",
  "weekly_usage_phone.0.0_30-59 mins" = "Phone Use (30-59m)",
  "morning_evening person.0.0_morning person" = "Morning Person",
  "morning_evening person.0.0_evening person" = "Evening Person",
  "Sleeplessness / insomnia.0.0_Never/rarely" = "Rare Insomnia",
  "Sleeplessness / insomnia.0.0_yes" = "Insomnia",
  "current_smok.0.0_No" = "Non-Smoker",
  "current_smok.0.0_current smoker" = "Current Smoker",
  "water_intake.0.0" = "Water Intake",
  "Alcohol intake frequency.0.0_Never" = "No Alcohol",
  "mood swing.0.0_No" = "No Mood Swings",
  "mood swing.0.0_Yes" = "Mood Swings",
  "Sensitivity / hurt feelings.0.0_No" = "No Sensitivity",
  "Sensitivity / hurt feelings.0.0_Yes" = "Sensitive",
  "Nervous feelings.0.0_No" = "No Nervousness",
  "Nervous feelings.0.0_Yes" = "Nervous",
  "Worrier / anxious feelings.0.0_No" = "No Anxiety",
  "Worrier / anxious feelings.0.0_Yes" = "Anxious",
  "Tense / 'highly strung'.0.0_No" = "Not Tense",
  "Tense / 'highly strung'.0.0_Yes" = "Tense",
  "Suffer from 'nerves'.0.0_No" = "No Nerves",
  "Suffer from 'nerves'.0.0_Yes" = "Nervous Issues",
  "Loneliness, isolation.0.0_No" = "Not Lonely",
  "Loneliness, isolation.0.0_Yes" = "Lonely",
  "Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No" = "No GP Anxiety Visit",
  "Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_Yes" = "GP Anxiety Visit",
  "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No" = "No Psychiatrist Visit",
  "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes" = "Psychiatrist Visit",
  "Long-standing illness, disability or infirmity.0.0_No" = "No Chronic Illness",
  "Long-standing illness, disability or infirmity.0.0_Yes" = "Chronic Illness",
  "cancer by doctor.0.0_No" = "No Cancer Diagnosis",
  "cancer by doctor.0.0_Yes" = "Cancer Diagnosis",
  "Current employment status.0.0_Employed" = "Employed",
  "Current employment status.0.0_Unemployed" = "Unemployed",
  "Illnesses of mother.0.0_Heart disease or stroke" = "Mother: Heart Disease",
  "Illnesses of siblings.0.0_cancer" = "Sibling: Cancer",
  "Smoking status.0.0_Never" = "Never Smoked",
  "Smoking status.0.0_Current" = "Current Smoker",
  "Alcohol drinker status.0.0_Previous" = "Ex-Drinker",
  "Alcohol drinker status.0.0_Current" = "Current Drinker",
  "Pack years of smoking.0.0" = "Smoking Pack Years",
  "Ethnic background.0.0_Irish" = "Irish Ethnicity",
  "Body mass index (BMI).0.0" = "BMI",
  "Nitrogen dioxide air pollution; 2006.0.0" = "NO2 Pollution",
  "Natural environment percentage, buffer 1000m.0.0" = "Green Space (1000m)",
  "income_score" = "Income Score",
  "health_score" = "Health Score",
  "housing_score" = "Housing Score",
  "Overall_health_rating_Poor" = "Poor Health",
  "Overall_health_rating_Fair" = "Fair Health",
  "Overall_health_rating_Excellent" = "Excellent Health"
)

# Replace long variable names with short names
coef_table_enef$Short_Variable <- rename_vars[coef_table_enef$Variable]

# If any variable is missing in the mapping, keep its original name
coef_table_enef$Short_Variable[is.na(coef_table_enef$Short_Variable)] <- coef_table_enef$Variable

# plotting non zero coeff
par(mar = c(14, 4, 2, 2))  # Adjust bottom margin for labels
plot(non_zero_coefs, type = "h", col = "navy", lwd = 3,
     xaxt = "n", xlab = "", ylab = expression(beta),
     main = "Nonzero Coefficients (Female Elastic Net)")
axis(side = 1, at = 1:length(non_zero_coefs), labels = coef_table_enef$Short_Variable,
     las = 2, cex.axis = 0.4)  # Reduce label size slightly for readability
abline(h = 0, lty = 2)


# prediction
enet_pred_prob <- predict(enet_fem_cv, newx = X_test_fem, 
                          s = "lambda.1se", type = "response")

# predicted probabilities to class
enet_pred_class <- as.factor(ifelse(enet_pred_prob > 0.5, 1, 0))
Y_test_fem_factor <- as.factor(Y_test_fem)

# confusion matrix (I used base R because I couldn't download caret package to r on demand :()
conf_matrix_enet <- table(Predicted = enet_pred_class, Actual = Y_test_fem_factor)
print(conf_matrix_enet)

#AUC and  ROC curve
roc_curve_enet <- roc(as.numeric(Y_test_fem_factor), as.numeric(enet_pred_prob))
auc_value_enet <- auc(roc_curve_enet)
cat("AUC:", auc_value_enet, "\n")

plot(roc_curve_enet, col = "blue", main = "AUC-ROC Curve for Elastic Net (Females)")
legend("bottomright", legend = paste("AUC =", round(auc_value_enet, 3)), col = "blue", lwd = 2)

fpr <- 1 - roc_curve_enet$specificities  # False Positive Rate
tpr <- roc_curve_enet$sensitivities      # True Positive Rate

# Save as CSV
roc_data <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value_enet, length(tpr)))
write.csv(roc_data, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_enet_fem.txt", 
          row.names = FALSE)


# For the female elastic net model:
# Create a dataframe with variable names and their selection proportions
stable_vars_female_df <- data.frame(
  Variable = names(stable_vars_female),
  Selection_Proportion = stable_vars_female
)

# Save the stably selected variables as an RDS file
saveRDS(stable_vars_female, "stable_vars_elnet_female.rds")



#ELASTIC NET FOR MALES


X_train_male <- readRDS("Datasets/X_train_male_nobio.rds")
X_test_male  <- readRDS("Datasets/X_test_male_nobio.rds")
Y_train_male <- readRDS("Datasets/Y_train_male.rds")
Y_test_male  <- readRDS("Datasets/Y_test_male.rds")

# Convert predictors to matrix, outcome to numeric
X_train_male <- as.matrix(X_train_male)
X_test_male  <- as.matrix(X_test_male)
Y_train_male <- as.numeric(as.character(Y_train_male))
Y_test_male  <- as.numeric(as.character(Y_test_male))

# CV error function for a given alpha
cvm_enet_male <- function(alpha) {
  cv_model <- cv.glmnet(x = X_train_male, y = Y_train_male, 
                        alpha = alpha, family = "binomial", nfolds = 10)
  cv_error <- cv_model$cvm[which.min(abs(cv_model$lambda - cv_model$lambda.1se))]
  return(cv_error)
}

# Optimize alpha
alpha_opt_male <- optimise(cvm_enet_male, c(0, 1))
optimal_alpha_male <- alpha_opt_male$minimum
cat("Optimal alpha (Males):", optimal_alpha_male, "\n")

# Elastic Net Model for Males
enet_male_cv <- cv.glmnet(x = X_train_male, y = Y_train_male, 
                          alpha = optimal_alpha_male, family = "binomial", nfolds = 10)
plot(enet_male_cv, main = "Elastic Net CV Curve (Males)")

# Extract the lambda value (lambda.1se)
lambda_male_val <- enet_male_cv$lambda.1se
cat("Lambda.1se:", lambda_male_val, "\n")

# Extract coefficients
coef_enet_male <- coef(enet_male_cv, s = "lambda.1se")
betas_male <- coef_enet_male[-1]  # Remove the intercept for plotting
names(betas_male) <- rownames(coef_enet_male)[-1]

# Table of non-zero coefficients
non_zero_coefs_male <- betas_male[betas_male != 0]
coef_table_male <- data.frame(Variable = names(non_zero_coefs_male), 
                              Coefficient = as.numeric(non_zero_coefs_male))

# Stability selection for the male dataset

t0 <- Sys.time()
male_stability_final <- VariableSelection(xdata = X_train_male, ydata = Y_train_male, 
                                          family = "binomial", n_cat = 3)
t1 <- Sys.time()
print(t1 - t0)

# Plot the calibration curve for stability selection
CalibrationPlot(male_stability_final)
# Save current par settings
old_par_male <- par(no.readonly = TRUE)

# Increase margins: c(bottom, left, top, right)
par(mar = c(7, 5, 5, 2) + 0.1)  # adjust these numbers as needed

# Call CalibrationPlot() with the new margin settings
CalibrationPlot(male_stability_final)

# Plot calibration curve for stability selection
CalibrationPlot(male_stability_final)

# Get the selection proportions for each variable
selprop_male <- SelectionProportions(male_stability_final)
print(selprop_male)

# Get the optimal (threshold) parameters from the stability selection results
hat_params_male <- Argmax(male_stability_final)
print(hat_params_male)

# Identify the stable variables: here, those with selection proportions above 0.88
stable_vars_male <- selprop_male[selprop_male >= 0.99] 
print(stable_vars_male)

# Plot the selection proportions with a customized axis
par(mar = c(10, 5, 1, 1))
plot(selprop_male, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", 
     col = ifelse(selprop_male >= hat_params_male[2], "red", "grey"),
     cex.lab = 1.5)
abline(h = hat_params_male[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_male)) {
  axis(side = 1, at = i, labels = names(selprop_male)[i],
       las = 2, 
       col = ifelse(selprop_male[i] >= hat_params_male[2], "red", "grey"), 
       col.axis = ifelse(selprop_male[i] >= hat_params_male[2], "red", "grey"))
}

# Create a mapping of long variable names to short labels
rename_vars_male <- c(
  "owner_or_rent.0.0_Rent\n" = "Rent",
  "moderate_phys_act.0.0_1" = "Moderate Phys. Act (1)",
  "moderate_phys_act.0.0_2" = "Moderate Phys. Act (2)",
  "moderate_phys_act.0.0_5" = "Moderate Phys. Act (5)",
  "vigorous_phys_act.0.0_4" = "Vigorous Phys. Act (4)",
  "vigorous_phys_act.0.0_5" = "Vigorous Phys. Act (5)",
  "vigorous_phys_act.0.0_6" = "Vigorous Phys. Act (6)",
  "vigorous_phys_act.0.0_7" = "Vigorous Phys. Act (7)",
  "friend_family visits.0.0_monthly" = "Family Visits (Monthly)",
  "time_watch computer.0.0" = "Screen Time",
  "Sleeplessness / insomnia.0.0_Never/rarely" = "Rare Insomnia",
  "Sleeplessness / insomnia.0.0_yes" = "Insomnia",
  "current_smok.0.0_No" = "Non-Smoker",
  "current_smok.0.0_current smoker" = "Current Smoker",
  "smokers_in_household.0.0" = "Smokers in Household",
  "water_intake.0.0" = "Water Intake",
  "Alcohol intake frequency.0.0_Never" = "No Alcohol",
  "mood swing.0.0_No" = "No Mood Swings",
  "mood swing.0.0_Yes" = "Mood Swings",
  "Irritability.0.0_No" = "Not Irritable",
  "Irritability.0.0_Yes" = "Irritable",
  "Worrier / anxious feelings.0.0_No" = "No Anxiety",
  "Worrier / anxious feelings.0.0_Yes" = "Anxious",
  "Tense / 'highly strung'.0.0_No" = "Not Tense",
  "Tense / 'highly strung'.0.0_Yes" = "Tense",
  "Suffer from 'nerves'.0.0_No" = "No Nerves",
  "Suffer from 'nerves'.0.0_Yes" = "Nervous Issues",
  "Loneliness, isolation.0.0_No" = "Not Lonely",
  "Loneliness, isolation.0.0_Yes" = "Lonely",
  "Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No" = "No GP Anxiety Visit",
  "Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_Yes" = "GP Anxiety Visit",
  "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No" = "No Psychiatrist Visit",
  "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes" = "Psychiatrist Visit",
  "Long-standing illness, disability or infirmity.0.0_No" = "No Chronic Illness",
  "Long-standing illness, disability or infirmity.0.0_Yes" = "Chronic Illness",
  "cancer by doctor.0.0_No" = "No Cancer Diagnosis",
  "cancer by doctor.0.0_Yes" = "Cancer Diagnosis",
  "Current employment status.0.0_Employed" = "Employed",
  "Current employment status.0.0_Unemployed" = "Unemployed",
  "Illnesses of siblings.0.0_Heart disease or stroke" = "Sibling: Heart Disease",
  "Smoking status.0.0_Current" = "Current Smoker",
  "Alcohol drinker status.0.0_Previous" = "Ex-Drinker",
  "Alcohol drinker status.0.0_Current" = "Current Drinker",
  "Pack years of smoking.0.0" = "Smoking Pack Years",
  "Nitrogen dioxide air pollution; 2010.0.0" = "NO2 Pollution (2010)",
  "Nitrogen dioxide air pollution; 2006.0.0" = "NO2 Pollution (2006)",
  "Particulate matter air pollution (pm10); 2007.0.0" = "PM10 Pollution (2007)",
  "Natural environment percentage, buffer 300m.0.0" = "Green Space (300m)",
  "Distance (Euclidean) to coast.0.0" = "Distance to Coast",
  "health_score" = "Health Score",
  "housing_score" = "Housing Score",
  "Overall_health_rating_Poor" = "Poor Health",
  "Overall_health_rating_Fair" = "Fair Health",
  "Overall_health_rating_Excellent" = "Excellent Health"
)

# Apply renaming
coef_table_male$Short_Variable <- rename_vars_male[coef_table_male$Variable]
coef_table_male$Short_Variable[is.na(coef_table_male$Short_Variable)] <- coef_table_male$Variable  # Keep original name if missing


# plotting non zero coeff
par(mar = c(14, 4, 2, 2) + 0.1)  # Adjust bottom margin for labels
plot(non_zero_coefs_male, type = "h", col = "red", lwd = 3,
     xaxt = "n", xlab = "", ylab = expression(beta),
     main = "Nonzero Coefficients (Elastic Net - Males)")
axis(side = 1, at = 1:length(non_zero_coefs_male), labels = coef_table_male$Short_Variable,
     las = 2, cex.axis = 0.5)  # Reduce label size slightly for readability
abline(h = 0, lty = 2)


# Prediction
enet_pred_prob_male <- predict(enet_male_cv, newx = X_test_male, 
                               s = "lambda.1se", type = "response")

# Convert predicted probabilities to class labels
enet_pred_class_male <- as.factor(ifelse(enet_pred_prob_male > 0.5, 1, 0))
Y_test_male_factor <- as.factor(Y_test_male)

# Confusion matrix
conf_matrix_enet_male <- table(Predicted = enet_pred_class_male, Actual = Y_test_male_factor)
print(conf_matrix_enet_male)

# AUC and ROC curve
roc_curve_enet_male <- roc(as.numeric(Y_test_male_factor), as.numeric(enet_pred_prob_male))
auc_value_enet_male <- auc(roc_curve_enet_male)
cat("AUC (Males):", auc_value_enet_male, "\n")

plot(roc_curve_enet_male, col = "red", main = "AUC-ROC Curve for Elastic Net (Males)")
legend("bottomright", legend = paste("AUC =", round(auc_value_enet_male, 3)), col = "red", lwd = 2)

fpr <- 1 - roc_curve_enet_male$specificities  # False Positive Rate
tpr <- roc_curve_enet_male$sensitivities      # True Positive Rate

# Save as CSV
roc_data <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value_enet_male, length(tpr)))
write.csv(roc_data, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_enet_male.txt", 
          row.names = FALSE)

# Create a dataframe with variable names and their selection proportions
stable_vars_male_df <- data.frame(
  Variable = names(stable_vars_male),
  Selection_Proportion = stable_vars_male
)

# Save the stably selected variables as an RDS file
saveRDS(stable_vars_male, "Datasets/stable_vars_elnet_male.rds")

