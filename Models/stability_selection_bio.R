rm(list=ls())
project_path=dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(project_path)

library(glmnet)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(data.table)
library(mltools)
library(caret)
library(pROC)
library(fake)
library(igraph)
library(pheatmap)
library(sharp)

ukb_fem_bio <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/NEW_ukb_match_fem_final_bio.rds")
ukb_male_bio <- readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/NEW_ukb_match_male_final_bio.rds")




# load data
# Female
X_train_fem <- readRDS("Datasets/X_train_fem_scaled_bio.rds")
X_test_fem <- readRDS("Datasets/X_test_fem_scaled_bio.rds")
Y_train_fem <- readRDS("Datasets/Y_train_fem_bio.rds")
Y_test_fem <- readRDS("Datasets/Y_test_fem_bio.rds")

# Males
X_train_male <- readRDS("Datasets/X_train_male_scaled_bio.rds")
X_test_male <- readRDS("Datasets/X_test_male_scaled_bio.rds")
Y_train_male <- readRDS("Datasets/Y_train_male_bio.rds")
Y_test_male <- readRDS("Datasets/Y_test_male_bio.rds")

# convert data types 

X_train_fem <- as.matrix(X_train_fem)
X_test_fem <- as.matrix(X_test_fem)
Y_train_fem <-  as.numeric(as.character(Y_train_fem))
Y_test_fem <-  as.numeric(as.character(Y_test_fem))

X_train_male <- as.matrix(X_train_male)
X_test_male <- as.matrix(X_test_male)
Y_train_male <- as.numeric(as.character(Y_train_male))
Y_test_male <- as.numeric(as.character(Y_test_male))

set.seed(5225678)

## stability selection Female --------------------------
t0 <- Sys.time()
fem_stability_final <- VariableSelection(xdata = X_train_fem, ydata = Y_train_fem, family = "binomial", n_cat = 3)
t1 <- Sys.time()
print(t1 - t0)

CalibrationPlot(fem_stability_final)

# Calibrated selection proportions
selprop_fem <- SelectionProportions(fem_stability_final)
print(selprop_fem)

# Calibrated parameters
hat_params_fem <- Argmax(fem_stability_final)
print(hat_params_fem)

stable_vars_fem <- selprop_fem[selprop_fem >= hat_params_fem[2]] 
print(sort(stable_vars_fem))

# Visualisation of selection proportions
par(mar = c(10, 5, 1, 1))
plot(selprop_fem, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_fem >= hat_params_fem[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_fem[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_fem)) {
  axis(side = 1, at = i, labels = names(selprop_fem)[i],
       las = 2, col = ifelse(selprop_fem[i] >= hat_params_fem[2],yes = "red", no = "grey"), col.axis = ifelse(selprop_fem[i] >= hat_params_fem[2], yes = "red", no = "grey"))
}

## saving the stable selected variables
# stable_vars_fem_table <- data.frame(stable_vars_fem)
# stable_vars_fem_table <- rownames_to_column(stable_vars_fem_table)
# stable_vars_fem_table <- stable_vars_fem_table[order(stable_vars_fem_table$stable_vars_fem, decreasing = TRUE), ]
# saveRDS(stable_vars_fem_table, "Datasets/stability.selection_bio_fem.rds")


## lasso Female ---------------------------------------

lasso_fem_cv <- cv.glmnet(x = X_train_fem, y = Y_train_fem, alpha = 1, family = "binomial")
plot(lasso_fem_cv)

lasso_fem_train <- glmnet(X_train_fem, Y_train_fem, alpha = 1, lambda = lasso_fem_cv$lambda.min, family = "binomial")

#extract and plot non 0 coefs
table(coef(lasso_fem_train, s = "lasso_fem_train$lambda.min")[-1] != 0)

betas <- coef(lasso_fem_train, s = "lasso_fem_train$lambda.min")[-1]
names(betas) = rownames(coef(lasso_fem_train, s = "lambda.min"))[-1]

#print(coef_table)

par(mar = c(16, 4, 2, 2) + 0.1) 
plot(betas[betas != 0], type = "h", col = "navy", lwd = 3,
     xaxt = "n", xlab = "", ylab = expression(beta))
axis(side = 1, at = 1:sum(betas != 0), labels = names(betas)[betas !=
                                                               0], las = 2, cex.axis = 0.5)
abline(h = 0, lty = 2)

# table of non 0 coefs
non_zero_indices <- which(betas != 0)
variable_names <- names(betas)[non_zero_indices]
non_zero_coefficients <- betas[non_zero_indices]
coef_table_fem <- data.frame(Variable = variable_names, Coefficient = non_zero_coefficients)
saveRDS(coef_table_fem, "Datasets/variable.selection_lasso_fem_bio.rds")

# prediction
lasso_pred_fem <- predict(lasso_fem_train, s = lasso_fem_cv$lambda.min, newx = X_test_fem, type = "response")

# Confusion Matrix
lasso_pred_class <- as.factor(ifelse(lasso_pred_fem > 0.5, 1, 0))
Y_test_fem <- as.factor(Y_test_fem)

conf_matrix <- confusionMatrix(as.factor(lasso_pred_class), Y_test_fem)
print(conf_matrix)

# -----
# Y_test_fem <- factor(Y_test_fem, levels = c("0", "1"))
# lasso_pred_class <- factor(lasso_pred_class, levels = c("0", "1"))
# 
# # Now call confusionMatrix
# conf_matrix <- confusionMatrix(data = lasso_pred_class, reference = Y_test_fem)
# print(conf_matrix)

# ------
#AUC plot 
roc_curve_fem <- roc(as.numeric(Y_test_fem), as.numeric(lasso_pred_fem))
auc_value_fem <- as.numeric(auc(roc_curve_fem))
fpr <- 1 - roc_curve_fem$specificities  # False Positive Rate
tpr <- roc_curve_fem$sensitivities      # True Positive Rate


# Save as CSV
roc_data_fem <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value_fem, length(tpr)))
write.csv(roc_data_fem, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_fem_bio.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value_fem))

plot(roc_curve_fem, col = "blue", main = "AUC-ROC Curve (female)")
legend("bottomright", legend = paste("AUC =", round(auc_value_fem, 3)), col = "blue", lwd = 2)
























## stability selection Male --------------------------------------------

t0 <- Sys.time()
male_stability_final <- VariableSelection(xdata = X_train_male, ydata = Y_train_male, family = "binomial", n_cat = 3)
t1 <- Sys.time()
print(t1 - t0)

CalibrationPlot(male_stability_final)

# Calibrated selection proportions
selprop_male <- SelectionProportions(male_stability_final)
print(selprop_male)

# Calibrated parameters
hat_params_male <- Argmax(male_stability_final)
print(hat_params_male)

stable_vars_male <- selprop_male[selprop_male >= hat_params_male[2]] 
print(sort(stable_vars_male))

# Visualisation of selection proportions
par(mar = c(10, 5, 1, 1))
plot(selprop_male, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_male >= hat_params_male[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_male[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_male)) {
  axis(side = 1, at = i, labels = names(selprop_male)[i],
       las = 2, col = ifelse(selprop_male[i] >= hat_params_male[2],yes = "red", no = "grey"), col.axis = ifelse(selprop_male[i] >= hat_params_male[2], yes = "red", no = "grey"))
}


stable_vars_male_table <- data.frame(stable_vars_male)
stable_vars_male_table <- rownames_to_column(stable_vars_male_table)
stable_vars_male_table <- stable_vars_male_table[order(stable_vars_male_table$stable_vars_male, decreasing = TRUE), ]
saveRDS(stable_vars_male_table, "Datasets/stability.selection_bio_male.rds")





# Lasso Male -----------------------------------------------------

lasso_male_cv <- cv.glmnet(x = X_train_male, y = Y_train_male, alpha = 1, family = "binomial")
plot(lasso_male_cv)

lasso_male_train <- glmnet(X_train_male, Y_train_male, alpha = 1, lambda = lasso_male_cv$lambda.min, family = "binomial")

# Extract and plot nonzero coefficients
table(coef(lasso_male_train, s = "lasso_male_train$lambda.min")[-1] != 0)

betas <- coef(lasso_male_train, s = "lasso_male_train$lambda.min")[-1]
names(betas) = rownames(coef(lasso_male_train, s = "lambda.min"))[-1]

#print(coef_table)

par(mar = c(16, 4, 2, 2) + 0.1) 
plot(betas[betas != 0], type = "h", col = "red", lwd = 3,
     xaxt = "n", xlab = "", ylab = expression(beta))
axis(side = 1, at = 1:sum(betas != 0), labels = names(betas)[betas !=
                                                               0], las = 2, cex.axis = 0.5)
abline(h = 0, lty = 2)

# Table of nonzero coefficients
non_zero_indices <- which(betas != 0)
variable_names <- names(betas)[non_zero_indices]
non_zero_coefficients <- betas[non_zero_indices]
coef_table_male <- data.frame(Variable = variable_names, Coefficient = non_zero_coefficients)
saveRDS(coef_table_male, "Datasets/variable.selection_lasso_male_bio.rds")


# Prediction
lasso_pred_male <- predict(lasso_male_train, s = lasso_male_cv$lambda.min, newx = X_test_male, type = "response")

# Confusion Matrix
lasso_pred_class <- as.factor(ifelse(lasso_pred_male > 0.5, 1, 0))
Y_test_male <- as.factor(Y_test_male)

conf_matrix <- confusionMatrix(as.factor(lasso_pred_class), Y_test_male)
print(conf_matrix)


# AUC plot 
roc_curve_male <- roc(as.numeric(Y_test_male), as.numeric(lasso_pred_male))
auc_value_male <- as.numeric(auc(roc_curve_male))
fpr <- 1 - roc_curve_male$specificities  # False Positive Rate
tpr <- roc_curve_male$sensitivities      # True Positive Rate

# Save as CSV
roc_data_male <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value_male, length(tpr)))
write.csv(roc_data_male, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_male_bio.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value_male))

plot(roc_curve_male, col = "red", main = "AUC-ROC Curve (male)")
legend("bottomright", legend = paste("AUC =", round(auc_value_male, 3)), col = "red", lwd = 2)




