#rm(list=ls())
#project_path=dirname(rstudioapi::getActiveDocumentContext()$path)

library(glmnet)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(data.table)
library(mltools)
library(missForestPredict)
library(caret)
library(pROC)
library(fake)
library(igraph)
library(pheatmap)
library(sharp)
library(broom)

#load datasets 

ukb_fem <- readRDS("Datasets/NEW_ukb_match_fem_final.rds")
ukb_fem <- as.data.frame(ukb_fem)
ukb_male <- readRDS("Datasets/NEW_ukb_match_male_final.rds")
ukb_male <- as.data.frame(ukb_male)

ukb_fem[sapply(ukb_fem, is.character)] <- 
  lapply(ukb_fem[sapply(ukb_fem, is.character)], as.factor)

ukb_fem[sapply(ukb_fem, is.integer)] <- 
  lapply(ukb_fem[sapply(ukb_fem, is.character)], as.numeric)

ukb_male[sapply(ukb_male, is.character)] <- 
  lapply(ukb_male[sapply(ukb_male, is.character)], as.factor)

ukb_male[sapply(ukb_male, is.integer)] <- 
  lapply(ukb_male[sapply(ukb_male, is.character)], as.numeric)

#### FEMALE #### test train split based on matching 

set.seed(5225678) 

unique_clusters <- unique(ukb_fem$cluster_case)

train_clusters <- sample(unique_clusters, size = 0.8 * length(unique_clusters))

train_fem <- which(ukb_fem$cluster_case %in% train_clusters)
test_fem  <- which(!ukb_fem$cluster_case %in% train_clusters)

Y <- as.factor(ukb_fem$case_control)
X <- as.data.frame(ukb_fem[, !colnames(ukb_fem) %in% c("case_control")])

X_train_fem <- X[train_fem, ]
Y_train_fem <- Y[train_fem]

X_test_fem <- X[test_fem, ]
Y_test_fem <- Y[test_fem]

# Check matching cases kept together
table(ukb_fem$cluster_case[train_fem] %in% ukb_fem$cluster_case[test_fem])  

#save

saveRDS(Y_train_fem, "Datasets/Y_train_fem.rds")
saveRDS(Y_test_fem, "Datasets/Y_test_fem.rds")

## FOR MALES ##

set.seed(5225678) 

unique_clusters <- unique(ukb_male$cluster_case)

train_clusters <- sample(unique_clusters, size = 0.8 * length(unique_clusters))

train_male <- which(ukb_male$cluster_case %in% train_clusters)
test_male  <- which(!ukb_male$cluster_case %in% train_clusters)

Y <- as.factor(ukb_male$case_control)
X <- as.data.frame(ukb_male[, !colnames(ukb_male) %in% c("case_control")])

X_train_male <- X[train_male, ]
Y_train_male <- Y[train_male]

X_test_male <- X[test_male, ]
Y_test_male <- Y[test_male]

# Check matching cases kept together
table(ukb_male$cluster_case[train_male] %in% ukb_male$cluster_case[test_male])  

saveRDS(Y_train_male, "Datasets/Y_train_male.rds")
saveRDS(Y_test_male, "Datasets/Y_test_male.rds")

## IMPUTATION ##  

#toy_dataset <- X_train_fem[1:500,!colnames(X_train_fem) %in% c("eid","Genetic sex.0.0", "cluster_case", "Well used for sample run.0.0")] #Remove eid,Genetic sex ad cluster case

#imputed_toy <- missForest(toy_dataset, save_models = TRUE)

#saveRDS(imputed_toy$ximp, "imputed_toy_dataset.rds")

#correct character variables
#toy_dataset[sapply(toy_dataset, is.character)] <- 
  #lapply(toy_dataset[sapply(toy_dataset, is.character)], as.factor)

#toy_dataset[sapply(toy_dataset, is.integer)] <- 
  #lapply(toy_dataset[sapply(toy_dataset, is.character)], as.numeric)

#Variable make the missForest to fail
#bug_var<- c("Well used for sample run.0.0")

#imputed_toy <- missForest(toy_dataset[,!colnames(toy_dataset) %in% bug_var],save_models = T)
#imputed_toy_test <- missForestPredict(imputed_toy, newdata = toy_test)

#REMOVE BUG VARIABLE AND EID/SEX/CASECLUSTER before imputing

remove_vars <- c("eid","Genetic sex.0.0", "cluster_case", "Well used for sample run.0.0", "non_cancer_illness.0.0")
X_train_fem <- X_train_fem[,!colnames(X_train_fem) %in% remove_vars]
X_test_fem <- X_test_fem[,!colnames(X_test_fem) %in% remove_vars]
X_train_male <- X_train_male[,!colnames(X_train_male) %in% remove_vars]
X_test_male <- X_test_male[,!colnames(X_test_male) %in% remove_vars]

#impute training data and extract
imputed_train_male <- missForest(X_train_male, save_models = TRUE)
imputed_train_fem  <- missForest(X_train_fem, save_models = TRUE)

saveRDS(imputed_train_male$ximp, "Datasets/imputed_train_male_X.rds")
saveRDS(imputed_train_fem$ximp, "Datasets/imputed_train_fem_X.rds")

#impute test data

imputed_test_male <- missForestPredict(imputed_train_male, newdata = X_test_male)
imputed_test_fem <- missForestPredict(imputed_train_fem, newdata = X_test_fem)

saveRDS(imputed_test_male, "Datasets/imputed_test_male_X.rds")
saveRDS(imputed_test_fem, "Datasets/imputed_test_fem_X.rds")

## STANDARDISE 

#load data

X_train_fem <- readRDS("Datasets/imputed_train_fem_X_recode.rds")
X_test_fem <- readRDS("Datasets/imputed_test_fem_X_recode.rds")

X_train_male <- readRDS("Datasets/imputed_train_male_X_recode.rds")
X_test_male <- readRDS("Datasets/imputed_test_male_X_recode.rds")

#change the NA in the home_area_population column to urban before one hot encoding 
X_train_fem$home_area_population_density[is.na(X_train_fem$home_area_population_density)] <- "Urban"
X_test_fem$home_area_population_density[is.na(X_test_fem$home_area_population_density)] <- "Urban"

X_train_male$home_area_population_density[is.na(X_train_male$home_area_population_density)] <- "Urban"
X_test_male$home_area_population_density[is.na(X_test_male$home_area_population_density)] <- "Urban"


# male standardise 
train_numeric_vars <- X_train_male %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)
names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_male[, col] <- (X_train_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_male[, col]  <- (X_test_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}


#female standardising
train_numeric_vars <- X_train_fem %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)

names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_fem[, col] <- (X_train_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_fem[, col]  <- (X_test_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}

##ONE HOT ENCODE##

#for female  train 

X_train_fem <- as.data.table(X_train_fem)

factor_vars <- sapply(X_train_fem, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_train_fem <- one_hot(X_train_fem, cols = var)
}

X_train_fem <- as.data.frame(X_train_fem)

# for female test 

X_test_fem <- as.data.table(X_test_fem)

factor_vars <- sapply(X_test_fem, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_test_fem <- one_hot(X_test_fem, cols = var)
}

X_test_fem <- as.data.frame(X_test_fem)

#for male train

X_train_male <- as.data.table(X_train_male)

factor_vars <- sapply(X_train_male, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_train_male <- one_hot(X_train_male, cols = var)
}

X_train_male <- as.data.frame(X_train_male)

#for male test 

X_test_male <- as.data.table(X_test_male)

factor_vars <- sapply(X_test_male, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_test_male <- one_hot(X_test_male, cols = var)
}

X_test_male <- as.data.frame(X_test_male)


# save again 

saveRDS(X_train_fem, "Datasets/X_train_fem_scaled_recoded.rds")
saveRDS(X_test_fem, "Datasets/X_test_fem_scaled_recoded.rds")

saveRDS(X_train_male, "Datasets/X_train_male_scaled_recoded.rds")
saveRDS(X_test_male, "Datasets/X_test_male_scaled_recoded.rds")

#convert back into matrix for GLM net 

#Delete Biomarkers----------------------------------
X_train_fem <- readRDS("Datasets/X_train_fem_scaled_recoded.rds")
X_test_fem  <- readRDS("Datasets/X_test_fem_scaled_recoded.rds")
biomarker <- readRDS("Datasets/Variables_blocks/list_biomarkers_var.rds")


# Extract biomarker column names as a character vector
cols_to_remove <- biomarker$list_biomarkers  # Extracts the column as a vector

# Remove the biomarker columns from the datasets
X_train_fem <- X_train_fem[, !(colnames(X_train_fem) %in% cols_to_remove)]
X_test_fem  <- X_test_fem[, !(colnames(X_test_fem) %in% cols_to_remove)]


# Save the filtered datasets if needed
saveRDS(X_train_fem, "Datasets/X_train_fem_nobio.rds")
saveRDS(X_test_fem, "Datasets/X_test_fem_nobio.rds")

#Males
X_train_male <- readRDS("Datasets/X_train_male_scaled_recoded.rds")
X_test_male <- readRDS("Datasets/X_test_male_scaled_recoded.rds")

X_train_male <- X_train_male[, !(colnames(X_train_male) %in% cols_to_remove)]
X_test_male  <- X_test_male[, !(colnames(X_test_male) %in% cols_to_remove)]

saveRDS(X_train_male, "Datasets/X_train_male_nobio.rds")
saveRDS(X_test_male, "Datasets/X_test_male_nobio.rds")

##LASSO##

#load data

X_train_male <- readRDS("Datasets/X_train_male_nobio.rds")
X_test_male <- readRDS("Datasets/X_test_male_nobio.rds")
Y_train_male <- readRDS("Datasets/Y_train_male.rds")
Y_test_male <- readRDS("Datasets/Y_test_male.rds")

X_train_fem <- readRDS("Datasets/X_train_fem_nobio.rds")
X_test_fem <- readRDS("Datasets/X_test_fem_nobio.rds")
Y_train_fem <- readRDS("Datasets/Y_train_fem.rds")
Y_test_fem <- readRDS("Datasets/Y_test_fem.rds")

#remove trans ppl cos all 1s
remove_this <- c("Transgender_Trans", "Transgender_Non-trans", "Plate used for sample run.0.0")

X_train_fem <- X_train_fem[,!colnames(X_train_fem) %in% remove_this]
X_test_fem <- X_test_fem[,!colnames(X_test_fem) %in% remove_this]

X_train_male <- X_train_male[,!colnames(X_train_male) %in% remove_this]
X_test_male <- X_test_male[,!colnames(X_test_male) %in% remove_this]

#convert data types 

X_train_male <- as.matrix(X_train_male)
X_test_male <- as.matrix(X_test_male)
Y_train_male <- as.numeric(as.character(Y_train_male))
Y_test_male <- as.numeric(as.character(Y_test_male))


X_train_fem <- as.matrix(X_train_fem)
X_test_fem <- as.matrix(X_test_fem)
Y_train_fem <-  as.numeric(as.character(Y_train_fem))
Y_test_fem <-  as.numeric(as.character(Y_test_fem))


#some cleaning, removing cols with 0 and 1

zero_one_columns <- colnames(X_train_fem)[which(apply(X_train_fem, 2, function(x) all(x==0) | all(x==1)))]
X_train_fem <- X_train_fem[,!colnames(X_train_fem) %in% zero_one_columns]
X_test_fem <- X_test_fem[,!colnames(X_test_fem) %in% zero_one_columns]

zero_one_columns_m <- colnames(X_train_male)[which(apply(X_train_male, 2, function(x) all(x==0) | all(x==1)))]
X_train_male <- X_train_male[,!colnames(X_train_male) %in% zero_one_columns_m]
X_test_male <- X_test_male[,!colnames(X_test_male) %in% zero_one_columns_m]

#for females 

lasso_fem_cv <- cv.glmnet(x = X_train_fem, y = Y_train_fem, alpha = 1, family = "binomial")
plot(lasso_fem_cv)

lasso_fem_train <- glmnet(X_train_fem, Y_train_fem, alpha = 1, lambda = lasso_fem_cv$lambda.1se, family = "binomial")

#extract and plot non 0 coefs
table(coef(lasso_fem_train, s = "lasso_fem_train$lambda.1se")[-1] != 0)

betas <- coef(lasso_fem_train, s = "lasso_fem_train$lambda.1se")[-1]
names(betas) = rownames(coef(lasso_fem_train, s = "lambda.1se"))[-1]

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
saveRDS(coef_table_fem, "Datasets/variable.selection_lasso_fem.rds")


#stability selection 

for(i in seq(11,ncol(X_train_fem),100)){
  print(i)
  fem_stability <- VariableSelection(xdata = X_train_fem[,1:i], ydata = Y_train_fem, family = "binomial", n_cat = 3, K=10)
}

CalibrationPlot(fem_stability)

t0 <- Sys.time()
fem_stability_lasso <- VariableSelection(xdata = X_train_fem, ydata = Y_train_fem, family = "binomial", n_cat = 3)
t1 <- Sys.time()
print(t1 - t0)

par(mar = c(7, 5, 3, 7) + 0.1, oma = c(0, 0, 2, 0))
CalibrationPlot(fem_stability_lasso)


selprop_fem_lasso <- SelectionProportions(fem_stability_lasso)
print(selprop_fem_lasso)

hat_params_fem_lasso <- Argmax(fem_stability_lasso)
print(hat_params_fem_lasso)

stable_vars_fem_lasso <- selprop_fem_lasso[selprop_fem_lasso >= 0.97] 
print(stable_vars_fem_lasso)

par(mar = c(10, 5, 1, 1))
plot(selprop_fem_lasso, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_fem_lasso >=
                                                               hat_params_fem_lasso[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_fem[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_fem_lasso)) {
  axis(side = 1, at = i, labels = names(selprop_fem_lasso)[i],
       las = 2, col = ifelse(selprop_fem_lasso[i] >= hat_params_fem_lasso[2],yes = "red", no = "grey"), col.axis = ifelse(selprop_fem_lasso[i] >=
                                                                                                                hat_params_fem_lasso[2], yes = "red", no = "grey"))
}

# prediction
lasso_pred_fem <- predict(lasso_fem_train, s = lasso_fem_cv$lambda.1se, newx = X_test_fem, type = "response")

# Confusion Matrix
lasso_pred_class <- as.factor(ifelse(lasso_pred_fem > 0.5, 1, 0))
Y_test_fem <- as.factor(Y_test_fem)

conf_matrix <- confusionMatrix(as.factor(lasso_pred_class), Y_test_fem)
print(conf_matrix)

#AUC plot 
roc_curve_f_og <- roc(as.numeric(Y_test_fem), as.numeric(lasso_pred_fem))
auc_value_f_og <- as.numeric(auc(roc_curve))
fpr <- 1 - roc_curve$specificities  # False Positive Rate
tpr <- roc_curve$sensitivities      # True Positive Rate


# Save as CSV
roc_data <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value, length(tpr)))
write.csv(roc_data, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_fem.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", main = "AUC-ROC Curve (female)")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)

#MALE lasso

lasso_male_cv <- cv.glmnet(x = X_train_male, y = Y_train_male, alpha = 1, family = "binomial")
plot(lasso_male_cv)

lasso_male_train <- glmnet(X_train_male, Y_train_male, alpha = 1, lambda = lasso_male_cv$lambda.1se, family = "binomial")

# Extract and plot nonzero coefficients
table(coef(lasso_male_train, s = "lasso_male_train$lambda.1se")[-1] != 0)

betas <- coef(lasso_male_train, s = "lasso_male_train$lambda.1se")[-1]
names(betas) = rownames(coef(lasso_male_train, s = "lambda.1se"))[-1]

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
#saveRDS(coef_table_male, "Datasets/variable.selection_lasso_male.rds")

# Stability selection 

t0_lasso <- Sys.time()
male_stability_final_lasso <- VariableSelection(xdata = X_train_male, ydata = Y_train_male, family = "binomial", n_cat = 3)
t1_lasso <- Sys.time()
print(t1_lasso - t0_lasso)

par(mar = c(7, 5, 3, 7) + 0.1, oma = c(0, 0, 2, 0))
CalibrationPlot(male_stability_final_lasso)


selprop_male_lasso <- SelectionProportions(male_stability_final_lasso)
print(selprop_male_lasso)

hat_params_male_lasso <- Argmax(male_stability_final_lasso)
print(hat_params_male_lasso)

stable_vars_male_lasso <- selprop_male_lasso[selprop_male_lasso >= 0.99] 
print(stable_vars_male_lasso)

par(mar = c(10, 5, 1, 1))
plot(selprop_male_lasso, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_male_lasso >=
                                                               hat_params_male_lasso[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_male_lasso[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_male_lasso)) {
  axis(side = 1, at = i, labels = names(selprop_male_lasso)[i],
       las = 2, col = ifelse(selprop_male_lasso[i] >= hat_params_male_lasso[2], yes = "red", no = "grey"), col.axis = ifelse(selprop_male_lasso[i] >=
                                                                                                                               hat_params_male_lasso[2], yes = "red", no = "grey"))
}

# Prediction
lasso_pred_male <- predict(lasso_male_train, s = lasso_male_cv$lambda.1se, newx = X_test_male, type = "response")

# Confusion Matrix
lasso_pred_class <- as.factor(ifelse(lasso_pred_male > 0.5, 1, 0))
Y_test_male <- as.factor(Y_test_male)

conf_matrix <- confusionMatrix(as.factor(lasso_pred_class), Y_test_male)
print(conf_matrix)

# AUC plot 
roc_curve_m_og <- roc(as.numeric(Y_test_male), as.numeric(lasso_pred_male))
auc_value_m_og <- as.numeric(auc(roc_curve))
fpr <- 1 - roc_curve$specificities  # False Positive Rate
tpr <- roc_curve$sensitivities      # True Positive Rate

# Save as CSV
roc_data <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value, length(tpr)))
write.csv(roc_data, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_male.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "red", main = "AUC-ROC Curve (male)")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "red", lwd = 2)

#combined male and female original mode var selection plot
betas_fem <- coef(lasso_fem_train, s = "lasso_fem_train$lambda.1se")[-1]
names(betas_fem) <- rownames(coef(lasso_fem_train, s = "lasso_fem_train$lambda.1se"))[-1]

# Extract nonzero coefficients for males
betas_male <- coef(lasso_male_train, s = "lasso_male_train$lambda.1se")[-1]
names(betas_male) <- rownames(coef(lasso_male_train, s = "lasso_male_train$lambda.1se"))[-1]

# Create data frames for male and female betas
beta_male_df <- data.frame(Variable = names(betas_male), Male = betas_male)
beta_fem_df <- data.frame(Variable = names(betas_fem), Female = betas_fem)

# Merge the two data frames by variable name
beta_df <- merge(beta_male_df, beta_fem_df, by = "Variable", all = TRUE)

# Replace NA values with 0 (assuming missing coefficients imply a value of 0)
beta_df[is.na(beta_df)] <- 0
beta_df <- beta_df[!(beta_df$Male == 0 & beta_df$Female == 0), ]

# Load ggplot2
library(ggplot2)

# Create the scatter plot
ggplot(beta_df, aes(x = Male, y = Female, label = Variable)) +
  geom_point(color = "blue", size = 3) +  # Scatter points
  geom_text(aes(label = Variable), vjust = -0.5, hjust = 0.5, size = 3) +  # Add labels
  theme_minimal() +  # Clean theme
  labs(
    x = "Male Beta Value",
    y = "Female Beta Value",
    title = "Scatter Plot of Male vs Female Beta Values"
  )


# Get all unique variables that have nonzero coefficients in either model
all_betas <- union(names(betas_fem[betas_fem != 0]), names(betas_male[betas_male != 0]))

betas_fem_ordered <- betas_fem[all_betas]
betas_male_ordered <- betas_male[all_betas]

betas_fem_ordered[is.na(betas_fem_ordered)] <- 0
betas_male_ordered[is.na(betas_male_ordered)] <- 0
sort_order <- order(-abs(pmax(betas_fem_ordered, betas_male_ordered)))
betas_fem_ordered <- betas_fem_ordered[sort_order]
betas_male_ordered <- betas_male_ordered[sort_order]
all_betas <- all_betas[sort_order]

# Create a dataframe for plotting
coef_df <- data.frame(
  Variable = rep(all_betas, 2),
  Coefficient = c(betas_fem_ordered, betas_male_ordered),
  Gender = rep(c("Female", "Male"), each = length(all_betas))
)


# Create a side-by-side bar plot

ggplot(coef_df, aes(x = reorder(Variable, -abs(Coefficient)), y = Coefficient, fill = Gender)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Lasso Coefficients for Males and Females",
       x = "Variable", y = expression(beta)) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1, size = 6),  # Rotate & shrink text
    axis.text.y = element_text(size = 8),  # Adjust y-axis text size
    plot.title = element_text(size = 10),  # Reduce title size
    legend.position = "bottom",  # Move legend to bottom
    legend.text = element_text(size = 7),  # Reduce legend text size
    plot.margin = margin(2, 10, 5, 5)  # Reduce margins to fit everything
  ) +
  scale_fill_manual(values = c("Female" = "navy", "Male" = "red")) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  coord_cartesian(ylim = c(min(coef_df$Coefficient) * 1.2, max(coef_df$Coefficient) * 1.2))  # Shrink y-axis

#LASSO LOGISTIC REGRESSION
#Female
# Step 1: Extract names of selected variables
selected_vars <- names(stable_vars_fem_lasso)
print(selected_vars)


# Step 2: Define variables to exclude (the ones causing singularities or you want to remove)
vars_to_exclude <- c(
  "Tense / 'highly strung'.0.0_No",
  "Suffer from 'nerves'.0.0_No",
  "Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No",
  "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No"
)

# Step 3: Create a new list of variables after excluding unwanted ones
final_vars <- setdiff(selected_vars, vars_to_exclude)  # keep all except those to exclude

# Step 4: Subset test dataset to final selected variables
X_test_fem_selected_final <- X_test_fem[, final_vars, drop = FALSE]

# Step 5: Prepare data frame for logistic regression
test_data_fem_final <- data.frame(Y = Y_test_fem, X_test_fem_selected_final)

# Step 6: Fit logistic regression model with filtered variables
logistic_model_fem_final <- glm(Y ~ ., data = test_data_fem_final, family = binomial)

# Step 7: Summary of the final logistic regression model
summary(logistic_model_fem_final)

# Step 8: Predict probabilities on test dataset
predicted_probs_fem_final <- predict(logistic_model_fem_final, type = "response")

# Step 9: Optional - Predict class labels (0/1) using threshold 0.5
predicted_classes_fem_final <- ifelse(predicted_probs_fem_final >= 0.5, 1, 0)

# Step 10: Optional - Confusion matrix to evaluate performance
table(Predicted = predicted_classes_fem_final, Actual = Y_test_fem)


#Male -----
# Step 1: Extract names of selected variables (selected from stability selection)
selected_vars_male <- names(stable_vars_male_lasso)
print(selected_vars_male)

vars_to_exclude_male <- c(
  "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes"
)

# Step 3: Create a new list of variables after excluding unwanted ones
final_vars_male <- setdiff(selected_vars_male, vars_to_exclude_male)  # keep all except those to exclude

# Step 2: Subset test dataset to selected variables
X_test_male_selected <- X_test_male[, final_vars_male, drop = FALSE]

# Step 3: Prepare the data frame for logistic regression
test_data_male <- data.frame(Y = Y_test_male, X_test_male_selected)

# Step 4: Fit logistic regression model on selected variables
logistic_model_male <- glm(Y ~ ., data = test_data_male, family = binomial)

# Step 5: Summary of the model to see significant predictors
summary(logistic_model_male)

# Step 6: Predict probabilities on test dataset
predicted_probs_male <- predict(logistic_model_male, type = "response")

# Step 7: Predict class labels (0/1) using threshold of 0.5
predicted_classes_male <- ifelse(predicted_probs_male >= 0.5, 1, 0)

# Step 8: Confusion matrix to evaluate model performance
table(Predicted = predicted_classes_male, Actual = Y_test_male)


##LASSO ANXIETY VS NON ANXIETY#### 

test_fem <- as.data.frame(X_test_fem)
train_fem <- as.data.frame(X_train_fem)
train_male <- as.data.frame(X_train_male)
test_male <- as.data.frame(X_test_male)

#link up X and Y columns before splitting 

train_fem$case_control <- Y_train_fem
test_fem$case_control <- Y_test_fem
train_male$case_control <- Y_train_male
test_male$case_control <- Y_test_male


#split groups up for males  

train_male_anx <- train_male %>%
  filter(if_any(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_Yes`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes`),
    ~ . == 1
  ))

train_male_noanx <- train_male %>%
  filter(if_all(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No`),
    ~ . == 1
  )) 

test_male_anx <- test_male %>%
  filter(if_any(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_Yes`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes`),
    ~ . == 1
  ))

test_male_noanx <- test_male %>%
  filter(if_all(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No`),
    ~ . == 1
  )) 

# and females 
train_fem_anx <- train_fem %>%
  filter(if_any(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_Yes`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes`),
    ~ . == 1
  ))

train_fem_noanx <- train_fem %>%
  filter(if_all(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No`),
    ~ . == 1
  )) 

test_fem_anx <- test_fem %>%
  filter(if_any(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_Yes`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes`),
    ~ . == 1
  ))

test_fem_noanx <- test_fem %>%
  filter(if_all(
    c(`Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No`,
      `Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No`),
    ~ . == 1
  )) 

#split into test and train
Y_train_male_anx <- as.numeric(train_male_anx$case_control)
X_train_male_anx <- as.data.frame(train_male_anx[, !colnames(train_male_anx) %in% c("case_control")])
Y_test_male_anx <- as.numeric(test_male_anx$case_control)
X_test_male_anx <- as.data.frame(test_male_anx[, !colnames(test_male_anx) %in% c("case_control")])

Y_train_fem_anx <- as.numeric(train_fem_anx$case_control)
X_train_fem_anx <- as.data.frame(train_fem_anx[, !colnames(train_fem_anx) %in% c("case_control")])
Y_test_fem_anx <- as.numeric(test_fem_anx$case_control)
X_test_fem_anx <- as.data.frame(test_fem_anx[, !colnames(test_fem_anx) %in% c("case_control")])

Y_train_male_noanx <- as.numeric(train_male_noanx$case_control)
X_train_male_noanx <- as.data.frame(train_male_noanx[, !colnames(train_male_noanx) %in% c("case_control")])
Y_test_male_noanx <- as.numeric(test_male_noanx$case_control)
X_test_male_noanx <- as.data.frame(test_male_noanx[, !colnames(test_male_noanx) %in% c("case_control")])

Y_train_fem_noanx <- as.numeric(train_fem_noanx$case_control)
X_train_fem_noanx <- as.data.frame(train_fem_noanx[, !colnames(train_fem_noanx) %in% c("case_control")])
Y_test_fem_noanx <- as.numeric(test_fem_noanx$case_control)
X_test_fem_noanx <- as.data.frame(test_fem_noanx[, !colnames(test_fem_noanx) %in% c("case_control")])


#remove subset variables from datasets 
remove_susbet_vars <- c("Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_No",
                        "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0_Yes",
                        "Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_No",
                        "Seen doctor (GP) for nerves, anxiety, tension or depression.0.0_Yes"
                        )

X_test_male_anx <- X_test_male_anx %>% select(-all_of(remove_susbet_vars)) 
X_train_male_anx <- X_train_male_anx %>% select(-all_of(remove_susbet_vars))
X_train_male_noanx <- X_train_male_noanx %>% select(-all_of(remove_susbet_vars))
X_test_male_noanx <- X_test_male_noanx %>% select(-all_of(remove_susbet_vars))

X_test_fem_anx <- X_test_fem_anx %>% select(-all_of(remove_susbet_vars))
X_train_fem_anx <- X_train_fem_anx %>% select(-all_of(remove_susbet_vars))
X_train_fem_noanx <- X_train_fem_noanx %>% select(-all_of(remove_susbet_vars))
X_test_fem_noanx <- X_test_fem_noanx %>% select(-all_of(remove_susbet_vars))

#convert to matrix for lasso 
X_test_male_anx <- as.matrix(X_test_male_anx)
X_train_male_anx <- as.matrix(X_train_male_anx)
X_train_male_noanx <- as.matrix(X_train_male_noanx)
X_test_male_noanx <- as.matrix(X_test_male_noanx)

X_test_fem_anx <- as.matrix(X_test_fem_anx)
X_train_fem_anx <- as.matrix(X_train_fem_anx)
X_train_fem_noanx <- as.matrix(X_train_fem_noanx)
X_test_fem_noanx <- as.matrix(X_test_fem_noanx)

#ANXIETY LASSO FOR FEMALES 

#Training: anxiety and non anxiety groups 
lasso_fem_cv_anx <- cv.glmnet(x = X_train_fem_anx, y = Y_train_fem_anx, alpha = 1, family = "binomial")
plot(lasso_fem_cv_anx) 
lasso_fem_train_anx <- glmnet(X_train_fem_anx, Y_train_fem_anx, alpha = 1, lambda = lasso_fem_cv_anx$lambda.1se, family = "binomial")

lasso_fem_cv_noanx <- cv.glmnet(x = X_train_fem_noanx, y = Y_train_fem_noanx, alpha = 1, family = "binomial")
plot(lasso_fem_cv_noanx) 
lasso_fem_train_noanx <- glmnet(X_train_fem_noanx, Y_train_fem_noanx, alpha = 1, lambda = lasso_fem_cv_noanx$lambda.1se, family = "binomial")

#variable selection + var selection plots. extract non 0 vars and plot

table(coef(lasso_fem_train_anx, s = "lasso_fem_train_anx$lambda.1se")[-1] != 0)
table(coef(lasso_fem_train_noanx, s = "lasso_fem_train_noanx$lambda.1se")[-1] != 0)

# Extract non-zero coefficients for anxious and non-anxious groups
betas_anx <- coef(lasso_fem_train_anx, s = "lambda.1se")[-1]
names(betas_anx) <- rownames(coef(lasso_fem_train_anx, s = "lambda.1se"))[-1]

betas_noanx <- coef(lasso_fem_train_noanx, s = "lambda.1se")[-1]
names(betas_noanx) <- rownames(coef(lasso_fem_train_noanx, s = "lambda.1se"))[-1]

# Select features where at least one model has a nonzero coefficient
selected_features <- union(names(betas_anx[betas_anx != 0]), names(betas_noanx[betas_noanx != 0]))

# Filter only selected features
betas_anx_selected <- betas_anx[selected_features]
betas_noanx_selected <- betas_noanx[selected_features]

# Replace NAs with 0 (for consistency)
betas_anx_selected[is.na(betas_anx_selected)] <- 0
betas_noanx_selected[is.na(betas_noanx_selected)] <- 0

# Sort by absolute value of the larger coefficient per feature
sort_order <- order(-abs(pmax(betas_anx_selected, betas_noanx_selected)))
betas_anx_sorted <- betas_anx_selected[sort_order]
betas_noanx_sorted <- betas_noanx_selected[sort_order]
selected_features_sorted <- selected_features[sort_order]

### -------- STABILITY SELECTION FOR ANXIETY FEMALES -------- ###
# Run stability selection for anxiety females
fem_stability_anx_lasso <- VariableSelection(
  xdata = X_train_fem_anx,
  ydata = Y_train_fem_anx,
  family = "binomial",
  n_cat = 3
)

par(mar = c(7, 5, 3, 7) + 0.1, oma = c(0, 0, 2, 0))
CalibrationPlot(fem_stability_anx_lasso)

# Extract selection proportions
selprop_fem_anx_lasso <- SelectionProportions(fem_stability_anx_lasso)
print(selprop_fem_anx_lasso)

# Get optimal parameters and stable variables (adjust threshold if needed)
hat_params_fem_anx_lasso <- Argmax(fem_stability_anx_lasso)
print(hat_params_fem_anx_lasso)
stable_vars_fem_anx_lasso <- selprop_fem_anx_lasso[selprop_fem_anx_lasso >= 0.88] 
print(stable_vars_fem_anx_lasso)

# Plot selection proportions
par(mar = c(10, 5, 1, 1))
plot(selprop_fem_anx_lasso, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_fem_anx_lasso >=
                                                               hat_params_fem_anx_lasso[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_fem_anx_lasso[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_fem_anx_lasso)) {
  axis(side = 1, at = i, labels = names(selprop_fem_anx_lasso)[i],
       las = 2, col = ifelse(selprop_fem_anx_lasso[i] >= hat_params_fem_anx_lasso[2], yes = "red", no = "grey"), col.axis = ifelse(selprop_fem_anx_lasso[i] >=
                                                                                                                                     hat_params_fem_anx_lasso[2], yes = "red", no = "grey"))
}


### -------- STABILITY SELECTION FOR NON-ANXIETY FEMALES -------- ###
# Run stability selection for non-anxiety females
fem_stability_noanx_lasso <- VariableSelection(
  xdata = X_train_fem_noanx,
  ydata = Y_train_fem_noanx,
  family = "binomial",
  n_cat = 3
)

par(mar = c(7, 5, 3, 7) + 0.1, oma = c(0, 0, 2, 0))
CalibrationPlot(fem_stability_noanx_lasso)

# Extract selection proportions
selprop_fem_noanx_lasso <- SelectionProportions(fem_stability_noanx_lasso)
print(selprop_fem_noanx_lasso)

# Get optimal parameters and stable variables (adjust threshold if needed)
hat_params_fem_noanx_lasso <- Argmax(fem_stability_noanx_lasso)
print(hat_params_fem_noanx_lasso)
stable_vars_fem_noanx_lasso <- selprop_fem_noanx_lasso[selprop_fem_noanx_lasso >= 0.9] 
print(stable_vars_fem_noanx_lasso)

# Plot selection proportions
par(mar = c(10, 5, 1, 1))
plot(selprop_fem_noanx_lasso, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_fem_noanx_lasso >=
                                                               hat_params_fem_noanx_lasso[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_fem_noanx_lasso[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_fem_noanx_lasso)) {
  axis(side = 1, at = i, labels = names(selprop_fem_noanx_lasso)[i],
       las = 2, col = ifelse(selprop_fem_noanx_lasso[i] >= hat_params_fem_noanx_lasso[2], yes = "red", no = "grey"), col.axis = ifelse(selprop_fem_noanx_lasso[i] >=
                                                                                                                                         hat_params_fem_noanx_lasso[2], yes = "red", no = "grey"))
}


# Combine into a matrix for side-by-side plotting
beta_matrix <- rbind(betas_anx_sorted, betas_noanx_sorted)

# Define colors
bar_colors <- c("red", "blue")

# Create barplot with side-by-side bars
barplot(beta_matrix, beside = TRUE, col = bar_colors, 
        names.arg = selected_features_sorted, las = 2, cex.names = 0.7, 
        ylab = expression(beta), main = "Female beta coefs - Anx vs. Non-Anx", 
        ylim = c(min(beta_matrix) - 0.1, max(beta_matrix) + 0.1))

# Add legend
legend("topright", legend = c("Anxious", "Non-Anxious"), fill = bar_colors, cex = 0.8, bty = "n")

selected_features <- union(names(betas_anx[betas_anx != 0]), names(betas_noanx[betas_noanx != 0]))
betas_anx_selected <- betas_anx[selected_features]
betas_noanx_selected <- betas_noanx[selected_features]

betas_anx_selected[is.na(betas_anx_selected)] <- 0
betas_noanx_selected[is.na(betas_noanx_selected)] <- 0

# Create a dataframe with variable names and their corresponding beta coefficients
beta_df_fem <- data.frame(
  Variable = selected_features,
  Beta_Anxious = betas_anx_selected,
  Beta_NonAnxious = betas_noanx_selected
)


#Prediction 

lasso_pred_fem_anx <- predict(lasso_fem_train_anx, s = lasso_fem_cv_anx$lambda.1se, newx = X_test_fem_anx, type = "response")
lasso_pred_fem_noanx <- predict(lasso_fem_train_noanx, s = lasso_fem_cv_noanx$lambda.1se, newx = X_test_fem_noanx, type = "response")

# Confusion Matrix
#anx
lasso_pred_class_anx <- as.factor(ifelse(lasso_pred_fem_anx > 0.5, 1, 0))  # Predicted
Y_test_fem_anx <- as.factor(Y_test_fem_anx)  # Actual
conf_matrix_fem_anx <- table(Predicted = lasso_pred_class_anx, Actual = Y_test_fem_anx)
print(conf_matrix_fem_anx) # Print confusion matrix

#noanx
lasso_pred_class_noanx <- as.factor(ifelse(lasso_pred_fem_noanx > 0.5, 1, 0))  # Predicted
Y_test_fem_noanx <- as.factor(Y_test_fem_noanx)  # Actual
conf_matrix_fem_noanx <- table(Predicted = lasso_pred_class_noanx, Actual = Y_test_fem_noanx) # Create confusion matrix
print(conf_matrix_fem_noanx)# Print confusion matrix

# AUC plot 

roc_curve_anx_fem <- roc(as.numeric(Y_test_fem_anx), as.numeric(lasso_pred_fem_anx))
roc_curve_noanx_fem <- roc(as.numeric(Y_test_fem_noanx), as.numeric(lasso_pred_fem_noanx))
auc_anx_f <- as.numeric(auc(roc_curve_anx_fem))
auc_noanx_f <- as.numeric(auc(roc_curve_noanx_fem))
fpr_anx <- 1 - roc_curve_anx_fem$specificities  
tpr_anx <- roc_curve_anx_fem$sensitivities     
fpr_no_anx <- 1 - roc_curve_noanx_fem$specificities 
tpr_no_anx <- roc_curve_noanx_fem$sensitivities      

# Save as CSV
roc_data_anx <- data.frame(FPR = fpr_anx, TPR = tpr_anx, AUC=rep(auc_anx_f, length(tpr_anx)))
write.csv(roc_data_anx, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_female_anx.txt", 
          row.names = FALSE)
roc_data_no_anx <- data.frame(FPR = fpr_no_anx, TPR = tpr_no_anx, AUC=rep(auc_noanx_f, length(tpr_no_anx)))
write.csv(roc_data_no_anx, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_female_no_anx.txt", 
         row.names = FALSE)

# Plot ROC curves on the same plot
plot(roc_curve_anx_fem, col = "blue", main = "AUC-ROC Curve for Female Lasso Regression (Anxiety vs No Anxiety)")
lines(roc_curve_noanx_fem, col = "red")
# Add legend
legend("bottomright", legend = c(paste("Anxiety AUC =", round(auc_anx_f, 3)), 
                                 paste("No Anxiety AUC =", round(auc_noanx_f, 3))),
       col = c("blue", "red"), lwd = 2, cex = 0.6)

#LASSO ANX FOR MALEs 

# Training: anxiety and non-anxiety groups 
lasso_male_cv_anx <- cv.glmnet(x = X_train_male_anx, y = Y_train_male_anx, alpha = 1, family = "binomial")
plot(lasso_male_cv_anx) 
lasso_male_train_anx <- glmnet(X_train_male_anx, Y_train_male_anx, alpha = 1, lambda = lasso_male_cv_anx$lambda.1se, family = "binomial")

lasso_male_cv_noanx <- cv.glmnet(x = X_train_male_noanx, y = Y_train_male_noanx, alpha = 1, family = "binomial")
plot(lasso_male_cv_noanx) 
lasso_male_train_noanx <- glmnet(X_train_male_noanx, Y_train_male_noanx, alpha = 1, lambda = lasso_male_cv_noanx$lambda.1se, family = "binomial")

# Variable selection + var selection plots

table(coef(lasso_male_train_anx, s = "lasso_male_train_anx$lambda.1se")[-1] != 0)
table(coef(lasso_male_train_noanx, s = "lasso_male_train_noanx$lambda.1se")[-1] != 0)

betas_anx <- coef(lasso_male_train_anx, s = "lasso_male_train_anx$lambda.1se")[-1]
names(betas_anx) = rownames(coef(lasso_male_train_anx, s = "lasso_male_train_anx$lambda.1se"))[-1]
betas_noanx <- coef(lasso_male_train_noanx, s = "lasso_male_train_noanx$lambda.1se")[-1]
names(betas_noanx) = rownames(coef(lasso_male_train_noanx, s = "lasso_male_train_noanx$lambda.1se"))[-1]

selected_features <- union(names(betas_anx[betas_anx != 0]), names(betas_noanx[betas_noanx != 0]))
betas_anx_selected <- betas_anx[selected_features]
betas_noanx_selected <- betas_noanx[selected_features]

betas_anx_selected[is.na(betas_anx_selected)] <- 0
betas_noanx_selected[is.na(betas_noanx_selected)] <- 0

# Create a dataframe with variable names and their corresponding beta coefficients
beta_df <- data.frame(
  Variable = selected_features,
  Beta_Anxious = betas_anx_selected,
  Beta_NonAnxious = betas_noanx_selected
)

# Extract non-zero coefficients for anxious and non-anxious male groups
betas_anx <- coef(lasso_male_train_anx, s = "lambda.1se")[-1]
names(betas_anx) <- rownames(coef(lasso_male_train_anx, s = "lambda.1se"))[-1]

betas_noanx <- coef(lasso_male_train_noanx, s = "lambda.1se")[-1]
names(betas_noanx) <- rownames(coef(lasso_male_train_noanx, s = "lambda.1se"))[-1]

# Select features where at least one model has a nonzero coefficient
selected_features <- union(names(betas_anx[betas_anx != 0]), names(betas_noanx[betas_noanx != 0]))

# Filter only selected features
betas_anx_selected <- betas_anx[selected_features]
betas_noanx_selected <- betas_noanx[selected_features]

# Replace NAs with 0 (for consistency)
betas_anx_selected[is.na(betas_anx_selected)] <- 0
betas_noanx_selected[is.na(betas_noanx_selected)] <- 0

# Sort by absolute value of the largest coefficient per feature
sort_order <- order(-abs(pmax(betas_anx_selected, betas_noanx_selected)))
betas_anx_sorted <- betas_anx_selected[sort_order]
betas_noanx_sorted <- betas_noanx_selected[sort_order]
selected_features_sorted <- selected_features[sort_order]

#### -------- STABILITY SELECTION FOR ANXIETY MALES -------- ###

# Run stability selection for male anxiety group
male_stability_anx_lasso <- VariableSelection(
  xdata = X_train_male_anx,
  ydata = Y_train_male_anx,
  family = "binomial",
  n_cat = 3
)

# Calibration plot
par(mar = c(7, 5, 3, 7) + 0.1, oma = c(0, 0, 2, 0))
CalibrationPlot(male_stability_anx_lasso)

# Extract selection proportions
selprop_male_anx_lasso <- SelectionProportions(male_stability_anx_lasso)
print(selprop_male_anx_lasso)

# Find optimal parameters and select stable variables (adjust threshold if needed)
hat_params_male_anx_lasso <- Argmax(male_stability_anx_lasso)
print(hat_params_male_anx_lasso)
stable_vars_male_anx_lasso <- selprop_male_anx_lasso[selprop_male_anx_lasso >= 0.93]  # adjust threshold if needed
print(stable_vars_male_anx_lasso)

# Plot selection proportions
par(mar = c(10, 5, 1, 1))
plot(selprop_male_anx_lasso, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_male_anx_lasso >=
                                                               hat_params_male_anx_lasso[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_male_anx_lasso[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_male_anx_lasso)) {
  axis(side = 1, at = i, labels = names(selprop_male_anx_lasso)[i],
       las = 2, col = ifelse(selprop_male_anx_lasso[i] >= hat_params_male_anx_lasso[2], yes = "red", no = "grey"), col.axis = ifelse(selprop_male_anx_lasso[i] >=
                                                                                                                                       hat_params_male_anx_lasso[2], yes = "red", no = "grey"))
}

### -------- STABILITY SELECTION FOR NON-ANXIETY MALES -------- ###

# Run stability selection for male non-anxiety group
male_stability_noanx_lasso <- VariableSelection(
  xdata = X_train_male_noanx,
  ydata = Y_train_male_noanx,
  family = "binomial",
  n_cat = 3
)

# Calibration plot
par(mar = c(7, 5, 3, 7) + 0.1, oma = c(0, 0, 2, 0))
CalibrationPlot(male_stability_noanx_lasso)

# Extract selection proportions
selprop_male_noanx_lasso <- SelectionProportions(male_stability_noanx_lasso)
print(selprop_male_noanx_lasso)

# Find optimal parameters and select stable variables (adjust threshold if needed)
hat_params_male_noanx_lasso <- Argmax(male_stability_noanx_lasso)
print(hat_params_male_noanx_lasso)
stable_vars_male_noanx_lasso <- selprop_male_noanx_lasso[selprop_male_noanx_lasso >= 0.97]  # adjust threshold if needed
print(stable_vars_male_noanx_lasso)

# Plot selection proportions
par(mar = c(10, 5, 1, 1))
plot(selprop_male_noanx_lasso, type = "h", lwd = 3, las = 1, xlab = "",
     ylab = "Selection Proportion", xaxt = "n", col = ifelse(selprop_male_noanx_lasso >=
                                                               hat_params_male_noanx_lasso[2], yes = "red", no = "grey"), cex.lab = 1.5)
abline(h = hat_params_male_noanx_lasso[2], lty = 2, col = "darkred")
for (i in 1:length(selprop_male_noanx_lasso)) {
  axis(side = 1, at = i, labels = names(selprop_male_noanx_lasso)[i],
       las = 2, col = ifelse(selprop_male_noanx_lasso[i] >= hat_params_male_noanx_lasso[2], yes = "red", no = "grey"), col.axis = ifelse(selprop_male_noanx_lasso[i] >=
                                                                                                                                           hat_params_male_noanx_lasso[2], yes = "red", no = "grey"))
}



# Combine into a matrix for side-by-side plotting
beta_matrix <- rbind(betas_anx_sorted, betas_noanx_sorted)

# Define colors
bar_colors <- c("red", "blue")

# Create barplot with side-by-side bars
barplot(beta_matrix, beside = TRUE, col = bar_colors, 
        names.arg = selected_features_sorted, las = 2, cex.names = 0.7, 
        ylab = expression(beta), main = "Lasso Coefficients - Males (Anxious vs. Non-Anxious)", 
        ylim = c(min(beta_matrix) - 0.1, max(beta_matrix) + 0.1))

# Add legend
legend("topright", legend = c("Anxious", "Non-Anxious"), fill = bar_colors, cex = 0.8, bty = "n")


# Add legend
legend("topright", legend = c("Anxious", "Non-Anxious"), fill = bar_colors, cex = 0.8, bty = "n")


# Prediction 
lasso_pred_male_anx <- predict(lasso_male_train_anx, s = lasso_male_cv_anx$lambda.1se, newx = X_test_male_anx, type = "response")
lasso_pred_male_noanx <- predict(lasso_male_train_noanx, s = lasso_male_cv_noanx$lambda.1se, newx = X_test_male_noanx, type = "response")

# Confusion Matrix
#anx
lasso_pred_class_anx <- as.factor(ifelse(lasso_pred_male_anx > 0.5, 1, 0))
Y_test_male_anx <- as.factor(Y_test_male_anx)
conf_matrix_male_anx <- table(Predicted = lasso_pred_class_anx, Actual = Y_test_male_anx)
print(conf_matrix_male_anx)

#no anxiety 
lasso_pred_class_noanx <- as.factor(ifelse(lasso_pred_male_noanx > 0.5, 1, 0))
Y_test_male_noanx <- as.factor(Y_test_male_noanx)
conf_matrix_male_noanx <- table(Predicted = lasso_pred_class_noanx, Actual = Y_test_male_noanx)
print(conf_matrix_male_noanx)

# AUC plot 
roc_curve_anx_male <- roc(as.numeric(Y_test_male_anx), as.numeric(lasso_pred_male_anx))
roc_curve_noanx_male <- roc(as.numeric(Y_test_male_noanx), as.numeric(lasso_pred_male_noanx))

auc_anx_m <- as.numeric(auc(roc_curve_anx_male))
auc_noanx_m <- as.numeric(auc(roc_curve_noanx_male))
fpr_anx <- 1 - roc_curve_anx_male$specificities  
tpr_anx <- roc_curve_anx_male$sensitivities     
fpr_no_anx <- 1 - roc_curve_noanx_male$specificities 
tpr_no_anx <- roc_curve_noanx_male$sensitivities      

# Save as CSV
roc_data_anx_male <- data.frame(FPR = fpr_anx, TPR = tpr_anx, AUC=rep(auc_anx_m, length(tpr_anx)))
write.csv(roc_data_anx_male, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_male_anx.txt", 
          row.names = FALSE)
roc_data_no_anx_male <- data.frame(FPR = fpr_no_anx, TPR = tpr_no_anx, AUC=rep(auc_noanx_m, length(tpr_no_anx)))
write.csv(roc_data_no_anx, "/rds/general/project/hda_24-25/live/TDS/Group03/roc_curve_lasso_male_no_anx.txt", 
          row.names = FALSE)

# Plot ROC curves on the same plot
plot(roc_curve_anx_male, col = "blue", main = "AUC-ROC Curve for Male Lasso Regression (Anxiety vs No Anxiety)")
lines(roc_curve_noanx_male, col = "red")
legend("bottomright", legend = c(paste("Anxiety AUC =", round(auc_anx_m, 3)), 
                                 paste("No Anxiety AUC =", round(auc_noanx_m, 3))),
       col = c("blue", "red"), lwd = 2, cex = 0.6)

#roc curve for male and female combined 

plot(roc_curve_anx_male, col = "blue", main = "AUC-ROC Curve: anxiety subgroups (lasso)")
lines(roc_curve_noanx_male, col = "red")
lines(roc_curve_anx_fem, col = "green")
lines(roc_curve_noanx_fem, col = "orange")
lines(roc_curve_m_og, col = "lightgrey")
lines(roc_curve_f_og, col = "grey")
# Add legend
legend("bottomright", legend = c(paste("Male Anx AUC =", round(auc_anx_m, 3)), 
                                 paste("Male No Anx AUC =", round(auc_noanx_m, 3)),
                                 paste("Fem Anx AUC =", round(auc_anx_f, 3)), 
                                 paste("Fem No Anx AUC =", round(auc_noanx_f, 3)),
                                 paste("Fem combined AUC =", round(auc_value_f_og, 3)), 
                                 paste("Male combined AUC =", round(auc_value_m_og, 3))),
       col = c("blue", "red", "green","orange", "grey", "lightgrey" ), lwd = 2, cex = 0.45) 

#logistic regression - male anx
selected_vars3 <- names(stable_vars_male_anx_lasso)
print(selected_vars3)

vars_to_exclude <- c("Suffer from 'nerves'.0.0_No")

final_vars3 <- setdiff(selected_vars3, vars_to_exclude)

X_test_male_anx_selected_final <- X_test_male_anx[, final_vars3, drop = FALSE]
test_data_male_anx_final <- data.frame(Y = Y_test_male_anx, X_test_male_anx_selected_final)

logistic_model_male_anx <- glm(Y ~ ., data = test_data_male_anx_final, family = binomial)

summary(logistic_model_male_anx)
tidy(logistic_model_male_anx, exponentiate = TRUE, conf.int = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

#logistic regression - male no anx
selected_vars4 <- names(stable_vars_male_noanx_lasso)
print(selected_vars4)

test_data_male_noanx_final <- data.frame(Y = Y_test_male_noanx, X_test_male_noanx[, selected_vars4, drop = FALSE])

logistic_model_male_noanx <- glm(Y ~ ., data = test_data_male_noanx_final, family = binomial)

summary(logistic_model_male_noanx)
tidy(logistic_model_male_noanx, exponentiate = TRUE, conf.int = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

#LOGISTIC REGRESSION FEMALE ANX
selected_vars5 <- names(stable_vars_fem_anx_lasso)
print(selected_vars5)

vars_to_exclude5 <- c(
  "Tense / 'highly strung'.0.0_No",
  "Suffer from 'nerves'.0.0_No",
  "Long-standing illness, disability or infirmity.0.0_No",
  "cancer by doctor.0.0_No",
  "Ethnic background.0.0_Irish",
  "Sleeplessness / insomnia.0.0_Never/rarely"
  )
final_vars5 <- setdiff(selected_vars5, vars_to_exclude5)
X_test_fem_anx_selected_final <- X_test_fem_anx[, final_vars5, drop = FALSE]
test_data_fem_anx_final <- data.frame(Y = Y_test_fem_anx, X_test_fem_anx_selected_final)
logistic_model_fem_anx_final <- glm(Y ~ ., data = test_data_fem_anx_final, family = binomial)
summary(logistic_model_fem_anx_final)
tidy(logistic_model_fem_anx_final, exponentiate = TRUE, conf.int = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))


#logistic regression female no anx
selected_vars6 <- names(stable_vars_fem_noanx_lasso)
print(selected_vars6)

vars_to_exclude6 <- c(
  "Long-standing illness, disability or infirmity.0.0_No",
  "cancer by doctor.0.0_No",
  "Overall_health_rating_Poor"
)
final_vars6 <- setdiff(selected_vars6, vars_to_exclude6)

X_test_fem_noanx_selected_final <- X_test_fem_noanx[, final_vars6, drop = FALSE]

test_data_fem_noanx_final <- data.frame(Y = Y_test_fem_noanx, X_test_fem_noanx_selected_final)
logistic_model_fem_noanx_final <- glm(Y ~ ., data = test_data_fem_noanx_final, family = binomial)

summary(logistic_model_fem_noanx_final)
tidy(logistic_model_fem_noanx_final, exponentiate = TRUE, conf.int = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

# FOREST PLOT --------------------------------------------------------------------------
# Function to extract odds ratios and confidence intervals
extract_or_ci <- function(model, group) {
  tidy(model) %>%
    mutate(
      Variable = case_when(
        term == "Suffer.from..nerves..0.0_Yes" ~ "Suffer from Nerves (Yes)",
        term == "Loneliness..isolation.0.0_No" ~ "Loneliness/Isolation (No)",
        term == "Long.standing.illness..disability.or.infirmity.0.0_No" ~ "Long-standing Illness (No)",
        term == "Long.standing.illness..disability.or.infirmity.0.0_Yes" ~ "Long-standing Illness (Yes)",
        term == "Current.employment.status.0.0_Employed" ~ "Current Employment Status (Employed)",
        term == "health_score" ~ "Health Score",
        term == "income_score" ~ "Income Score",
        term == "housing_score" ~ "Housing Score",
        term == "Overall_health_rating_Poor" ~ "Overall Health Rating (Poor)",
        term == "Overall_health_rating_Fair" ~ "Overall Health Rating (Fair)",
        term == "Overall_health_rating_Excellent" ~ "Overall Health Rating (Excellent)",
        term == "Pack.years.of.smoking.0.0" ~ "Pack Years of Smoking",
        term == "weekly_usage_phone.0.0_30.59.mins" ~ "Weekly Phone Usage (30–59 mins)",
        term == "morning_evening.person.0.0_morning.person" ~ "Morning Person",
        term == "Sleeplessness...insomnia.0.0_yes" ~ "Sleeplessness/Insomnia (Yes)",
        term == "Sleeplessness...insomnia.0.0_Never.rarely" ~ "Sleeplessness/Insomnia (Never/Rarely)",
        term == "water_intake.0.0" ~ "Water Intake",
        term == "Sensitivity...hurt.feelings.0.0_No" ~ "Sensitivity (No)",
        term == "Tense....highly.strung..0.0_Yes" ~ "Tense/Highly Strung (Yes)",
        term == "Tense....highly.strung..0.0_No" ~ "Tense/Highly Strung (No)",
        term == "cancer.by.doctor.0.0_Yes...you.will.be.asked.about.this.later.by.an.interviewer" ~ "Cancer Diagnosed by Doctor (Yes)",
        term == "cancer.by.doctor.0.0_No" ~ "Cancer Diagnosed by Doctor (No)",
        term == "owner_or_rent.0.0_Rent." ~ "Owner or Rent (Rent)",
        TRUE ~ term
      ),
      OR = exp(estimate),
      lower_CI = exp(estimate - 1.96 * std.error),
      upper_CI = exp(estimate + 1.96 * std.error),
      Group = group
    ) %>%
    filter(term != "(Intercept)") # remove intercept
}

### ---- MALE ----

# Male Anxiety
male_anx_results <- extract_or_ci(loistic_model_male_anx, "Male - Anxiety")

# Male No Anxiety
male_noanx_results <- extract_or_ci(logistic_model_male_noanx, "Male - No Anxiety")

# Combine and plot Male
male_results <- bind_rows(male_anx_results, male_noanx_results)

ggplot(male_results, aes(x = reorder(Variable, OR), y = OR, color = Group)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower_CI, ymax = upper_CI), width = 0.2) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Forest Plot of Odds Ratios - Male",
    x = "Variables",
    y = "Odds Ratio (95% CI)"
  ) +
  scale_color_manual(values = c("Male - Anxiety" = "red", "Male - No Anxiety" = "blue")) +
  geom_hline(yintercept = 1, linetype = "dashed")

### ---- FEMALE ----

# Female Anxiety
female_anx_results <- extract_or_ci(logistic_model_fem_anx_final, "Female - Anxiety")

# Female No Anxiety
female_noanx_results <- extract_or_ci(logistic_model_fem_noanx_final, "Female - No Anxiety")

# Combine and plot Female
female_results <- bind_rows(female_anx_results, female_noanx_results)

ggplot(female_results, aes(x = reorder(Variable, OR), y = OR, color = Group)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower_CI, ymax = upper_CI), width = 0.2) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Forest Plot of Odds Ratios - Female",
    x = "Variables",
    y = "Odds Ratio (95% CI)"
  ) +
  scale_color_manual(values = c("Female - Anxiety" = "red", "Female - No Anxiety" = "blue")) +
  geom_hline(yintercept = 1, linetype = "dashed")

#Elastic Net --------------------------------------------------------

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

# extract coeff
coef_enet <- coef(enet_fem_cv, s = "lambda.1se")
betas <- coef_enet[-1]  # Remove the intercept for plotting
names(betas) <- rownames(coef_enet)[-1]

#table of non-zero coefficients
non_zero_coefs <- betas[betas != 0]
coef_table_enef <- data.frame(Variable = names(non_zero_coefs), 
                              Coefficient = as.numeric(non_zero_coefs))
print(coef_table_enef)

# plotting non zero coeff
par(mar = c(14, 4, 2, 2) + 0.1)
plot(non_zero_coefs, type = "h", col = "navy", lwd = 3,
     xaxt = "n", xlab = "", ylab = expression(beta),
     main = "Nonzero Coefficients (Elastic Net)")
axis(side = 1, at = 1:length(non_zero_coefs), labels = names(non_zero_coefs),
     las = 2, cex.axis = 0.5)
abline(h = 0, lty = 2)

# stability selection
t0 <- Sys.time()
enet_stability <- VariableSelection(xdata = X_train_fem[,-which(colnames(X_train_male) %in% c("Home area population density - urban or rural.0.0_England/Wales - Hamlet and Isolated dwelling",
                                                                                              "hearing_difficulties.0.0_I am completely deaf",
                                                                                              "Ethnic background.0.0_White",
                                                                                              "Home area population density - urban or rural.0.0_Scotland - Small Town",
                                                                                              "Home area population density - urban or rural.0.0_Scotland - Rural",
                                                                                              "Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor.0.0_Blood clot in the lung",
                                                                                              "owner_or_rent.0.0_Others"))], ydata = Y_train_fem, 
                                    family = "binomial", n_cat = 3)
t1 <- Sys.time()
cat("Stability selection runtime:", t1 - t0, "\n")

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

##BIOMARKER + DEMOGRAPHIC VARS - is there a gain in power? 

X_train_fem <- readRDS("Datasets/imputed_train_fem_bio_X.rds")
X_test_fem <- readRDS("Datasets/imputed_test_fem_bio_X.rds")

X_train_male <- readRDS("Datasets/imputed_train_male_bio_X.rds")
X_test_male <- readRDS("Datasets/imputed_test_male_bio_X.rds")

Y_train_fem <- readRDS("Datasets/Y_train_fem_bio.rds")
Y_test_fem <- readRDS("Datasets/Y_test_fem_bio.rds")

Y_train_male <- readRDS("Datasets/Y_train_male_bio.rds")
Y_test_male <- readRDS("Datasets/Y_test_male_bio.rds")

#for fems 
train_numeric_vars <- X_train_fem %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)

names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_fem[, col] <- (X_train_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_fem[, col]  <- (X_test_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}

#for male 

train_numeric_vars <- X_train_male %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)
names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_male[, col] <- (X_train_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_male[, col]  <- (X_test_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}


#one hot encode 

#for female  train 

X_train_fem <- as.data.table(X_train_fem)

factor_vars <- sapply(X_train_fem, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_train_fem <- one_hot(X_train_fem, cols = var)
}

X_train_fem <- as.data.frame(X_train_fem)

# for female test 

X_test_fem <- as.data.table(X_test_fem)

factor_vars <- sapply(X_test_fem, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_test_fem <- one_hot(X_test_fem, cols = var)
}

X_test_fem <- as.data.frame(X_test_fem)

#for male train

X_train_male <- as.data.table(X_train_male)

factor_vars <- sapply(X_train_male, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_train_male <- one_hot(X_train_male, cols = var)
}

X_train_male <- as.data.frame(X_train_male)

#for male test 

X_test_male <- as.data.table(X_test_male)

factor_vars <- sapply(X_test_male, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_test_male <- one_hot(X_test_male, cols = var)
}

X_test_male <- as.data.frame(X_test_male)


#change data types for lasso
X_train_male <- as.matrix(X_train_male)
X_train_fem <- as.matrix(X_train_fem)
X_test_male <- as.matrix(X_test_male)
X_test_fem <- as.matrix(X_test_fem)
Y_train_fem <- as.numeric(Y_train_fem)
Y_test_fem <- as.numeric(Y_test_fem)
Y_train_male <- as.numeric(Y_train_male)
Y_test_male <- as.numeric(Y_test_male)


## run lasso 
#for females 

#for females 

lasso_fem_cv <- cv.glmnet(x = X_train_fem, y = Y_train_fem, alpha = 1, family = "binomial")
plot(lasso_fem_cv)

lasso_fem_train <- glmnet(X_train_fem, Y_train_fem, alpha = 1, lambda = lasso_fem_cv$lambda.1se, family = "binomial")

# prediction
lasso_pred_fem <- predict(lasso_fem_train, s = lasso_fem_cv$lambda.1se, newx = X_test_fem, type = "response")

# Confusion Matrix
lasso_pred_class <- as.factor(ifelse(lasso_pred_fem > 0.5, 1, 0))
Y_test_fem <- as.factor(Y_test_fem)
Y_test_fem <- factor(ifelse(Y_test_fem == "2", "1", "0"))

conf_matrix <- confusionMatrix(as.factor(lasso_pred_class), Y_test_fem)
print(conf_matrix)

#AUC plot 
roc_curve <- roc(as.numeric(Y_test_fem), as.numeric(lasso_pred_fem))
auc_value <- as.numeric(auc(roc_curve))
fpr <- 1 - roc_curve$specificities  # False Positive Rate
tpr <- roc_curve$sensitivities      # True Positive Rate


# Save as CSV
roc_data <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value, length(tpr)))
write.csv(roc_data, "/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_fem_all_variables_T.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", main = "AUC-ROC Curve (female)")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)

# for males 

lasso_male_cv <- cv.glmnet(x = X_train_male, y = Y_train_male, alpha = 1, family = "binomial")
plot(lasso_male_cv)

lasso_male_train <- glmnet(X_train_male, Y_train_male, alpha = 1, lambda = lasso_male_cv$lambda.1se, family = "binomial")

# prediction
lasso_pred_male <- predict(lasso_male_train, s = lasso_male_cv$lambda.1se, newx = X_test_male, type = "response")

# Confusion Matrix
lasso_pred_class_male <- as.factor(ifelse(lasso_pred_male > 0.5, 1, 0))
Y_test_male <- as.factor(Y_test_male)
Y_test_male <- factor(ifelse(Y_test_male == "2", "1", "0"))

conf_matrix_male <- confusionMatrix(as.factor(lasso_pred_class_male), Y_test_male)
print(conf_matrix_male)

# AUC plot 
roc_curve_male <- roc(as.numeric(Y_test_male), as.numeric(lasso_pred_male))
auc_value_male <- as.numeric(auc(roc_curve_male))
fpr_male <- 1 - roc_curve_male$specificities  # False Positive Rate
tpr_male <- roc_curve_male$sensitivities      # True Positive Rate

# Save as CSV
roc_data_male <- data.frame(FPR = fpr_male, TPR = tpr_male, AUC = rep(auc_value_male, length(tpr_male)))
write.csv(roc_data_male, "/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_male_all_variables_T.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value_male))

plot(roc_curve_male, col = "red", main = "AUC-ROC Curve (male)")
legend("bottomright", legend = paste("AUC =", round(auc_value_male, 3)), col = "red", lwd = 2)

#combined ROC plot for all variables (including biomarkers) using lasso 

plot(roc_curve, col = "blue", main = "AUC-ROC Curve (Male vs Female)")
legend_text <- c(paste("Female AUC =", round(auc_value, 3)), paste("Male AUC =", round(auc_value_male, 3)))

lines(roc_curve_male, col = "red")

legend("bottomright", legend = legend_text, col = c("blue", "red"), lwd = 2)

# DEMOGRAPHIC VARS without biomarkers - is there significant loss of power? 

X_train_fem <- readRDS("Datasets/imputed_train_fem_bio_X.rds")
X_test_fem <- readRDS("Datasets/imputed_test_fem_bio_X.rds")

X_train_male <- readRDS("Datasets/imputed_train_male_bio_X.rds")
X_test_male <- readRDS("Datasets/imputed_test_male_bio_X.rds")

Y_train_fem <- readRDS("Datasets/Y_train_fem_bio.rds")
Y_test_fem <- readRDS("Datasets/Y_test_fem_bio.rds")

Y_train_male <- readRDS("Datasets/Y_train_male_bio.rds")
Y_test_male <- readRDS("Datasets/Y_test_male_bio.rds")

global_list_bio = readRDS("/rds/general/project/hda_24-25/live/TDS/Group03/list_bio_var_global.rds")
X_train_fem <- X_train_fem[,!colnames(X_train_fem) %in% global_list_bio]
X_test_fem <- X_test_fem[,!colnames(X_test_fem) %in% global_list_bio]
X_train_male <- X_train_male[,!colnames(X_train_male) %in% global_list_bio]
X_test_male <- X_test_male[,!colnames(X_test_male) %in% global_list_bio]

#for fems 
train_numeric_vars <- X_train_fem %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)

names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_fem[, col] <- (X_train_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_fem[, col]  <- (X_test_fem[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}

#for male 

train_numeric_vars <- X_train_male %>% select_if(is.numeric) 

Train_means <- data.frame(as.list(train_numeric_vars %>% apply(2, mean)))
Train_stddevs <- data.frame(as.list(train_numeric_vars %>% apply(2, sd)))

col_names <- names(train_numeric_vars)
names(Train_means) <- colnames(train_numeric_vars)
names(Train_stddevs) <- colnames(train_numeric_vars)

for (i in 1:length(col_names)) {
  col <- col_names[i]
  X_train_male[, col] <- (X_train_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
  X_test_male[, col]  <- (X_test_male[[col]] - Train_means[[col]]) / Train_stddevs[[col]]
}


#one hot encode 

#for female  train 

X_train_fem <- as.data.table(X_train_fem)

factor_vars <- sapply(X_train_fem, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_train_fem <- one_hot(X_train_fem, cols = var)
}

X_train_fem <- as.data.frame(X_train_fem)

# for female test 

X_test_fem <- as.data.table(X_test_fem)

factor_vars <- sapply(X_test_fem, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_test_fem <- one_hot(X_test_fem, cols = var)
}

X_test_fem <- as.data.frame(X_test_fem)

#for male train

X_train_male <- as.data.table(X_train_male)

factor_vars <- sapply(X_train_male, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_train_male <- one_hot(X_train_male, cols = var)
}

X_train_male <- as.data.frame(X_train_male)

#for male test 

X_test_male <- as.data.table(X_test_male)

factor_vars <- sapply(X_test_male, function(x) is.factor(x))
factor_vars <- names(factor_vars[factor_vars])

for (var in factor_vars) {
  X_test_male <- one_hot(X_test_male, cols = var)
}

X_test_male <- as.data.frame(X_test_male)


#change data types for lasso
X_train_male <- as.matrix(X_train_male)
X_train_fem <- as.matrix(X_train_fem)
X_test_male <- as.matrix(X_test_male)
X_test_fem <- as.matrix(X_test_fem)
Y_train_fem <- as.numeric(Y_train_fem)
Y_test_fem <- as.numeric(Y_test_fem)
Y_train_male <- as.numeric(Y_train_male)
Y_test_male <- as.numeric(Y_test_male)


## run lasso 
#for females 

#for females 

lasso_fem_cv <- cv.glmnet(x = X_train_fem, y = Y_train_fem, alpha = 1, family = "binomial")
plot(lasso_fem_cv)

lasso_fem_train <- glmnet(X_train_fem, Y_train_fem, alpha = 1, lambda = lasso_fem_cv$lambda.1se, family = "binomial")

# prediction
lasso_pred_fem <- predict(lasso_fem_train, s = lasso_fem_cv$lambda.1se, newx = X_test_fem, type = "response")

# Confusion Matrix
lasso_pred_class <- as.factor(ifelse(lasso_pred_fem > 0.5, 1, 0))
Y_test_fem <- as.factor(Y_test_fem)
Y_test_fem <- factor(ifelse(Y_test_fem == "2", "1", "0"))

conf_matrix <- confusionMatrix(as.factor(lasso_pred_class), Y_test_fem)
print(conf_matrix)

#AUC plot 
roc_curve <- roc(as.numeric(Y_test_fem), as.numeric(lasso_pred_fem))
auc_value <- as.numeric(auc(roc_curve))
fpr <- 1 - roc_curve$specificities  # False Positive Rate
tpr <- roc_curve$sensitivities      # True Positive Rate


# Save as CSV
roc_data <- data.frame(FPR = fpr, TPR = tpr, AUC=rep(auc_value, length(tpr)))
write.csv(roc_data, "/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_fem_all_variables_but_biomarkers.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", main = "AUC-ROC Curve (female)")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)

# for males 

lasso_male_cv <- cv.glmnet(x = X_train_male, y = Y_train_male, alpha = 1, family = "binomial")
plot(lasso_male_cv)

lasso_male_train <- glmnet(X_train_male, Y_train_male, alpha = 1, lambda = lasso_male_cv$lambda.1se, family = "binomial")

# prediction
lasso_pred_male <- predict(lasso_male_train, s = lasso_male_cv$lambda.1se, newx = X_test_male, type = "response")

# Confusion Matrix
lasso_pred_class_male <- as.factor(ifelse(lasso_pred_male > 0.5, 1, 0))
Y_test_male <- as.factor(Y_test_male)
Y_test_male <- factor(ifelse(Y_test_male == "2", "1", "0"))

conf_matrix_male <- confusionMatrix(as.factor(lasso_pred_class_male), Y_test_male)
print(conf_matrix_male)

# AUC plot 
roc_curve_male <- roc(as.numeric(Y_test_male), as.numeric(lasso_pred_male))
auc_value_male <- as.numeric(auc(roc_curve_male))
fpr_male <- 1 - roc_curve_male$specificities  # False Positive Rate
tpr_male <- roc_curve_male$sensitivities      # True Positive Rate

# Save as CSV
roc_data_male <- data.frame(FPR = fpr_male, TPR = tpr_male, AUC = rep(auc_value_male, length(tpr_male)))
write.csv(roc_data_male, "/rds/general/project/hda_24-25/live/TDS/Group03/plots/roc_curve_lasso_male_all_variables_but_biomarkers.txt", 
          row.names = FALSE)
print(paste("AUC:", auc_value_male))

plot(roc_curve_male, col = "red", main = "AUC-ROC Curve (male)")
legend("bottomright", legend = paste("AUC =", round(auc_value_male, 3)), col = "red", lwd = 2)

#combined ROC plot for all variables (including biomarkers) using lasso 

plot(roc_curve, col = "blue", main = "AUC-ROC Curve (Male vs Female)")
legend_text <- c(paste("Female AUC =", round(auc_value, 3)), paste("Male AUC =", round(auc_value_male, 3)))

lines(roc_curve_male, col = "red")

legend("bottomright", legend = legend_text, col = c("blue", "red"), lwd = 2)



