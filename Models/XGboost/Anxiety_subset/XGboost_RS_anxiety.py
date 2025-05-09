
import pyreadr
import pandas as pd
import json

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/imputed_train_male_X_recode.rds")
X_train = list(results.values())[0]

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/Y_train_male.rds")
Y_train = list(results.values())[0]
Y_train = Y_train.values.ravel()  # Converts (n,1) to (n,)

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/Y_test_male.rds")
Y_test = list(results.values())[0]
Y_test = Y_test.values.ravel()  # Converts (n,1) to (n,)

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/imputed_test_male_X_recode.rds")
X_test = list(results.values())[0]

sex="male"

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

categorical_columns = [col for col in X_train.columns if str(X_train[col].dtype) == 'category']

label_encoders = {}

# Apply label encoding to all categorical columns
for col in categorical_columns:  # Replace with your actual list of categorical columns
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.fit_transform(X_test[col])
    label_encoders[col] = le  # Store the encoder for future inverse transformations
  

full_train = pd.concat([X_train, pd.DataFrame(Y_train)], axis=1)

full_train_anx = full_train[(full_train["Seen doctor (GP) for nerves, anxiety, tension or depression.0.0"]==1) | (full_train["Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"==1])]
full_train_no_anx = full_train[(full_train["Seen doctor (GP) for nerves, anxiety, tension or depression.0.0"]==0) & (full_train["Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"==0])]

X_train_anx = full_train_anx.iloc[:, :-1]
Y_train_anx = full_train_anx.iloc[:, -1].values.ravel()

X_train_no_anx = full_train_no_anx.iloc[:, :-1]
Y_train_no_anx = full_train_no_anx.iloc[:, -1].values.ravel()

full_test = pd.concat([X_test, pd.DataFrame(Y_test)], axis=1)

full_test_anx = full_test[(full_test["Seen doctor (GP) for nerves, anxiety, tension or depression.0.0"]==1) | (full_test["Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"==1])]
full_test_no_anx = full_test[(full_test["Seen doctor (GP) for nerves, anxiety, tension or depression.0.0"]==0) & (full_test["Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"==0])]

X_test_anx = full_test_anx.iloc[:, :-1]
Y_test_anx = full_test_anx.iloc[:, -1].values.ravel()


X_test_no_anx = full_test_no_anx.iloc[:, :-1]
Y_test_no_anx = full_test_no_anx.iloc[:, -1].values.ravel()


#MODELLING FOR NON-ANXIETY PEOPLE
anx="no_anx"

X_train =X_train_no_anx 
Y_train =Y_train_no_anx 
X_test =X_test_no_anx 
Y_test =Y_test_no_anx 

X_train = X_train.drop(columns=['Seen doctor (GP) for nerves, anxiety, tension or depression.0.0', "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"])
X_test = X_test.drop(columns=['Seen doctor (GP) for nerves, anxiety, tension or depression.0.0', "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"])


# Create regression matrices
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

dtrain = xgb.DMatrix(X_train, Y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, Y_test, enable_categorical=True)


param = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    }

xgb_clf = xgb.XGBClassifier(**param)

param_dist = {
    'objective': ['binary:logistic'],  # Add this line
    'max_depth': np.arange(3, 10, 2),
    'learning_rate': np.linspace(0.01, 0.2, 5),
    'subsample': np.linspace(0.5, 0.9, 3),
    'colsample_bytree': np.linspace(0.5, 0.9, 3),
    'n_estimators': [100, 500, 1000],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.3],
}

random_search = RandomizedSearchCV(
    estimator=xgb_clf, 
    param_distributions=param_dist, 
    n_iter=20,  # Try 20 random combinations
    scoring='roc_auc', 
    cv=5, 
    verbose=2, 
    random_state=42, 
    n_jobs=10,
    error_score="raise"  # This forces it to display full errors
)

random_search.fit(X_train, Y_train)

print("Best parameters:", random_search.best_params_)

best_params = random_search.best_params_ 
param.update(best_params)
    
bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=best_params['n_estimators'])


#METRICS CALCULATION
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

y_pred_proba = bst.predict(dtest)

# Convert to binary predictions (0 or 1) using 0.5 threshold
y_pred = (y_pred_proba > 0.5).astype(int)

# Get true labels from dtest
y_true = dtest.get_label()

# Compute metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary')  # Assuming binary classification
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', "Specificity", "Sensitivity"],
    'Value': [acc, f1, precision, recall, specificity, sensitivity]
})

metrics_df.to_csv(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/classification_metrics_{sex}_{anx}.csv", index=False)

print('Accuracy of {}, f1 score of {}, precision of {},  recall of {}, specificity of {} and sensitivity of {}'.format(acc, f1, precision, recall, specificity, sensitivity))

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

auc = roc_auc_score(y_true, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for {sex} {anx}")
plt.legend()
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/roc_curve_{sex}_{anx}.png", dpi=300, bbox_inches="tight")

# Save FPR and TPR as a CSV file
np.savetxt(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/roc_curve_XGboost_{sex}_{anx}.txt",
           np.column_stack((fpr, tpr)), delimiter=",", header="FPR,TPR", comments="")

# Save AUC separately
with open(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/auc_XGboost_{sex}_{anx}.txt", "w") as f:
    f.write(f"AUC: {auc}\n")
    
#Confusion Matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for {sex} {anx}')
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/confusion_matrix_{sex}_{anx}.png", dpi=300, bbox_inches="tight")


#Importance Plot
xgb.plot_importance(bst, importance_type='gain', max_num_features=30)  # Top 30 features
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/variable_importance_{sex}_{anx}.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure

feature_importance = bst.get_score(importance_type="gain")  # "weight" counts how many times a feature is used

with open(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/variable_importance_{sex}_{anx}.json", "w") as f:
    json.dump(feature_importance, f)

import shap

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(dtest)

# Define save path
save_path = "/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/"

# --- SHAP Bar Plot ---

plt.figure() 
shap.summary_plot(shap_values, features=X_test, plot_type="bar", show=False, plot_size=[12,6])
plt.title(f'SHAP Plot for BZD Repetitions for {sex} {anx}')
plt.savefig(f"{save_path}shap_bar_plot_{sex}_{anx}.png", dpi=300)
plt.close()  # Close the figure to avoid overlap

# --- SHAP Density Plot ---
plt.figure()  
shap.summary_plot(shap_values, features=X_test, show=False, plot_size=[12,6])
plt.title(f'SHAP Density Plot for BZD Repetitions for {sex} {anx}')
plt.savefig(f"{save_path}shap_density_plot_{sex}_{anx}.png", dpi=300)
plt.close()  # Close the figure


#MODELLING FOR ANXIETY PEOPLE
anx="anx"

X_train =X_train_anx 
Y_train =Y_train_anx 
X_test =X_test_anx 
Y_test =Y_test_anx 

X_train = X_train.drop(columns=['Seen doctor (GP) for nerves, anxiety, tension or depression.0.0', "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"])
X_test = X_test.drop(columns=['Seen doctor (GP) for nerves, anxiety, tension or depression.0.0', "Seen a psychiatrist for nerves, anxiety, tension or depression.0.0"])


# Create regression matrices
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

dtrain = xgb.DMatrix(X_train, Y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, Y_test, enable_categorical=True)


param = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    }

xgb_clf = xgb.XGBClassifier(**param)

param_dist = {
    'objective': ['binary:logistic'],  # Add this line
    'max_depth': np.arange(3, 10, 2),
    'learning_rate': np.linspace(0.01, 0.2, 5),
    'subsample': np.linspace(0.5, 0.9, 3),
    'colsample_bytree': np.linspace(0.5, 0.9, 3),
    'n_estimators': [100, 500, 1000],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.3],
}

random_search = RandomizedSearchCV(
    estimator=xgb_clf, 
    param_distributions=param_dist, 
    n_iter=20,  # Try 20 random combinations
    scoring='roc_auc', 
    cv=5, 
    verbose=2, 
    random_state=42, 
    n_jobs=10,
    error_score="raise"  # This forces it to display full errors
)

random_search.fit(X_train, Y_train)

print("Best parameters:", random_search.best_params_)

best_params = random_search.best_params_ 
param.update(best_params)
    
bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=best_params['n_estimators'])


#METRICS CALCULATION
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

y_pred_proba = bst.predict(dtest)

# Convert to binary predictions (0 or 1) using 0.5 threshold
y_pred = (y_pred_proba > 0.5).astype(int)

# Get true labels from dtest
y_true = dtest.get_label()

# Compute metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary')  # Assuming binary classification
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', "Specificity", "Sensitivity"],
    'Value': [acc, f1, precision, recall, specificity, sensitivity]
})

metrics_df.to_csv(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/classification_metrics_{sex}_{anx}.csv", index=False)

print('Accuracy of {}, f1 score of {}, precision of {},  recall of {}, specificity of {} and sensitivity of {}'.format(acc, f1, precision, recall, specificity, sensitivity))

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

auc = roc_auc_score(y_true, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for {sex} {anx}")
plt.legend()
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/roc_curve_{sex}_{anx}.png", dpi=300, bbox_inches="tight")

# Save FPR and TPR as a CSV file
np.savetxt(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/roc_curve_XGboost_{sex}_{anx}.txt",
           np.column_stack((fpr, tpr)), delimiter=",", header="FPR,TPR", comments="")

# Save AUC separately
with open(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/auc_XGboost_{sex}_{anx}.txt", "w") as f:
    f.write(f"AUC: {auc}\n")
    
#Confusion Matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for {sex} {anx}')
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/confusion_matrix_{sex}_{anx}.png", dpi=300, bbox_inches="tight")


#Importance Plot
xgb.plot_importance(bst, importance_type='gain', max_num_features=30)  # Top 30 features
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/variable_importance_{sex}_{anx}.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure

feature_importance = bst.get_score(importance_type="gain")  # "weight" counts how many times a feature is used

with open(f"/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/variable_importance_{sex}_{anx}.json", "w") as f:
    json.dump(feature_importance, f)

import shap

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(dtest)

# Define save path
save_path = "/rds/general/project/hda_24-25/live/TDS/Group03/XGboost/Anxiety_subset/"

# --- SHAP Bar Plot ---

plt.figure() 
shap.summary_plot(shap_values, features=X_test, plot_type="bar", show=False, plot_size=[12,6])
plt.title(f'SHAP Plot for BZD Repetitions for {sex} {anx}')
plt.savefig(f"{save_path}shap_bar_plot_{sex}_{anx}.png", dpi=300)
plt.close()  # Close the figure to avoid overlap

# --- SHAP Density Plot ---
plt.figure()  
shap.summary_plot(shap_values, features=X_test, show=False, plot_size=[12,6])
plt.title(f'SHAP Density Plot for BZD Repetitions for {sex} {anx}')
plt.savefig(f"{save_path}shap_density_plot_{sex}_{anx}.png", dpi=300)
plt.close()  # Close the figure
