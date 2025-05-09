
import pyreadr
import pandas as pd
import json

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_train_male_nobio_RF.rds")
X_train = list(results.values())[0]

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/Y_train_male.rds")
Y_train = list(results.values())[0]
Y_train = Y_train.values.ravel()  # Converts (n,1) to (n,)

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/Y_test_male.rds")
Y_test = list(results.values())[0]
Y_test = Y_test.values.ravel()  # Converts (n,1) to (n,)

results = pyreadr.read_r("/rds/general/project/hda_24-25/live/TDS/Group03/Datasets/X_test_male_nobio_RF.rds")
X_test = list(results.values())[0]

sex="male"

save_path="/rds/general/project/hda_24-25/live/TDS/Group03/XGboost"

import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import xgboost as xgb

categorical_columns = [col for col in X_train.columns if str(X_train[col].dtype) == 'category']

label_encoders = {}

# Apply label encoding to all categorical columns
for col in categorical_columns:  # Replace with your actual list of categorical columns
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.fit_transform(X_test[col])
    label_encoders[col] = le  # Store the encoder for future inverse transformations
  

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
    'subsample': np.linspace(0.5, 0.9, 5),
    'colsample_bytree': np.linspace(0.5, 0.9, 5),
    'n_estimators': [50,100, 500, 1000],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
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

metrics_df.to_csv(f"{save_path}/classification_metrics_{sex}.csv", index=False)

print('Accuracy of {}, f1 score of {}, precision of {},  recall of {}, specificity of {} and sensitivity of {}'.format(acc, f1, precision, recall, specificity, sensitivity))

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

auc = roc_auc_score(y_true, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{save_path}/roc_curve_{sex}.png", dpi=300, bbox_inches="tight")

# Save FPR and TPR as a CSV file
np.savetxt(f"{save_path}/roc_curve_XGboost_{sex}.txt",
           np.column_stack((fpr, tpr)), delimiter=",", header="FPR,TPR", comments="")

# Save AUC separately
with open(f"{save_path}/auc_XGboost_{sex}.txt", "w") as f:
    f.write(f"AUC: {auc}\n")

#Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f"{save_path}/confusion_matrix_{sex}.png", dpi=300, bbox_inches="tight")


#Importance Plot
xgb.plot_importance(bst, importance_type='gain', max_num_features=30)  # Top 30 features
plt.savefig(f"{save_path}/variable_importance_{sex}.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure

feature_importance = bst.get_score(importance_type="gain")  # "weight" counts how many times a feature is used

with open(f"{save_path}/variable_importance_{sex}.json", "w") as f:
    json.dump(feature_importance, f)

import shap

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(dtest)


# --- SHAP Bar Plot ---

plt.figure() 
shap.summary_plot(shap_values, features=X_test, plot_type="bar", show=False, plot_size=[12,6])
plt.title('SHAP Plot for BZD Repetitions')
plt.savefig(f"{save_path}/shap_bar_plot_{sex}.png", dpi=300)
plt.close()  # Close the figure to avoid overlap

# --- SHAP Density Plot ---
plt.figure()  
shap.summary_plot(shap_values, features=X_test, show=False, plot_size=[12,6])
plt.title('SHAP Density Plot for BZD Repetitions')
plt.savefig(f"{save_path}/shap_density_plot_{sex}.png", dpi=300)
plt.close()  # Close the figure

