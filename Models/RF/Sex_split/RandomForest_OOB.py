
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

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

categorical_columns = [col for col in X_train.columns if str(X_train[col].dtype) == 'category']

label_encoders = {}

# Apply label encoding to all categorical columns
for col in categorical_columns:  # Replace with your actual list of categorical columns
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.fit_transform(X_test[col])
    label_encoders[col] = le  # Store the encoder for future inverse transformations
    
rf_params = {
    'max_features': ['sqrt', 'log2', 0.3, 0.5],  
    'n_estimators': [500, 1000, 1250],  
    'max_depth': range(3, 8),  
    'min_samples_leaf': [5, 10, 15], 
    'min_samples_split': range(2, 5),  
    'criterion': ['gini', 'entropy']
}

def oob_mse(additional_rf_params={}):
    model = RandomForestClassifier(oob_score=True, **additional_rf_params, n_jobs= 5, random_state=5225678, class_weight="balanced_subsample")
    model.fit(X_train, Y_train)
    oob_error = 1 - model.oob_score_
    return oob_error, model # OOB error = 1 - OOB accuracy

# Iterate over all parameter combinations
def evaluate_params(param_set):
    oob_error, model = oob_mse(param_set)
    return {"params": param_set, "oob": oob_error, "model": model}

results = Parallel(n_jobs=4, verbose=1)(
    delayed(evaluate_params)(params) for params in ParameterGrid(rf_params)
)

# Convert results to DataFrame for easy sorting & analysis
df_results = pd.DataFrame(results).sort_values(by="oob")

best_result = df_results.iloc[0]  # Lowest OOB error
best_params, best_oob, best_model = best_result["params"], best_result["oob"], best_result["model"]

print("Best Hyperparameters:", best_params)
print("Best OOB MSE:", best_oob)

with open(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/best_params_OOB_{sex}.json", "w") as f:
    json.dump(best_params, f)

y_pred = best_model.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


y_test_int = Y_test.astype(int)
y_pred_int = y_pred.astype(int)

acc = accuracy_score(y_test_int, y_pred_int)
f1 = f1_score(y_test_int, y_pred_int, average='binary')  # Assuming binary classification
precision = precision_score(y_test_int, y_pred_int, average='binary')
recall = recall_score(y_test_int, y_pred_int, average='binary')

tn, fp, fn, tp = confusion_matrix(y_test_int, y_pred_int).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', "Specificity", "Sensitivity"],
    'Value': [acc, f1, precision, recall, specificity, sensitivity]
})

metrics_df.to_csv(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/classification_metrics_{sex}.csv", index=False)

print('Accuracy of {}, f1 score of {}, precision of {},  recall of {}, specificity of {} and sensitivity of {}'.format(acc, f1, precision, recall, specificity, sensitivity))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Get predicted probabilities for the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

Y_test_arr = np.array(Y_test, dtype=int)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(Y_test_arr, y_pred_proba)

auc = roc_auc_score(Y_test_arr, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Save FPR and TPR as a CSV file
np.savetxt(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/roc_curve_RF_{sex}.txt",
           np.column_stack((fpr, tpr)), delimiter=",", header="FPR,TPR", comments="")

# Save AUC separately
with open(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/auc_RF_{sex}.txt", "w") as f:
    f.write(f"AUC: {auc}\n")

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_int, y_pred_int)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/confusion_matrix_{sex}.png", dpi=300, bbox_inches="tight")

plt.show()


#### Plotting the Variable importance plot
import matplotlib.pyplot as plt

# Number of top features to display
top_n = 30  # Adjust as needed

importances = best_model.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[-top_n:]  # Select top N features

with open(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/importances_var_{sex}.json", "w") as f:
    json.dump(importances.tolist(), f)

# Increase figure size for readability
plt.figure(figsize=(12, 8))

# Plot horizontal bar chart
plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# Set feature labels
plt.yticks(range(len(indices)), [features[i] for i in indices])

# Labels and title
plt.xlabel('Relative Importance')
plt.title(f'Top {top_n} Feature Importances')
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/variable_importance_plot_{sex}.png", dpi=300, bbox_inches="tight")


#SHAP 
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)


# --- SHAP Bar Plot ---
plt.figure()  # Ensure a fresh figure
shap.summary_plot(shap_values[:, :, 1], features=X_test, plot_type="bar", show=False, plot_size=[12,6])
plt.title('SHAP Plot for BZD Repetitions')
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/shap_bar_plot_{sex}.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure to avoid overlap

# --- SHAP Density Plot ---
plt.figure()  # Ensure a fresh figure
shap.summary_plot(shap_values[:, :, 1], features=X_test, show=False, plot_size=[12,6])
plt.title('SHAP Density Plot for BZD Repetitions')
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/shap_density_plot_{sex}.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure

#PERMUTATION PLOT
from sklearn.inspection import permutation_importance

result = permutation_importance(best_model, X_test, Y_test, n_repeats=5,random_state=5225678, n_jobs=5)

sorted_idx = result.importances_mean.argsort()

# Choose top N features for readability
top_n = 30  
top_idx = sorted_idx[-top_n:]  

fig, ax = plt.subplots(figsize=(12, 8))  # Wider for better readability
ax.boxplot(result.importances[top_idx].T, vert=False, tick_labels=X_test.columns[top_idx])
ax.set_title(f"Top {top_n} Permutation Importances (Test Set)")
ax.set_xlabel("Importance Score")

fig.tight_layout()
plt.savefig(f"/rds/general/project/hda_24-25/live/TDS/Group03/RF/permutation_importance_plot_{sex}.png", dpi=300, bbox_inches="tight")





