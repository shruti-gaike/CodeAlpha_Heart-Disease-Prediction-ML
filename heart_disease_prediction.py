import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report,
roc_auc_score
)

print("=" * 60)
print("HEART DISEASE PREDICTION USING MACHINE LEARNING")
print("=" * 60)

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

try:
    df = pd.read_csv(
        "heart.csv.data",
        header=None,
        names=columns,
        na_values="?"
    )

    print("Dataset loaded successfully!")
    print(df.head())

except FileNotFoundError:
    print("ERROR: processed.cleveland.data was not found.")
    raise SystemExit(1)

print("\nDataset loaded successfully!")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())



for column in columns:
  df[column] = pd.to_numeric(df[column], errors="coerce")



df = df.dropna(subset=["num"]).copy()



df["target"] = (df["num"] > 0).astype(int)



df = df.drop(columns=["num"])



df = df.drop_duplicates().reset_index(drop=True)

print("\nDataset cleaned successfully!")

print("\nCleaned dataset shape:")
print(df.shape)

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())



print("\nCreating visualizations...")



plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="target")

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target: 0 = No Disease, 1 = Disease")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()



plt.figure(figsize=(7, 4))

sns.histplot(data=df, x="age", bins=15, kde=True)

plt.title("Distribution of Patient Age")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 8))

sns.heatmap(
df.corr(numeric_only=True),
annot=True,
cmap="coolwarm",
fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()



X = df.drop(columns=["target"])
y = df["target"]

print("\nInput features:")
print(list(X.columns))

print("\nTarget variable: target")

# 7. Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. Create the machine learning models

# Missing numerical values are filled using the training-set median.

# Scaling is used for Logistic Regression.

# The Random Forest pipeline does not require feature scaling.

models = {
"Logistic Regression": Pipeline([
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler()),
("model", LogisticRegression(max_iter=2000))
]),

"Random Forest": Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

}

# 9. Train and evaluate the models

results = []
trained_models = {}

for name, model in models.items():


  print("\n" + "=" * 50)
print("Training:", name)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

auc = roc_auc_score(y_test, y_prob)

results.append({
    "Model": name,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": auc
})

trained_models[name] = model

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Disease", "Disease"],
        zero_division=0
    )
)

# Confusion matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title(f"Confusion Matrix - {name}")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()


# 10. Compare model performance

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.round(3).to_string(index=False))

# 11. Plot model accuracy

plt.figure(figsize=(7, 4))

sns.barplot(
data=results_df,
x="Model",
y="Accuracy"
)

plt.title("Accuracy Comparison of ML Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# 12. Select a model based on ROC-AUC

best_model_name = results_df.loc[
results_df["ROC-AUC"].idxmax(), "Model"
]

best_model = trained_models[best_model_name]

print("\nSelected model based on test ROC-AUC:")
print(best_model_name)

print("\nNote: Model selection and evaluation on the same test")
print("set can make the reported performance optimistic.")
print("For a formal study, use validation data or cross-validation.")

# 13. Predict a sample from the test dataset

sample = X_test.iloc[[0]]

actual_value = y_test.iloc[0]

prediction = best_model.predict(sample)[0]

probability = best_model.predict_proba(sample)[0, 1]

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

print("\nPatient input features:")
print(sample.to_string(index=False))

print("\nActual dataset class:", actual_value)
print("Predicted dataset class:", prediction)

print(
"Estimated model probability for positive class:",
f"{probability:.2%}"
)

if prediction == 1:
    print("\nModel output: Positive heart disease classification.")
else:
    print("\nModel output: Negative heart disease classification.")

print("\nProject execution completed successfully!")
print("This educational model is not a medical diagnostic tool.")