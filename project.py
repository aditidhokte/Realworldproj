import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset (example: diabetes.csv with columns like Glucose, BMI, Outcome)
df = pd.read_csv("data.csv").dropna().drop_duplicates()
print("Dataset Shape:", df.shape)
print(df.head())

# Features (X) and Target (y)
X = df[['Glucose', 'BMI']]   # choose relevant columns
y = df['Outcome']            # 0 = No diabetes, 1 = Diabetes

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualization
plt.scatter(X_test['Glucose'], y_test, color="blue", label="Actual")
plt.scatter(X_test['Glucose'], y_pred, color="red", marker="x", label="Predicted")
plt.xlabel("Glucose")
plt.ylabel("Outcome")
plt.legend()
plt.title("Diabetes Prediction (Actual vs Predicted)")
plt.show()
