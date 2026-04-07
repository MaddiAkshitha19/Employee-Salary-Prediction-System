# Employee Salary Prediction System

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Step 1: Load Excel File (FIXED)
# -------------------------------
data = pd.read_excel(r"C:\Users\Maddi Akshitha\Downloads\Employee Salary Prediction System\data\salary_data.xlsx", engine="openpyxl")

print("Dataset Preview:")
print(data.head())

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------
data = data.dropna()

# Encode categorical columns
le = LabelEncoder()

data['education_level'] = le.fit_transform(data['education_level'])
data['skills'] = le.fit_transform(data['skills'])
data['location'] = le.fit_transform(data['location'])

# -------------------------------
# Step 3: Feature Selection
# -------------------------------
X = data[['experience', 'education_level', 'skills', 'location']]
y = data['salary']

# -------------------------------
# Step 4: Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 6: Model Training
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 7: Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 8: Model Evaluation
# -------------------------------
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# Step 9: Visualization
# -------------------------------
plt.scatter(data['experience'], data['salary'])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

# 2. Salary vs Education Level
sns.boxplot(x='education_level', y='salary', data=data)
plt.title("Salary vs Education Level")
plt.show()

# 3. Average Salary by Skills
sns.barplot(x='skills', y='salary', data=data, estimator=np.mean)
plt.title("Average Salary by Skills")

# -------------------------------
# Step 10: User Input Prediction
# -------------------------------
print("\n--- Enter Details for Salary Prediction ---")

exp = float(input("Enter Experience (years): "))
edu = int(input("Enter Education (0=Bachelor,1=Master,2=PhD): "))
skill = int(input("Enter Skill (encoded number): "))
loc = int(input("Enter Location (encoded number): "))

new_data = scaler.transform([[exp, edu, skill, loc]])
predicted_salary = model.predict(new_data)

print("\nPredicted Salary:", predicted_salary[0])