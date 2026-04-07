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
# Step 1: Load Excel File
# -------------------------------
data = pd.read_excel(
    r"C:\Users\Maddi Akshitha\Downloads\Employee Salary Prediction System\data\salary_data.xlsx",
    engine="openpyxl"
)

print("Dataset Preview:")
print(data.head())

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------
data = data.dropna()

# Encode categorical columns
le_edu = LabelEncoder()
le_skill = LabelEncoder()
le_loc = LabelEncoder()

data['education_level_enc'] = le_edu.fit_transform(data['education_level'])
data['skills_enc'] = le_skill.fit_transform(data['skills'])
data['location_enc'] = le_loc.fit_transform(data['location'])

# -------------------------------
# Step 3: Feature Selection
# -------------------------------
X = data[['experience', 'education_level_enc', 'skills_enc', 'location_enc']]
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
# Step 9: Visualizations
# -------------------------------

# 1. Salary vs Experience
plt.scatter(data['experience'], data['salary'], color='blue')
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.show()

# 2. Salary vs Education Level (Box Plot with original labels)
sns.boxplot(x='education_level', y='salary', data=data)
plt.title("Salary vs Education Level")
plt.show()

# 3. Average Salary by Skills (Bar Plot with original labels)
sns.barplot(x='skills', y='salary', data=data, estimator=np.mean)
plt.title("Average Salary by Skills")
plt.show()

# -------------------------------
# Step 10: User Input Prediction
# -------------------------------
print("\n--- Enter Details for Salary Prediction ---")

exp = float(input("Enter Experience (years): "))
edu = input("Enter Education (Bachelor/Master/PhD): ")
skill = input("Enter Skill (Python/Java/SQL/AI/Machine Learning/Data Science): ")
loc = input("Enter Location (Hyderabad/Bangalore/Chennai): ")

# Encode user inputs using fitted label encoders
edu_enc = le_edu.transform([edu])[0]
skill_enc = le_skill.transform([skill])[0]
loc_enc = le_loc.transform([loc])[0]

new_data = scaler.transform([[exp, edu_enc, skill_enc, loc_enc]])
predicted_salary = model.predict(new_data)

print("\nPredicted Salary:", predicted_salary[0])
