import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Make sure screenshots folder exists
os.makedirs("screenshots", exist_ok=True)

# Salary vs Experience
plt.scatter(data['experience'], data['salary'], color='blue')
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.savefig("screenshots/salary_vs_experience.png")
plt.close()

# Salary vs Education Level
sns.boxplot(x='education_level', y='salary', data=data)
plt.title("Salary vs Education Level")
plt.savefig("screenshots/salary_vs_education.png")
plt.close()

# Average Salary