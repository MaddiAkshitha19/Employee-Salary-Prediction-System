import os
os.makedirs("screenshots", exist_ok=True)

plt.scatter(data['experience'], data['salary'], color='blue')
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.savefig("screenshots/salary_vs_experience.png")
plt.show()

