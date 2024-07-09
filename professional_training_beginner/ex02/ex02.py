import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings

df = pd.read_csv('salary_dataset.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

X = df[['YearsExperience']]
y = df['Salary']

X.columns = ['YearsExperience']

model = LinearRegression()

warnings.filterwarnings('ignore', message="X does not have valid feature names")

model.fit(X, y)

years_exp = float(input("Enter years of experience: "))

predicted_salary = model.predict([[years_exp]])

formatted_salary = f"{predicted_salary[0]:.0f}" if predicted_salary[0].is_integer() else f"{predicted_salary[0]}"
formatted_years = f"{years_exp:.0f}" if years_exp.is_integer() else f"{years_exp}"

print(f"Predicted salary for {formatted_years} years of experience : {formatted_salary}")