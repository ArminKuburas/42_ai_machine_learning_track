import csv
import matplotlib.pyplot as plt

filename = '/Salary_dataset.csv'

years_experience = []
salaries = []

with open(filename, mode='r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        years_experience.append(float(row[1]))
        salaries.append(float(row[2]))

plt.figure(figsize=(7, 5))
plt.scatter(years_experience, salaries, color='blue', edgecolors='blue', linewidth=0.5)
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.grid(False)
plt.show()