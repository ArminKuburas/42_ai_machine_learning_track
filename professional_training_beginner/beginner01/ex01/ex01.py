import csv
import numpy as np
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

X = np.array(years_experience)
y = np.array(salaries)

beta_0 = 0
beta_1 = 0

learning_rate = 0.01
num_iterations = 10

def compute_cost(X, y, beta_0, beta_1):
    n = len(y)
    y_pred = beta_0 + beta_1 * X
    cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    return cost

def gradient_descent(X, y, beta_0, beta_1, learning_rate, num_iterations):
    n = len(y)
    cost_history = []

    for i in range(num_iterations):
        y_pred = beta_0 + beta_1 * X
        d_beta_0 = (1 / n) * np.sum(y_pred - y)
        d_beta_1 = (1 / n) * np.sum((y_pred - y) * X)
        beta_0 -= learning_rate * d_beta_0
        beta_1 -= learning_rate * d_beta_1
        cost = compute_cost(X, y, beta_0, beta_1)
        cost_history.append(cost)
        if i % 1 == 0:
            plt.figure(figsize=(12, 8))
            plt.scatter(X, y, color='purple', edgecolors='white', linewidth=0.5, label='Data points')
            plt.plot(X, y_pred, color='blue', label=f'Iteration {i}')
            plt.xlabel('Years of Experience')
            plt.ylabel('Salary')
            plt.title('Scatter Plot with Linear Regression Line')
            plt.grid(True)
            plt.legend()
            plt.show()

    return beta_0, beta_1, cost_history

beta_0, beta_1, cost_history = gradient_descent(X, y, beta_0, beta_1, learning_rate, num_iterations)

plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='purple', edgecolors='white', linewidth=0.5, label='Data points')
plt.plot(X, beta_0 + beta_1 * X, color='blue', label='Final Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Scatter Plot with Final Linear Regression Line')
plt.grid(True)
plt.legend()
plt.show()