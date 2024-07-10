import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_dataset.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

sns.set_theme(
    style="whitegrid", 
    rc={
        'axes.facecolor': '#F0E8F5',  
        'grid.color': 'white',  
        'grid.linestyle': '-',  
        'grid.linewidth': 1, 
        'axes.edgecolor': '#F0E8F5',
        'scatter.marker': 'o',
        'scatter.edgecolors': 'none',
        'lines.linewidth': 1.5
    }
)

X = df[['YearsExperience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

joint_plot = sns.jointplot(x='YearsExperience', y='Salary', data=df, kind='scatter', marginal_ticks=False, color='#6484B9')

coef = model.coef_[0]
intercept = model.intercept_

x_line = pd.Series([X.min().values[0], X.max().values[0]])
y_line = coef * x_line + intercept
joint_plot.ax_joint.plot(x_line, y_line, color='#6484B9', linewidth=2)

joint_plot.set_axis_labels('Years of Experience', 'Salary', fontsize=12)
joint_plot.fig.tight_layout()
joint_plot.fig.suptitle('Linear Regression: Years of Experience vs Salary')
joint_plot.fig.subplots_adjust(top=0.92)

joint_plot.ax_marg_x.set_visible(False)
joint_plot.ax_marg_y.set_visible(False)

joint_plot.ax_joint.set_aspect('auto')
joint_plot.fig.set_figwidth(12)
joint_plot.fig.set_figheight(8)

joint_plot;