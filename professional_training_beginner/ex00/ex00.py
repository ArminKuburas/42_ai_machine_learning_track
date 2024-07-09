import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

joint_plot = sns.jointplot(x='YearsExperience', y='Salary', data=df, kind='scatter', marginal_ticks=False, color='#9B90C2')

joint_plot.set_axis_labels('YearsExperience', 'Salary', fontsize=12)

joint_plot.fig.tight_layout()

for spine in joint_plot.ax_joint.spines.values():
    spine.set_alpha(1.0)

joint_plot.fig.suptitle('')
joint_plot.fig.subplots_adjust(top=0.92)

joint_plot;