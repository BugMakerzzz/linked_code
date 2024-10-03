import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {
    'Methods': ['BM25', 'BM25', 'DPR', 'DPR', 'CoT', 'CoT', 
             'SC', 'SC', 'SR', 'SR', 'LtM', 'LtM', 'Ours', 'Ours'],
    'Type': ['ES', 'PS', 'ES', 'PS', 'ES', 'PS','ES', 'PS', 'ES', 'PS', 'ES', 'PS', 'ES', 'PS'],
    'Scores': [15.6, 90.1, 41.5, 85.6, 45.6, 79.0, 49.7, 81.0, 46.3, 67.7, 53.7, 77.1, 66.7, 87.8]
}
df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid",font='Times New Roman')
custom_colors = ['#E29957', '#86B5A1', '#B95A58', '#4292C6']
# ax = sns.barplot(x='Dataset', y='CCR', data=df, palette=sns.color_palette(custom_colors),width=0.5)
ax = sns.barplot(x='Methods', y='Scores', hue='Type', data=df, palette=sns.color_palette(custom_colors),width=0.5)
# ax2 = ax.twinx() 
 
ax.set_xlabel(ax.get_xlabel(), fontsize=24)  # X轴标签
ax.set_ylabel(ax.get_ylabel(), fontsize=24)  # Y轴标签
# ax.legend(fontsize='18', loc='upper left')  # 图例

ax.legend(fontsize='22', loc='upper left') 
# 调整刻度文字大小
ax.tick_params(axis='both', which='major', labelsize=22)

# 调整图像边距
plt.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.13)

plt.savefig('./human.pdf')
plt.close()