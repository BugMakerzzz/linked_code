import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_topk():
    # 数据
    topk = [1, 2, 3, 4, 5]
    winogrande_acc = [79.0, 81.6, 79.6, 79.2, 78.4]
    hellaswag_acc = [71.0, 69.8, 69.8, 67.4, 68.4]

    # 创建数据框
    data_winogrande = pd.DataFrame({'Top k': topk, 'Accuracy': winogrande_acc, 'Dataset': 'Winogrande'})
    data_hellaswag = pd.DataFrame({'Top k': topk, 'Accuracy': hellaswag_acc, 'Dataset': 'HellaSwag'})
    
    # 设置风格
    
    sns.set(font_scale=2)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    # 第一个子图 - Winogrande
    plt.autoscale(False)
    # 第一个条形图 - Winogrande
    plt.subplot(1, 2, 1)
    
    colors = sns.color_palette("Blues", n_colors=len(topk))
    color_values_winogrande = sorted(winogrande_acc)
    color_palette_winogrande = [colors[color_values_winogrande.index(val)] for val in winogrande_acc]
    
    
    sns.barplot(x="Top k", y="Accuracy", data=data_winogrande,palette=color_palette_winogrande)
    plt.xlabel("Top k")
    plt.ylabel("Accuracy")
    plt.title("Winogrande")
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=18, color='black', xytext=(0, 5), textcoords='offset points')
   
    # 减小纵轴刻度值
    plt.ylim(75, 85)

    # 第二个条形图 - HellaSwag
    plt.subplot(1, 2, 2)
    
    color_values_hellaswag = sorted(hellaswag_acc)
    color_palette_hellaswag = [colors[color_values_hellaswag.index(val)] for val in hellaswag_acc]
    
    sns.barplot(x="Top k", y="Accuracy", data=data_hellaswag, palette=color_palette_hellaswag)
    plt.xlabel("Top k")
    plt.ylabel("Accuracy")
    plt.title("Hellaswag")
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=18, color='black', xytext=(0, 5), textcoords='offset points')
   
    # 减小纵轴刻度值
    plt.ylim(65, 75)
    # 调整子图之间的间距
    plt.tight_layout()



    plt.savefig("topk_plot.pdf", format="pdf")
    # 显示图表
    plt.show()
    
def draw_samplecnt():

    # 数据
    sample_counts = [3, 5, 7, 10]
    winogrande_acc = [79.4, 81.6, 78.6, 81.6]
    hellaswag_acc = [68.6, 71.0, 67.4, 68.4]

    data_winogrande = pd.DataFrame({'Sample Counts': sample_counts, 'Accuracy': winogrande_acc, 'Dataset': 'Winogrande'})
    data_hellaswag = pd.DataFrame({'Sample Counts': sample_counts, 'Accuracy': hellaswag_acc, 'Dataset': 'HellaSwag'})

    # 设置Seaborn样式
    # sns.set(style="whitegrid")
    sns.set(font_scale=2)
    
    sns.set_style("whitegrid")
    # 自定义Seaborn主题

    # 创建两个条形图
    plt.figure(figsize=(12, 6))
    plt.autoscale(False)
    # 第一个条形图 - Winogrande
    plt.subplot(1, 2, 1)
    
    colors = sns.color_palette("Blues", n_colors=len(sample_counts))
    color_values_winogrande = sorted(winogrande_acc)
    color_palette_winogrande = [colors[color_values_winogrande.index(val)] for val in winogrande_acc]
    
    sns.barplot(x="Sample Counts", y="Accuracy", data=data_winogrande, palette=color_palette_winogrande)
    plt.xlabel("Sample Counts")
    plt.ylabel("Accuracy")
    plt.title("Winogrande")
    # 减小纵轴刻度值
    plt.ylim(75, 85)
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=18, color='black', xytext=(0, 5), textcoords='offset points')

    # 第二个条形图 - HellaSwag
    plt.subplot(1, 2, 2)
    
    color_values_hellaswag = sorted(hellaswag_acc)
    color_palette_hellaswag = [colors[color_values_hellaswag.index(val)] for val in hellaswag_acc]

    sns.barplot(x="Sample Counts", y="Accuracy", data=data_hellaswag, palette=color_palette_hellaswag)
    plt.xlabel("Sample Counts")
    plt.ylabel("Accuracy")
    plt.title("HellaSwag")
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=18, color='black', xytext=(0, 5), textcoords='offset points')
    # 减小纵轴刻度值
    plt.ylim(65, 75)
    # 调整子图之间的间距
    plt.tight_layout()

    # 添加图例
    plt.savefig("samplecnt_plot.pdf", format="pdf")
    # 显示图表
    plt.show()
    
def draw_robust():

    methods = ['Ours', 'Few-shot', 'CoT-SC']
    max_values = [82.6, 73.2, 73.4]
    avg_values = [79.9, 71.3, 71.8]
    min_values = [77.2, 70.6, 70.0]

    # 设置字体大小
    plt.rcParams.update({'font.size': 16})

    # 条形图的宽度
    bar_width = 0.2
    index = np.arange(len(methods))

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制最大值纵向柱状图，并标注数据
    bars1 = plt.bar(index, max_values, bar_width, label='Max', color='lightblue', edgecolor='black')
    for bar, value in zip(bars1, max_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{value:.1f}', ha='center')

    # 绘制平均值纵向柱状图，并标注数据
    bars2 = plt.bar(index + bar_width, avg_values, bar_width, label='Avg', color='lightgreen', edgecolor='black')
    for bar, value in zip(bars2, avg_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{value:.1f}', ha='center')

    # 绘制最小值纵向柱状图，并标注数据
    bars3 = plt.bar(index + 2 * bar_width, min_values, bar_width, label='Min', color='lightcoral', edgecolor='black')
    for bar, value in zip(bars3, min_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{value:.1f}', ha='center')

    # 设置横轴标签
    plt.xlabel('Methods')

    # 设置纵轴标签
    plt.ylabel('Accuracy')

    # 设置标题
    # plt.title('Accuracy Comparison')

    # 设置横轴刻度
    plt.xticks(index + bar_width, methods)
    plt.legend()
    plt.legend(loc='lower left')


    # 显示图形
    plt.tight_layout()

    plt.savefig("robust_plot.pdf", format="pdf")
    # plt.savefig("robust_plot.png", format="png")
    # 显示图形
    plt.show()
    
draw_topk()
draw_samplecnt()
draw_robust()