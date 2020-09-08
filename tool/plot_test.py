# coding=UTF-8
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  

def plot1(x, y, title, path):
    matplotlib.rcParams['font.sans-serif']=['SimHei']
    matplotlib.rcParams['axes.unicode_minus']=False     
    
    plt.figure(figsize=(4, 7), dpi=80)
    plt.bar(x,y,width=0.2, color=['g','c'])
    for a, b in zip(x, y):
        plt.text(a, b+0.1, b, ha='center', va='bottom')
    plt.title(title)
    plt.savefig(path)
#     plt.show()

def plot2():
    x = ['IPU','GPU'] 
    y=[2503,116]
    
    plt.bar(x, y,  width=0.2, color=['g', 'darkorange'])
    for a, b in zip(x, y):
        plt.text(a, b+0.1, b, ha='center', va='bottom')
#     plt.figure(figsize=(5, 7), dpi=80)
    plt.xticks()
#     plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.ylabel('value')
#     plt.xlabel('line')
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['figure.figsize'] = (2.0, 7.0)  # 尺寸
    plt.title("Thoughput")
#     plt.savefig('D:\\result.png')
    plt.show()

def plot3():
    data = {'IPU':2503, "GPU":116}
    group_data = list(data.values())
    group_names = list(data.keys())
    fig, ax = plt.subplots()
    ax.barh(group_names, group_data)
    for a, b in zip(group_names, group_data):
        plt.text(a, b+0.1, b, ha='center', va='bottom')
    plt.show()
# plot3()
plot1(['IPU','GPU'], [2503,116], 'Throughput at lowest latency', '/Users/chendanxia/sophie/1.png')
plot1(['IPU','GPU'], [0.4,8.6], 'Latency at lowest latency', '/Users/chendanxia/sophie/2.png')
plot1(['IPU','GPU'], [157674,18275], 'Throughput for max throughput', '/Users/chendanxia/sophie/3.png')
plot1(['IPU','GPU'], [6.5,56], 'Latency for max throughput', '/Users/chendanxia/sophie/4.png')
