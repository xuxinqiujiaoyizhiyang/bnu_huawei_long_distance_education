import matplotlib.pyplot as plt


x = ['happy', 'surprised', 'normal', 'anger', 'digust', 'sad', 'fear']
y = [5.12995e-01, 2.76295e-06, 4.16388e-01, 3.07659e-04, 1.89905e-02, 5.10438e-02, 2.72497e-04]
plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16, weight='bold', rotation=-15)#设置大小及加粗
plt.rcParams['font.sans-serif'] = ['Dengxian']
plt.title("表情分布", fontweight="bold", size=20)

plt.bar(x, y)
plt.show()

