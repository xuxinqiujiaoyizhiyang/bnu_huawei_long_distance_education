from sklearn.decomposition import PCA
from data2onehot_label import labeldata2onehot
import matplotlib.pyplot as plt

def pcadata(n, j):
    data = labeldata2onehot(j)
    pca = PCA(n_components=n)
    pca.fit(data)
    print("pca降维维度方差占比：", pca.explained_variance_ratio_)
    print("pca降维维度方差：", pca.explained_variance_)
    return pca.transform(data)

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# pca = PCA(n_components=0.9)
# pca.fit(data)
# print("pca降维维度方差占比：", pca.explained_variance_ratio_)
# print("pca降维维度方差：", pca.explained_variance_)


#  PCA可视化
# var = pca.explained_variance_ratio_[0: 28] #percentage of variance explained
# labels = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11', 'pc12', 'pc13', 'pc14', 'pc15', 'pc16', 'pc17', 'pc18', 'pc19', 'pc20', 'pc21', 'pc22', 'pc23', 'pc24', 'pc25', 'pc26', 'pc27', 'pc28']
#
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 20,
#          }
# plt.figure(figsize=(15,7))
# plt.xticks(fontproperties = 'Times New Roman', size = 12)
# plt.yticks(fontproperties = 'Times New Roman', size = 12)
# plt.bar(labels,var)
# plt.xlabel('Principal Component', font2)
# plt.ylabel('Proportion of Variance Explained', font2)
# plt.show()

