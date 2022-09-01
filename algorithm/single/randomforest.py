import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from PCA import pcadata
from sklearn.model_selection import train_test_split as ts
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.combine import SMOTETomek

from algorithm.single.data2onehot_label import labeldata2onehot

list1 = ['02', '03', '05', '06', '07', '08', '09', '912', '012', '23', '25', '26', '27', '28', '29', '35', '36', '37',
         '38', '39', '56', '57', '58', '59', '67', '68', '69', '78', '79', '89', '212', '312', '512', '612', '712', '812']
re_list = []
re_list1 = []
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
X_test = labeldata2onehot('80')
Y_test = pd.read_csv("../test/top_3_test_final.csv")['class'].values.T
print('x', X_test.shape)
print('y', Y_test.shape)

X = labeldata2onehot('1')
Y = pd.read_csv("../test/top_3_final.csv")

Y = Y['class'].values.T
print(Y)
ros = SMOTETomek(random_state=0)
# X_sample, Y = ros.fit_resample(X, Y)
print(X.shape)
print(Y.shape)
# X_train, X_test, Y_train, Y_test = ts(X_sample, Y, test_size=.3)
clf = RandomForestClassifier(n_estimators=500, random_state=0, bootstrap=False).fit(X, Y)
Y_test, y_pred = Y_test, clf.predict(X_test)
print('y_pred', y_pred)

print('score', accuracy_score(Y_test, y_pred))
print("f1", f1_score(Y_test, y_pred, labels=None,
               pos_label=1, average='micro', sample_weight=None,
               zero_division='warn'))

# print(f1)
# print(score)
# def Get_Average(list):
#     sum = 0
#     for item in list:
#         sum += item
#     return sum/len(list)
# print("f1:", Get_Average(f1))
# print("score:", Get_Average(score))
# plt.subplot(2, 1, 1)
# plt.scatter(list1, f1)
# plt.plot(list1, f1, linestyle='solid', color='r')
# plt.title("Xgboost F1值")
#
# for i in range(len(f1)):
#     plt.annotate(str(round(f1[i], 3)), xy=(i, f1[i]), xytext=(i + 0.05, f1[i] - 0.002))
#
# plt.subplot(2, 1, 2)
# plt.scatter(list1, score)
# plt.plot(list1, score, linestyle='solid', color='g')
#
# for i in range(len(score)):
#     plt.annotate(str(round(score[i], 3)), xy=(i, score[i]), xytext=(i + 0.05, score[i] - 0.002))
# plt.title("Xgboost Accuracy")
# plt.show()



