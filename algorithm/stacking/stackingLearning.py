from RandomForest import RFClassfier
from XGBoost import XGBClassfier
import pandas as pd
import numpy as np
from BpNeural import Bplassfier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# 通过输入一组数据分别进入18个分类器中进行预测，得到最终的药物推荐清单
from algorithm.single.data2onehot_label import labeldata2onehot

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
X_test = labeldata2onehot('80')
Y_test = pd.read_csv("../test/top_3_test_final.csv")['class'].values.T
print('x', X_test.shape)
print('y', Y_test.shape)

X = labeldata2onehot('1')
Y = pd.read_csv("../test/top_3_final.csv")

Y = Y['class'].values.T


rf_classifier = RFClassfier()
rf_oof_train, rf_oof_test = rf_classifier.get_oof(x_train=X, y_train=Y, x_test=X_test)

xgb_classifier = XGBClassfier()
xgb_oof_train, xgb_oof_test = xgb_classifier.get_oof(X, Y, X_test)

svm_classifier = Bplassfier()
svm_oof_train, svm_oof_test = svm_classifier.get_oof(X, Y, X_test)

input_train = [xgb_oof_train, xgb_oof_train, svm_oof_train]
input_test = [xgb_oof_test, xgb_oof_test, svm_oof_test]
stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
final_model = LogisticRegression()
# final_model = RandomForestClassifier(n_estimators=100, bootstrap=True)
final_model.fit(stacked_train, Y)
test_score = final_model.score(stacked_test, Y)
test_prediction = final_model.predict(stacked_test)

print("准确率：", accuracy_score(Y_test, test_prediction))
print("f1值：", f1_score(Y_test, test_prediction, labels=None,
               pos_label=1, average='binary', sample_weight=None,
               zero_division='warn'))
# score.append(accuracy_score(y_test, test_prediction))
# f1.append(f1_score(y_test, test_prediction, labels=None,
#                    pos_label=1, average='binary', sample_weight=None,
#                    zero_division='warn'))


# def Get_Average(list):
#     sum = 0
#     for item in list:
#         sum += item
#     return sum / len(list)
#
#
# print("f1:", Get_Average(f1))
# print("score:", Get_Average(score))
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 10,
#          }
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }
# plt.subplot(2, 1, 2)
# plt.xticks(fontproperties='Times New Roman', size=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.scatter(list1, f1)
# plt.axhline(round(Get_Average(f1), 3), color='b', linestyle='--', label="Average F1:" + str(round(Get_Average(f1), 3)))
# plt.plot(list1, f1, linestyle='solid', color='r')
# plt.legend()
# plt.title("StackingLearning Model F1", font2)
#
# for i in range(len(f1)):
#     plt.annotate(str(round(f1[i], 3)), xy=(i, f1[i]), xytext=(i + 0.05, f1[i] - 0.002))
#
# plt.subplot(2, 1, 1)
# plt.xticks(fontproperties='Times New Roman', size=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.axhline(round(Get_Average(score), 3), color='b', linestyle='--',
#             label="Average Accuracy:" + str(round(Get_Average(score), 3)))
# plt.scatter(list1, score)
# plt.plot(list1, score, linestyle='solid', color='g')
#
# for i in range(len(score)):
#     plt.annotate(str(round(score[i], 3)), xy=(i, score[i]), xytext=(i + 0.05, score[i] - 0.002))
# plt.title("StackingLearning Model Accuracy", font2)
# plt.legend()
# plt.show()
