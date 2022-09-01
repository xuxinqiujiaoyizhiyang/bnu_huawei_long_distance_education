import pandas as pd
from PCA import pcadata
from sklearn.model_selection import train_test_split as ts
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from imblearn.combine import SMOTETomek
from xgboost import plot_tree
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from algorithm.single.data2onehot_label import labeldata2onehot
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import dtreeviz

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
dtrain = xgb.DMatrix(X, label=Y)
# X_train, X_test, Y_train, Y_test = ts(X_sample, Y, test_size=.3)
clf = XGBClassifier(
    n_estimators=25,
    learning_rate=0.2,
    max_depth=8,
    subsample=1,
    gamma=0,
    reg_lambda=1,
    max_delta_step=0,
    colsample_bytree=1,
    min_child_weight=1,
    seed=1000
)
params = {
    'booster':'gbtree',
    'objective':'multi:softmax',   # 多分类问题
    'num_class':13,  # 类别数，与multi softmax并用
    'gamma':0.1,    # 用于控制是否后剪枝的参数，越大越保守，一般0.1 0.2的样子
    'max_depth':8,  # 构建树的深度，越大越容易过拟合
    'lambda':1,  # 控制模型复杂度的权重值的L2 正则化项参数，参数越大，模型越不容易过拟合
    # 'colsample_bytree':3,# 这个参数默认为1，是每个叶子里面h的和至少是多少
    # 对于正负样本不均衡时的0-1分类而言，假设h在0.01附近，min_child_weight为1
    #意味着叶子节点中最少需要包含100个样本。这个参数非常影响结果，
    # 控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    'silent':0,  # 设置成1 则没有运行信息输入，最好是设置成0
    'eta':0.007,  # 如同学习率
    'seed':1000,
    'eval_metric':'auc'
}
# evallist = []
# clf = xgb.train(params, dtrain, 100, evallist)
evalset = [(X, Y), (X_test,Y_test)]
# fit the model
clf.fit(X, Y, eval_metric='mlogloss', eval_set=evalset)
# evaluate performance
Y_test, y_pred = Y_test, clf.predict(X_test)
score = accuracy_score(Y_test, y_pred)
print('Accuracy: %.3f' % score)
# retrieve performance metrics
results = clf.evals_result()
# plot learning curves
plt.plot(results['validation_0']['mlogloss'], label='train')
plt.plot(results['validation_1']['mlogloss'], label='test')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
plt.xlabel("Iterations", font2)
plt.ylabel("Loss", font2)

# show the legend
plt.legend()
# show the plot
plt.savefig('./iteration.jpg')
plt.show()
clf.save_model('xgb2.model')
# clf.save_model('xgb1.model')
# dtest = xgb.DMatrix(X_test)
print('y_pred', y_pred)
# print(mean_squared_error(Y_test, y_pred))

plot_tree(clf)
# viz = dtreeviz(clf)
# viz.view()

print('score', accuracy_score(Y_test, y_pred))
print("f1", f1_score(Y_test, y_pred, labels=None,
               pos_label=1, average='micro', sample_weight=None,
               zero_division='warn'))

# print(f1)
# print(score)
# def Get_Average(list)=
#     sum = 0
#     for item in list=
#         sum += item
#     return sum/len(list)
# print("f1=", Get_Average(f1))
# print("score=", Get_Average(score))
# plt.subplot(2, 1, 1)
# plt.scatter(list1, f1)
# plt.plot(list1, f1, linestyle='solid', color='r')
# plt.title("Xgboost F1值")
#
# for i in range(len(f1))=
#     plt.annotate(str(round(f1[i], 3)), xy=(i, f1[i]), xytext=(i + 0.05, f1[i] - 0.002))
#
# plt.subplot(2, 1, 2)
# plt.scatter(list1, score)
# plt.plot(list1, score, linestyle='solid', color='g')
#
# for i in range(len(score))=
#     plt.annotate(str(round(score[i], 3)), xy=(i, score[i]), xytext=(i + 0.05, score[i] - 0.002))
# plt.title("Xgboost Accuracy")
# plt.show()



