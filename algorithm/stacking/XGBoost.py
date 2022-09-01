from xgboost import XGBClassifier as xgb
from sklearn.metrics import roc_auc_score
from BasicModel import BasicModel

class XGBClassfier(BasicModel):
    def __init__(self):
        super().__init__()
    def train(self, x_train, y_train, x_val, y_val):
        clf = xgb(
            n_estimators=2000,
            learning_rate=0.5,
            max_depth=10,
            subsample=1,
            gamma=0,
            reg_lambda=1,
            max_delta_step=0,
            colsample_bytree=1,
            min_child_weight=1,
            seed=1000
        ).fit(x_train, y_train)
        Xgbc_auc = roc_auc_score(y_val, clf.predict(x_val), multi_class='ovo')
        return clf, Xgbc_auc

    def predict(self, model, x_test):
        print('test with xgb model')
        return model.predict(x_test)



