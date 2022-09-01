from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from BasicModel import BasicModel

class RFClassfier(BasicModel):
    def __init__(self):
        super().__init__()
    def train(self, x_train, y_train, x_val, y_val):
        clf = RandomForestClassifier(n_estimators=100, bootstrap=True).fit(x_train, y_train)
        RFC_auc = roc_auc_score(y_val, clf.predict(x_val), multi_class='ovo')
        return clf, RFC_auc

    def predict(self, model, x_test):
        print('test with RFC model')
        return model.predict(x_test)



