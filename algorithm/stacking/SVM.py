from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from BasicModel import BasicModel

class SVM(BasicModel):
    def __init__(self):
        super().__init__()
    def train(self, x_train, y_train, x_val, y_val):
        clf = SVC(kernel='rbf').fit(x_train, y_train)
        SVM_auc = roc_auc_score(y_val, clf.predict(x_val))
        return clf, SVM_auc

    def predict(self, model, x_test):
        print('test with SVM model')
        return model.predict(x_test)



