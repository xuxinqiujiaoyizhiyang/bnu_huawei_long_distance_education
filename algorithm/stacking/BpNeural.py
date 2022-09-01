from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from BasicModel import BasicModel

class Bplassfier(BasicModel):
    def __init__(self):
        super().__init__()
    def train(self, x_train, y_train, x_val, y_val):
        clf = MLPClassifier(hidden_layer_sizes=(50, 100, 100, 30), activation='relu',
                            solver='adam', alpha=0.0001, batch_size=60, early_stopping=False, random_state=1,
                            learning_rate='constant', max_iter=1000).fit(x_train, y_train)
        RFC_auc = roc_auc_score(y_val, clf.predict(x_val), multi_class='ovo')
        return clf, RFC_auc

    def predict(self, model, x_test):
        print('test with Bp model')
        return model.predict(x_test)



