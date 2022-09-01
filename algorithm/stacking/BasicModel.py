import numpy as np
from sklearn.model_selection import KFold


class BasicModel(object):

    def train(self, x_train, y_train, x_val, y_val):
        pass

    def predict(self, model, x_test):
        pass

    def get_oof(self, x_train, y_train, x_test, n_folds=7):
        """K-fold stacking"""
        num_train, num_test = x_train.shape[0], x_test.shape[0]
        oof_train = np.zeros((num_train,))
        oof_test = np.zeros((num_test,))
        oof_test_all_fold = np.zeros((num_test, n_folds))
        aucs = []
        KF = KFold(n_splits=n_folds, random_state=2017, shuffle=True)
        for i, (train_index, val_index) in enumerate(KF.split(x_train)):
            print('{0} fold, train {1}, val {2}'.format(i,
                                                        len(train_index),
                                                        len(val_index)))
            x_tra, y_tra = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]
            model, auc = self.train(x_tra, y_tra, x_val, y_val)
            aucs.append(auc)
            oof_train[val_index] = self.predict(model, x_val)
            oof_test_all_fold[:, i] = self.predict(model, x_test)
        oof_test = np.mean(oof_test_all_fold, axis=1)
        print('all aucs {0}, average {1}'.format(aucs, np.mean(aucs)))
        return oof_train, oof_test