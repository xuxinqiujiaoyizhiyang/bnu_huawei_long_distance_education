from numpy import array
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
def labeldata2onehot(j):

    if j == '80':
        medical = pd.read_csv('../test/top_3_test_final.csv', low_memory=False)
    else:
        medical = pd.read_csv('../test/top_3_final.csv', low_memory=False)
    dic = {}
    data = []
    for index, row in medical.iterrows():
        data.append(row['Body'])
    dic['data'] = data
    data = []
    for index, row in medical.iterrows():
        data.append(row['Head'])
    dic['data0'] = data
    data = []
    for index, row in medical.iterrows():
        data.append(row['Interaction'])
    dic['data1'] = data


    list1 = np.arange(0, 80, 1)

    onehot_encoded = np.array([])
    for i in np.arange(-1, 2, 1):
        if i == -1:
            values = array(dic['data'])

            # integer encode
            label_encoder = LabelEncoder()
            label_encoder.fit(list1)
            integer_encoded = label_encoder.transform(values)
            # binary encode
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded_fit = list1.reshape(len(list1), 1)
            integer_encoded_transfomer = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoder = onehot_encoder.fit(integer_encoded_fit)
            onehot_encoded = onehot_encoder.transform(integer_encoded_transfomer)
        else:
            values = array(dic['data'+str(i)])

            # # integer encode
            label_encoder = LabelEncoder()
            label_encoder.fit(list1)
            integer_encoded = label_encoder.transform(values)
            # binary encode
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded_fit1 = list1.reshape(len(list1), 1)
            integer_encoded_transfomer1 = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoder = onehot_encoder.fit(integer_encoded_fit1)
            onehot_encoded1 = onehot_encoder.transform(integer_encoded_transfomer1)
            onehot_encoded = np.append(onehot_encoded, onehot_encoded1, axis=1)
    return onehot_encoded
