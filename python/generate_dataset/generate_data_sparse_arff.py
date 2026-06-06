import numpy as np
from scipy.io import arff, savemat
import torch
import scipy.io
import scipy.sparse as sp

filename_train = '../data/eurlex-subject-matters/eurlex-sm-fold1-train.arff'
filename_test = '../data/eurlex-subject-matters/eurlex-sm-fold1-test.arff'
mat_filename = '../data/eurlex-subject-matters/finaldata.mat'
num_attributes = 5000
num_class = 201
# bibtex fea 1836 class 159 / enron fea 1001 class 53 / eurlex-dc fea 5000 class 412
# eurlex-ed fea 5000 class 3993 / eurlex-sm fea 5000 class 201 / medical fea 1449 class 45
# rcv1subset1 fea 47236 class 101 / rcv1subset2 fea 47236 class 101 / rcv1subset3 fea 47236 class 101
# rcv1subset4 fea 47229 class 101 / rcv1subset5 fea 47235 class 101 / tmc2007 fea 49060 class 22
for train_or_test in range(2):
    if train_or_test == 0:
        filename = filename_train
    else:
        filename = filename_test
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines[lines.index('@data\n')+1:]:
        line = line.strip()
        if line:
            line = line.strip('{}')
            feature_pairs = line.split(',')
            row = []
            for feature_pair in feature_pairs:
                feature_index, value = feature_pair.split()
                feature_index = int(feature_index)
                row.append((feature_index, float(value)))
            data.append(row)

    num_samples = len(data)
    num_features = num_attributes + num_class

    print(num_samples, num_features)

    rows = []
    cols = []
    values = []

    for i, row in enumerate(data):
        for feature_index, value in row:
            rows.append(i)
            cols.append(feature_index)
            values.append(value)

    X = sp.csr_matrix((values, (rows, cols)), shape=(num_samples, num_features))
    if train_or_test == 0:
        data_train = X.toarray()
        print(data_train.shape)
    else:
        data_test = X.toarray()
        print(data_test.shape)

Xapp = data_train[:, :num_attributes]
Yapp = data_train[:, num_attributes:]
Xgen = data_test[:, :num_attributes]
Ygen = data_test[:, num_attributes:]

print(Xapp.shape, Yapp.shape, Xgen.shape, Ygen.shape)

mat_dict = {
    'Xapp': Xapp,
    'Yapp': Yapp,
    'Xgen': Xgen,
    'Ygen': Ygen
}

savemat(mat_filename, mat_dict)
mat = scipy.io.loadmat('%s' % mat_filename)

Xapp = mat['Xapp']  # shape: (num_ins, num_fea)
Yapp = mat['Yapp']  # shape: (num_ins, num_class)
Xgen = mat['Xgen']
Ygen = mat['Ygen']

print(Xapp, Yapp)
