import numpy as np
from scipy.io import arff, savemat
import torch
import scipy.io

filename_train = '../data/birds/birds-train.arff'
filename_test = '../data/birds/birds-test.arff'
mat_filename = '../data/birds/finaldata.mat'
num_attributes = 260
# corel5k fea 499 class ? / bibtex fea 1836 class 159 / emotions fea 72 class 6
# mediamill fea 120 class 101 / scene fea 294 class 6 / tmc2007 fea ? class 22
# CAL500 fea 68 class 174 / birds fea 260 class 19

data_train, _ = arff.loadarff(filename_train)
data_test, _ = arff.loadarff(filename_test)

data_train = np.array(data_train)
data_train = np.array([[int(x.decode('utf-8')) if isinstance(x, bytes)
                        else x for x in row] for row in data_train])
data_train = torch.tensor(data_train, dtype=torch.float32)

data_test = np.array(data_test)
data_test = np.array([[int(y.decode('utf-8')) if isinstance(y, bytes)
                       else y for y in row] for row in data_test])
data_test = torch.tensor(data_test, dtype=torch.float32)
Xapp = data_train[:, :num_attributes]
Yapp = data_train[:, num_attributes:]
Xgen = data_test[:, :num_attributes]
Ygen = data_test[:, num_attributes:]

if isinstance(Xapp, torch.Tensor):
    Xapp = Xapp.numpy()
if isinstance(Yapp, torch.Tensor):
    Yapp = Yapp.numpy()
if isinstance(Xgen, torch.Tensor):
    Xgen = Xgen.numpy()
if isinstance(Ygen, torch.Tensor):
    Ygen = Ygen.numpy()

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
