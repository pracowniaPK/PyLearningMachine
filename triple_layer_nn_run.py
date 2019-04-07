import idx2numpy
import numpy as np

from nnet import NNet

# prepare mnist data
tr_i_raw = idx2numpy.convert_from_file('mnist\\train-images.idx3-ubyte')
tr_l_raw = idx2numpy.convert_from_file('mnist\\train-labels.idx1-ubyte')
test_i_raw = idx2numpy.convert_from_file('mnist\\t10k-images.idx3-ubyte')
test_l_raw = idx2numpy.convert_from_file('mnist\\t10k-labels.idx1-ubyte')
tr_i = tr_i_raw.reshape([len(tr_i_raw), 784])/255
test_i = test_i_raw.reshape([len(test_i_raw), 784])/255
tr_l = np.full([len(tr_l_raw), 10], 0)
test_l = np.full([len(test_l_raw), 10], 0)
for i in range(len(tr_l_raw)):
    tr_l[i][tr_l_raw[i]] = 1
for i in range(len(test_l_raw)):
    test_l[i][test_l_raw[i]] = 1

nn = NNet(784, 10, layers=[16,20])
nn.fit(tr_i, tr_l, batch=32, timeout=60*0.5)
nn.plot_stats()
