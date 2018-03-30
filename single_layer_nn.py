import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

# parameters
lrate = 0.3  # learning rate
# batch = 0  # batch size
np.set_printoptions(precision=4, suppress=True)

# define sigmoid


def sigm(x):
    return 1/(1+np.exp(-x))


def sigm_d(x):
    return np.exp(-x)/np.power((np.exp(-x) + 1), 2)


# prepare mnist datasest
# mnist taken from: http://yann.lecun.com/exdb/mnist/
tr_i_raw = idx2numpy.convert_from_file('mnist\\train-images.idx3-ubyte')
tr_l_raw = idx2numpy.convert_from_file('mnist\\train-labels.idx1-ubyte')
test_i_raw = idx2numpy.convert_from_file('mnist\\t10k-images.idx3-ubyte')
test_l_raw = idx2numpy.convert_from_file('mnist\\t10k-labels.idx1-ubyte')

print("ladad mnist")

# preparing images
tr_i = np.empty([len(tr_i_raw), 784])
test_i = np.empty([len(test_i_raw), 784])
for i in range(len(tr_i_raw)):
    tr_i[i] = tr_i_raw[i].flatten()/255
for i in range(len(test_i_raw)):
    test_i[i] = test_i_raw[i].flatten()/255

# preparing labels
tr_l = np.full([len(tr_l_raw), 10], 0)
test_l = np.full([len(test_l_raw), 10], 0)
for i in range(len(tr_l_raw)):
    tr_l[i][tr_l_raw[i]] = 1
for i in range(len(test_l_raw)):
    test_l[i][test_l_raw[i]] = 1

print("data converted to [1x784] and [1x10] arrays")

# generating initial weights
np.random.seed(1)
w = 2*np.random.random([784, 10])-1

# lets get this party started

err = np.empty([len(test_i), 10])
err2 = np.empty([len(test_i), 10])
err_list = []
for i in range(100):
    z = np.matmul(test_i, w)
    err = sigm(z) - test_l
    err2 = np.absolute(sigm(z) - test_l)
    delta = (-2) * lrate * test_i.T.dot(err * sigm_d(z)) / 784
    w += delta
    err_list.append(np.mean(err2))
    print(np.mean(err2))

# TODO
# zamienić test na tr
# osobne testy
# sprawdzaczka poprawności odpowiedzi

plt.plot(err_list)
plt.show()

# plt.plot(err, 'ro')
# plt.show()
# t = np.arange(-3, 3, .05)
# plt.plot(t, sigm(t), t, sigm_d(t))
# plt.show()
# plt.imshow(tr_i_raw[1])
# plt.show()
