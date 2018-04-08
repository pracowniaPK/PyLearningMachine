import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import time

# paramaters
lrate = 5
print("Learning rate: {}".format(lrate))
loops = 1000000
timeout = 60*5
acc_check = 10
np.set_printoptions(precision=4, suppress=True)

# sigmoid


def sigm(x):
    return 1/(1+np.exp(-x))


def sigm_d(x):
    return np.exp(-x)/np.power((np.exp(-x) + 1), 2)


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

# prepare variables
np.random.seed(1)
l1_width = 16
l2_width = 20
w2 = 2*np.random.random([784, l2_width])-1
w1 = 2*np.random.random([l2_width, l1_width])-1
w0 = 2*np.random.random([l1_width, 10])-1
n = len(tr_i)
print("Trainig array length: {}".format(n))
err_test = np.empty([len(test_i), 10])
err_list = []
acc_list_tr = []
acc_list_test = []
stopwatch = time.time()

# main loop
i = 0
stopwatch = time.time()
while time.time() - stopwatch < timeout and i < loops:
    # l3 = tr_i
    z2 = np.matmul(tr_i, w2)
    l2 = sigm(z2)
    z1 = np.matmul(l2, w1)
    l1 = sigm(z1)
    z0 = np.matmul(l1, w0)
    l0 = sigm(z0)

    err = l0 - tr_l

    """   nope -_-
    # c,t - temp cache for calculations
    # dn - changes in nth layer of weigths
    # w0
    c0 = err * sigm_d(z0)
    d0 = (-2*lrate/n) * l1.T.dot(c0)
    # w1
    c1 = np.empty([n, l1_width, 10])
    for j in range(10):
        c1[:, :, j] = sigm_d(z1) * w0[:, j]
    # t1 = cośtam cośtam w sumie 
    # l2.T.dot()
    d1 = np.zeros(np.shape(w1))
    for j in range(10):
        d1 += (-2*lrate/n) * l2.T.dot((c0[:, j] * c1[:, :, j].T).T)
    # w2
    """
    c0 = err * sigm_d(z0)
    d0 = (-2*lrate/n) * l1.T.dot(c0)
    err1 = c0.dot(w0.T)

    c1 = err1 * sigm_d(z1)
    d1 = (-2*lrate/n) * l2.T.dot(c1)
    err2 = c1.dot(w1.T)

    c2 = err2 * sigm_d(z2)
    d2 = (-2*lrate/n) * tr_i.T.dot(c2)

    # tests
    # TODO
    # actual tests with test set
    if i % acc_check == 0:
        jr = len(l0)
        kr = len(l0[0])
        ok = 0
        not_ok = 0
        for j in range(jr):
            max = -1000
            for k in range(kr):
                if max < l0[j, k]:
                    max = l0[j, k]
                    ans = k
            if tr_l[j][ans] == 1:
                ok += 1
            else:
                not_ok += 1
        acc_list_tr.append(ok/(ok+not_ok))
        err_list.append(np.mean(np.absolute(err)))
        print("{}th loop - acc: {:.2%} err: {:.2%}"
              .format(i, acc_list_tr[-1], err_list[-1]))

    # updating weights (applying -gradient)
    w0 += d0
    w1 += d1
    w2 += d2

    # end of main loop
    i += 1


print("{} learning loops @ {} learning rate, {:.2f} s, {:.2f} loops/s"
      .format(i, lrate, time.time() - stopwatch, i/(time.time() - stopwatch)))

plt.plot(err_list, label="err tr")
plt.plot(acc_list_tr, label="acc tr")
plt.legend()
plt.show()
