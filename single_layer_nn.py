import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import time

# parameters
lrate = 1  # float(input("leraning rate:\n"))  # learning rate
lrate2 = 3
loops = 3000000
timeout = 60*60
acc_check = 100
# batch = 0  # batch size
np.set_printoptions(precision=4, suppress=True)

# define sigmoid


def sigm(x):
    return 1/(1+np.exp(-x))


def sigm_d(x):
    return np.exp(-x)/np.power((np.exp(-x) + 1), 2)
    # return (4-(x*x))*0.0625


def lrateD(t, n):
    return (t*lrate + (n-t) * lrate2)/n


# prepare mnist datasest
# mnist taken from: http://yann.lecun.com/exdb/mnist/
tr_i_raw = idx2numpy.convert_from_file('mnist\\train-images.idx3-ubyte')
tr_l_raw = idx2numpy.convert_from_file('mnist\\train-labels.idx1-ubyte')
test_i_raw = idx2numpy.convert_from_file('mnist\\t10k-images.idx3-ubyte')
test_l_raw = idx2numpy.convert_from_file('mnist\\t10k-labels.idx1-ubyte')

print("ladad mnist")

# preparing images
tr_i = tr_i_raw.reshape([len(tr_i_raw), 784])/255
test_i = test_i_raw.reshape([len(test_i_raw), 784])/255
# tr_i = np.empty([len(tr_i_raw), 784])
# test_i = np.empty([len(test_i_raw), 784])
# for i in range(len(tr_i_raw)):
#     tr_i[i] = tr_i_raw[i].flatten()/255
# for i in range(len(test_i_raw)):
#     test_i[i] = test_i_raw[i].flatten()/255

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

# preparing variables
n = len(tr_i)
print("Trainig array length: {}".format(n))
# tr_i = tr_i[:n]
# tr_l = tr_l[:n]
err = np.empty([len(tr_i), 10])
err_test = np.empty([len(test_i), 10])
err_list = []
acc_list_tr = []
acc_list_test = []
stopwatch = time.time()
# values_list = np.empty([loops, n, 10])

# lets get this party started
# for i in range(loops):
i = 0
while time.time() - stopwatch < timeout and i < loops:
    z = np.matmul(tr_i, w)
    z_s = sigm(z)
    err = z_s - tr_l
    # delta = (-2) * lrate * tr_i.T.dot(err * z_s) / 784
    # delta = (-2) * lrate * tr_i.T.dot(err * sigm_d(z) / n)
    delta = (-2) * lrateD(i, n) * tr_i.T.dot(err * sigm_d(z) / n)
    # tests
    if(i % acc_check == 0):
        # "test" on training data
        ok = 0
        not_ok = 0
        for j in range(len(z)):
            ans = 0
            max = -1000
            for k in range(len(z[j])):
                if max < z[j, k]:
                    max = z[j, k]
                    ans = k
            if tr_l[j][ans] == 1:
                ok += 1
            else:
                not_ok += 1
        acc_list_tr.append(ok/(ok+not_ok))
        # calculation on test set
        z = np.matmul(test_i, w)
        z = sigm(z)
        err_test = sigm(z) - test_l
        ok = 0
        not_ok = 0
        for j in range(len(z)):
            ans = 0
            max = -1000
            for k in range(len(z[j])):
                if max < z[j, k]:
                    max = z[j, k]
                    ans = k
            if test_l[j][ans] == 1:
                ok += 1
            else:
                not_ok += 1
        acc_list_test.append(ok/(ok+not_ok))
        err_list.append(np.mean(np.absolute(err)))
        print(str(i) + "th loop - acc: " +
              '{0:.2%}'.format(acc_list_test[-1]) + " err: " + '{0:.2%}'.format(err_list[-1]))
    # values_list[i] = z_s
    # apply -gradient
    w += delta
    i += 1
    # end of main loop

print(str(i) + " learning loops @ " + str(lrate) +
      " lrate: " + '{0:.2f}'.format(time.time() - stopwatch) +
      "s, {0:.2f} loops/s".format(i/(time.time() - stopwatch)))
plt.plot(err_list, label="err tr")
plt.plot(acc_list_tr, label="acc tr")
plt.plot(acc_list_test, label="acc test")
plt.legend()
plt.show()


# def value(x):
#     for j in range(10):
#         plt.plot(values_list[:, x, j], label=str(j))
#     plt.legend()
#     print(str(tr_l_raw[x]))
#     plt.show()

# plt.plot(err, 'ro')
# plt.show()
# plt.imshow(tr_i_raw[1])
# plt.show()
