import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


class NNet:

    def __init__(self, tr_i, test_i, tr_l, test_l, l1_width=16, l2_width=20, random_seed=1):
        self.tr_i = tr_i
        self.test_i = test_i
        self.tr_l = tr_l
        self.test_l = test_l
        self.n = len(tr_i)
        m = len(test_i)
        print("nodes in layer: input {}, output {}".format(self.n, m))
        if self.n != len(self.tr_l) or m != len(test_l):
            raise Exception("lengths of training and test sets dont't match")

        np.random.seed(random_seed)
        self.w2 = 2*np.random.random([len(tr_i[0]), l2_width])-1
        self.w1 = 2*np.random.random([l2_width, l1_width])-1
        self.w0 = 2*np.random.random([l1_width, len(tr_l[0])])-1

        self.err_list = []
        self.acc_list_tr = []
        self.acc_list_test = []
        self.total_time = 0
        self.total_loops = 0

    def sigm(self, x):
        return 1/(1+np.exp(-x))

    def sigm_d(self, x):
        return np.exp(-x)/np.power((np.exp(-x) + 1), 2)

    def spin(self, lrate, loops=sys.maxsize, timeout=sys.maxsize, acc_check=10, verbose=True):

        i = 0
        stopwatch = time.time()

        while time.time() - stopwatch < timeout and i < loops:
            # spinning the network:
            # l3 = tr_i
            z2 = np.matmul(self.tr_i, self.w2)
            l2 = self.sigm(z2)
            z1 = np.matmul(l2, self.w1)
            l1 = self.sigm(z1)
            z0 = np.matmul(l1, self.w0)
            l0 = self.sigm(z0)

            err0 = l0 - self.tr_l

            c0 = err0 * self.sigm_d(z0)
            d0 = (-2*lrate/self.n) * l1.T.dot(c0)
            err1 = c0.dot(self.w0.T)

            c1 = err1 * self.sigm_d(z1)
            d1 = (-2*lrate/self.n) * l2.T.dot(c1)
            err2 = c1.dot(self.w1.T)

            c2 = err2 * self.sigm_d(z2)
            d2 = (-2*lrate/self.n) * self.tr_i.T.dot(c2)

            if i % acc_check == 0:
                # testing on training data
                jr = len(l0)
                kr = len(l0[0])
                ok = 0
                not_ok = 0
                for j in range(jr):
                    max = 0
                    for k in range(kr):
                        if max < l0[j, k]:
                            max = l0[j, k]
                            ans = k
                    if self.tr_l[j][ans] == 1:
                        ok += 1
                    else:
                        not_ok += 1
                self.acc_list_tr.append(ok/(ok+not_ok))

                # spinning test set
                z2 = np.matmul(self.test_i, self.w2)
                l2 = self.sigm(z2)
                z1 = np.matmul(l2, self.w1)
                l1 = self.sigm(z1)
                z0 = np.matmul(l1, self.w0)
                l0 = self.sigm(z0)
                # testing on test set
                jr = len(l0)
                kr = len(l0[0])
                ok = 0
                not_ok = 0
                for j in range(jr):
                    max = 0
                    for k in range(kr):
                        if max < l0[j, k]:
                            max = l0[j, k]
                            ans = k
                    if self.test_l[j][ans] == 1:
                        ok += 1
                    else:
                        not_ok += 1
                self.acc_list_test.append(ok/(ok+not_ok))

                self.err_list.append(np.mean(np.absolute(err0)))
                if verbose:
                    print("{}th loop - acc: tr {:.2%} test {:.2%}, err: {:.2%}"
                          .format(i, self.acc_list_tr[-1], self.acc_list_test[-1], self.err_list[-1]))

            # updating weights (applying -gradient)
            self.w0 += d0
            self.w1 += d1
            self.w2 += d2

            # end of loop
            i += 1

        self.total_time += time.time() - stopwatch
        self.total_loops += i
        if verbose:
            print("{} learning loops @ {} learning rate, {:.2f} s, {:.2f} loops/s"
                  .format(i, lrate, time.time() - stopwatch, i/(time.time() - stopwatch)))

    def stats(self):
        print("{} learning loops: {:.2f} s, {:.2f} loops/s"
              .format(self.total_loops, self.total_time, self.total_loops/self.total_time))

        plt.plot(self.err_list, label="err tr")
        plt.plot(self.acc_list_tr, label="acc tr")
        plt.plot(self.acc_list_test, label="acc test")
        plt.legend()
        plt.show()
