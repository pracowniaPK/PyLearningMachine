import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


class NNet:

    def __init__(self, training_in, test_in, trainig_val, test_val, layers, random_seed=1):
        if len(training_in) != len(trainig_val) or len(test_in) != len(test_val):
            raise ValueError("lengths of training and test sets dont't match")

        self.training_in = training_in
        self.test_in = test_in
        self.trainig_val = trainig_val
        self.test_val = test_val
        self.n = len(training_in)
        m = len(test_in)
        print("nodes in layer: input {}, output {}".format(self.n, m))
        self.layers = len(layers)
        print("{} hidden layers: {}".format(self.layers, layers))

        np.random.seed(random_seed)
        self.weights = []
        self.weights.append(2*np.random.random([layers[0], len(trainig_val[0])])-1)
        for i in range(1, self.layers):
            self.weights.append(2*np.random.random([layers[i], layers[i-1]])-1)
        self.weights.append(2*np.random.random([len(self.training_in[0]), layers[self.layers-1]])-1)

        self.err_list = []
        self.acc_list_tr = []
        self.acc_list_test = []
        self.stats_loopstamp = []
        self.total_time = 0
        self.total_loops = 0

    def sigm(self, x):
        return 1/(1+np.exp(-x))

    def sigm_d(self, x):
        return np.exp(-x)/np.power((np.exp(-x) + 1), 2)

    def spin(self, lrate, loops=sys.maxsize, timeout=sys.maxsize, acc_check=10, verbose=False):
        loop_no = 0
        stopwatch = time.time()

        while time.time() - stopwatch < timeout and loop_no < loops:
            z = [0] * (self.layers+1)
            neuron_output = [0] * (self.layers+1)
            err = []
            c = []
            gradient = []

            z[self.layers] = np.matmul(self.training_in, self.weights[self.layers])
            for i in range(self.layers):
                neuron_output[self.layers-i] = self.sigm(z[self.layers-i])
                z[self.layers-i-1] = np.matmul(neuron_output[self.layers-i], self.weights[self.layers-i-1])
            neuron_output[0] = self.sigm(z[0])

            err.append(neuron_output[0] - self.trainig_val)
            for i in range(self.layers):
                c.append(err[i] * self.sigm_d(z[i]))
                gradient.append((-2*lrate/self.n) * neuron_output[i+1].T.dot(c[i]))
                err.append(c[i].dot(self.weights[i].T))
            c.append(err[self.layers] * self.sigm_d(z[self.layers]))
            gradient.append((-2*lrate/self.n) * self.training_in.T.dot(c[self.layers]))

            if loop_no % acc_check == 0:
                self.stats_loopstamp.append(loop_no)
                # testing on training data
                ok = 0
                not_ok = 0
                for j in range(len(neuron_output[0])):
                    max = 0
                    for k in range(len(neuron_output[0][0])):
                        if max < neuron_output[0][j, k]:
                            max = neuron_output[0][j, k]
                            ans = k
                    if self.trainig_val[j][ans] == 1:
                        ok += 1
                    else:
                        not_ok += 1
                self.acc_list_tr.append(ok/(ok+not_ok))

                # spinning test set
                z[self.layers] = np.matmul(self.test_in, self.weights[self.layers])
                for i in range(self.layers):
                    neuron_output[self.layers-i] = self.sigm(z[self.layers-i])
                    z[self.layers-i-1] = np.matmul(neuron_output[self.layers-i], self.weights[self.layers-i-1])
                neuron_output[0] = self.sigm(z[0])
                # testing on test set
                jr = len(neuron_output[0])
                kr = len(neuron_output[0][0])
                ok = 0
                not_ok = 0
                for j in range(jr):
                    max = 0
                    for k in range(kr):
                        if max < neuron_output[0][j, k]:
                            max = neuron_output[0][j, k]
                            ans = k
                    if self.test_val[j][ans] == 1:
                        ok += 1
                    else:
                        not_ok += 1
                self.acc_list_test.append(ok/(ok+not_ok))

                self.err_list.append(np.mean(np.square(err[0])))
                if verbose:
                    print("{}th loop - acc: tr {:.2%} test {:.2%}, err: {:.3}"
                          .format(loop_no, self.acc_list_tr[-1], self.acc_list_test[-1], self.err_list[-1]))

            # updating weights (applying -gradient)
            for i in range(self.layers+1):
                self.weights[i] += gradient[i]

            loop_no += 1

        self.total_time += time.time() - stopwatch
        self.total_loops += loop_no
        if verbose:
            print("{} learning loops @ {} learning rate, {:.2f} s, {:.2f} loops/s"
                  .format(loop_no, lrate, time.time() - stopwatch, loop_no/(time.time() - stopwatch)))

    def plot_stats(self):
        print("{} learning loops: {:.2f} s, {:.2f} loops/s"
              .format(self.total_loops, self.total_time, self.total_loops/self.total_time))

        plt.plot(self.stats_loopstamp, self.err_list, label="err tr")
        plt.plot(self.stats_loopstamp, self.acc_list_tr, label="acc tr")
        plt.plot(self.stats_loopstamp, self.acc_list_test, label="acc test")
        plt.legend()
        plt.show()