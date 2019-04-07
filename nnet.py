import sys
import time
import math

import numpy as np
import matplotlib.pyplot as plt


class NNet:
    def __init__(self, in_size, val_size, layers, random_seed=1, verbose=True,
            sigm=lambda x:1/(1+np.exp(-x)),
            sigm_d=lambda x:np.exp(-x)/np.power((np.exp(-x) + 1), 2)):
        self.verbose = verbose
        self.layers = len(layers)
        if self.verbose:
            print("nodes in layer: input {}, output {}".format(in_size, val_size))
            print("{} hidden layers: {}".format(self.layers, layers))

        self.sigm = sigm
        self.sigm_d = sigm_d

        np.random.seed(random_seed)
        self.weights = []
        self.weights.append(2*np.random.random([layers[0], val_size])-1)
        for i in range(1, self.layers):
            self.weights.append(2*np.random.random([layers[i], layers[i-1]])-1)
        self.weights.append(2*np.random.random([in_size, layers[self.layers-1]])-1)

        self.stats_err_list = []
        self.stats_acc_list = []
        self.stats_loopstamp = []
        self.total_time = 0
        self.total_epochs = 0

    def fit(self, x, y, epochs=sys.maxsize, batch=16, 
            timeout=sys.maxsize, lrate=5, stats_record=1):
        print('Fitting: batch size: {}'.format(batch))
        stopwatch = time.time()
        start_epoch = self.total_epochs

        while (time.time() - stopwatch < timeout 
            and self.total_epochs - start_epoch < epochs):

            for i in range(math.ceil(len(x)/batch)):
                batch_x = x[batch*i:batch*i+batch]
                batch_y = y[batch*i:batch*i+batch]

                err = []
                gradient = []
                c = []

                neuron_output, z = self._spin(batch_x)

                err.append(neuron_output[0] - batch_y)
                for i in range(self.layers):
                    c.append(err[i] * self.sigm_d(z[i]))
                    gradient.append((-2*lrate/batch) * neuron_output[i+1].T.dot(c[i]))
                    err.append(c[i].dot(self.weights[i].T))
                c.append(err[self.layers] * self.sigm_d(z[self.layers]))
                gradient.append((-2*lrate/batch) * batch_x.T.dot(c[self.layers]))

                for i in range(self.layers+1):
                    self.weights[i] += gradient[i]

            if (self.total_epochs - start_epoch) % stats_record == 0:
                self.record_stats(x, y)

            self.total_epochs += 1

        self.total_time += time.time() - stopwatch
        print('time: {:.2f} s, {:.2f} epochs/s'.format(
            self.total_time, 
            self.total_epochs/self.total_time))

    def predict(self, x):
        res, _ = self._spin(x)
        return res[0] 

    def evaluate(self, x, y):
        ok = 0
        not_ok = 0
        neuron_output, _ = self._spin(x)
        for j in range(len(neuron_output[0])):
            max = -1
            for k in range(len(neuron_output[0][0])):
                if max < neuron_output[0][j, k]:
                    max = neuron_output[0][j, k]
                    ans = k
            if y[j][ans] == 1:
                ok += 1
            else:
                not_ok += 1
        acc = ok/(ok+not_ok)
        err = np.mean(np.square(neuron_output[0] - y))

        return acc, err

    def record_stats(self, x, y):
        acc, err = self.evaluate(x, y)
        if self.verbose:
            self.print_stats(x, y, acc, err)

        self.stats_loopstamp.append(self.total_epochs)
        self.stats_acc_list.append(acc)
        self.stats_err_list.append(err)

    def print_stats(self, x, y, acc=None, err=None):
        if acc == None:
            acc, err = self.evaluate(x, y)

        print("{}th epoch - acc: {:.2%}, err: {:.3}".format(
            self.total_epochs, acc, err))

    def plot_stats(self):
        plt.figure()
        plt.subplot(211)
        plt.plot(self.stats_loopstamp, self.stats_err_list, label="Cost function")
        plt.legend()
        plt.subplot(212)
        plt.plot(self.stats_loopstamp, self.stats_acc_list, label="Accuracy")
        plt.legend()
        plt.show()

    def _spin(self, x):
        neuron_output = [0] * (self.layers+2)
        z = [0] * (self.layers+1)

        neuron_output[self.layers+1] = x
        for i in range(self.layers+1):
            z[self.layers-i] = np.matmul(
                neuron_output[self.layers-i+1], 
                self.weights[self.layers-i])
            neuron_output[self.layers-i] = self.sigm(z[self.layers-i])

        return neuron_output, z
