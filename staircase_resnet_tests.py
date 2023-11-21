import itertools
import functools
# import pypac.tools

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader

import random

import copy
import math
import pickle

from neural_net_architectures import ReLUResNet, QuadResNet, OneLayer, ReLUResNetNormalized
# QuadNetSumLayers

torch.random.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# Evaluate function given by Fourier representation (i.e., multilinear polynomial) at a certain point:
# Multilinear polynomial is given as a tuple : float dict, where each tuple consists of the relevant indices.
def eval_fourier_fn(x,fourier_fn):
    val = 0
    for term, coeff in fourier_fn.items():
        curr_mon = coeff
        for idx in term:
            curr_mon *= x[idx]
        val += curr_mon
    return val

def get_staircase_fourier_fn(n):
    fourier_fn = {}
    for i in range(n):
        fourier_fn[tuple(range(i+1))] = 1
    return fourier_fn

def get_sparse_fourier_fn(n):
    fourier_fn = {}
    fourier_fn[tuple(range(n))] = 1
    return fourier_fn

def eval_parity_fast(x,d):
    val = 1
    for i in range(d):
        val *= x[i]
    return val

def eval_staircase_fast(x,d):
    val = 0
    for i in range(d):
        val += 1
        val *= x[d-1-i]
    return val/math.sqrt(d)

class DatasetFromFunc(IterableDataset):
    """Face Landmarks dataset."""

    def __init__(self, n, eval_fn):
        """
        Args:
            n (int): input length.
            eval_fn (fn): evaluation function $\{-1,1\}^n \to \RR$
        """
        self.n = n
        self.eval_fn = eval_fn

    def __iter__(self):
        while True:
#            yield torch.randint(2,(1,self.n))*2-1, torch.rand(1)
            x = [random.randint(0,1)*2-1 for i in range(self.n)]
            x = torch.FloatTensor(x)
            yield x, torch.FloatTensor([self.eval_fn(x)])

class ERMDatasetFromFunc(IterableDataset):
    """Face Landmarks dataset."""

    def __init__(self, n, eval_fn,erm_num_samples):
        """
        Args:
            n (int): input length.
            eval_fn (fn): evaluation function $\{-1,1\}^n \to \RR$
        """
        self.n = n
        self.eval_fn = eval_fn


        self.erm_num_samples = erm_num_samples
        self.counter = 0
        self.xs = []
        self.ys = []

        fn_dataset = DatasetFromFunc(n, eval_fn)
        dataloader = DataLoader(fn_dataset, batch_size=erm_num_samples, num_workers=0)
        dataloaderiter = iter(dataloader)

        data = next(dataloaderiter)
        self.xs, self.ys = data

    def __iter__(self):
        while True:
            self.counter += 1
            if self.counter == self.erm_num_samples:
                self.counter = 0
            yield self.xs[self.counter,:], self.ys[self.counter,:]



def truthtable(n,eval_fn):
    return [eval_fn(x) for x in itertools.product([1,-1], repeat=n)]

from sympy import fwht


def is_stair_coeff(tup):
    isOne = False
    for j in tup:
        if j == 1:
            isOne = True
        else:
            if isOne:
                return False
    return True


def net_statistics(net):
    if net.__class__ != OneLayer:
        print('Net statistics only implemented for OneLayer model.')
        return
    wts1 = net.linear1.weight.detach().numpy()
    biases1 = net.linear1.bias.detach().numpy()
    wts2 = net.linear2.weight.detach().numpy()

def save_net_statistics(net):
    if net.__class__ != OneLayer:
        print('Net statistics only implemented for OneLayer model.')
        return
    wts1 = net.linear1.weight.detach().numpy()
    biases1 = net.linear1.bias.detach().numpy()
    wts2 = net.linear2.weight.detach().numpy()
    all_wts = (wts1, biases1, wts2)
    pickle.dump(all_wts, open('data/onelayer_stats.pkl','wb'))

def train_function(eval_fn,n,num_layers,layer_width,track_fourier_coeffs,num_iter_limit=None,refresh_fourier_freq=4096,learning_rate=0.05,sgd_noise=0,batch_size=1,net_type=QuadResNet,close_to_zero_init=False,netparamsfolder=None,refresh_save_rate=1000,erm=False,erm_num_samples=10000):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    print(device)

    print('Train_function started')
    net = net_type(n,num_layers,layer_width)
    if close_to_zero_init:
        net.set_closetozeroinit(1e-7);
    net.to(device)


    criterion = nn.MSELoss()
    paramstotrain = filter(lambda p: p.requires_grad, net.parameters())

    def lr_sched(iter_num):
        lr_decay = 15000 #learning rate decay parameter
        if iter_num < lr_decay:
            return learning_rate
        if iter_num < 2*lr_decay:
            return learning_rate/2
        if iter_num < 3*lr_decay:
            return learning_rate/4
        if iter_num < 4*lr_decay:
            return learning_rate/8
        if iter_num < 5*lr_decay:
            return learning_rate/16
        if iter_num < 6*lr_decay:
            return learning_rate/32
        else:
            return learning_rate/64
        #return learning_rate*math.exp(-iter_num/lr_decay)

    running_losses = []


    if not erm:
        fn_dataset = DatasetFromFunc(n, eval_fn)
    else:
        fn_dataset = ERMDatasetFromFunc(n,eval_fn,erm_num_samples)

    print(fn_dataset)
#    dataloader = DataLoader(fn_dataset, batch_size=1, shuffle=True, num_workers=0)
    dataloader = DataLoader(fn_dataset, batch_size=batch_size, num_workers=0)

    if erm:
        popfn_dataset = DatasetFromFunc(n, eval_fn)
        popdataloader = DataLoader(popfn_dataset, batch_size=batch_size, num_workers=0)
        popdataloaderiter = iter(popdataloader)

    dataloaderiter = iter(dataloader)

    maxi = num_iter_limit
    running_loss = 0.0
    runpoploss = 0.0
    print('Starting training')
    for iter_num in range(maxi):  # loop over the dataset multiple times

        paramstotrain = filter(lambda p: p.requires_grad, net.parameters())

        #optimizer = optim.Adam(paramstotrain, lr=lr_sched(iter_num))
        optimizer = optim.SGD(paramstotrain, lr=lr_sched(iter_num), momentum=0)

        data = next(dataloaderiter)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
#        print(inputs)

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels[:,0]

        # print(labels.shape)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if erm:
            with torch.no_grad():
                popdata = next(popdataloaderiter)
                popinputs, poplabels = popdata
                popinputs = popinputs.to(device)
                poplabels = poplabels.to(device)
                poplabels = poplabels[:,0]
                poploss = criterion(net(popinputs),poplabels)
                runpoploss += poploss

        if iter_num % 100 == 0:
            print(iter_num)
        if iter_num % 100 == 0 and sgd_noise > 0:
            paramstoperturb = filter(lambda p: p.requires_grad, net.parameters())
            with torch.no_grad():
                for currparam in paramstoperturb:
                    noisetensor = torch.randn(currparam.size()) * sgd_noise
                    noisetensor = noisetensor.to(device)
                    currparam.add_(noisetensor)

        # print statistics
        running_loss += loss.item()
        # print(iter_num)
        if iter_num % 100 == 99:    # print every 1000 mini-batches
            print(iter_num)
            print('running training loss', running_loss)
            running_losses.append(running_loss)
            running_loss = 0.0

            print('Population loss',runpoploss)
            runpoploss = 0.0


        if iter_num % refresh_save_rate == 0:
            pickle.dump(net, open(netparamsfolder + 'net' + str(iter_num) + '.pkl', 'wb'))


    return running_losses



def plot_fourier_coeffs(stair_fourier_coeffs,title):
    if stair_fourier_coeffs:
        stair_fourier_coeffs = np.asarray(stair_fourier_coeffs).T
        print(stair_fourier_coeffs.shape)
        plt.figure()
        for i in range(1,20):
            plt.plot(range(stair_fourier_coeffs.shape[1]), stair_fourier_coeffs[i,:], label=str(i))
        plt.legend()
        plt.title(title)
        plt.show()

import os
import glob
def train_net_and_save_params(n,d,num_layers,layer_width,num_iter,learning_rate,batch_size,net_type,eval_fn,netparamsfolder,refresh_save_rate,erm=False,erm_num_samples=10000):

    sgd_noise=0
    close_to_zero_init=False
    track_fourier_coeffs = []

    try:
        os.mkdir(netparamsfolder)
    except Exception as e:
        print(e)


    # window = Tkinter.Tk()
    # window.wm_withdraw()


    # tkMessageBox.showinfo(title="Greetings", message="Deleting all files: " + netparamsfolder + 'net*.pkl')
    files = glob.glob(netparamsfolder + '*.pkl')
    for f in files:
        os.remove(f)

    train_function(eval_fn,n,num_layers,layer_width, \
                track_fourier_coeffs,num_iter_limit=num_iter,refresh_fourier_freq=None, \
                learning_rate=learning_rate,sgd_noise=sgd_noise,batch_size=batch_size, \
                net_type=net_type, close_to_zero_init=close_to_zero_init,netparamsfolder=netparamsfolder,refresh_save_rate=refresh_save_rate,erm=erm,erm_num_samples=erm_num_samples)


    print('Finished training -- saved network parameters over time to ' + netparamsfolder)

def eval_fourier_tup(inputs, fourier_tup):
    # print(inputs,fourier_tup)
    labels = torch.ones(len(inputs[:,1]))
    for j,v in enumerate(fourier_tup):
        if v == -1:
            labels = labels * inputs[:,j]
    return labels

def output_losses_and_fourier_coeffs(eval_fn,n,iter_range,batch_size=1,track_fourier_coeffs_tuples=[],netparamsfolder=None):

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    print(device)

    print('Loss_output function started')

    with torch.no_grad():
        running_losses = []
        running_coeffs = []
        # for i in range(len(track_fourier_coeffs_tuples)):
        #     running_coeffs.append([])
        r_range = iter_range
        print(r_range)
        for r in r_range:
            print(r)
            net = pickle.load(open(netparamsfolder + 'net' + str(r) + '.pkl', 'rb'))
            net.to(device)
            # print(net)

            criterion = nn.MSELoss()



            fn_dataset = DatasetFromFunc(n, eval_fn)
            print(fn_dataset)
        #    dataloader = DataLoader(fn_dataset, batch_size=1, shuffle=True, num_workers=0)
            dataloader = DataLoader(fn_dataset, batch_size=batch_size, num_workers=0)
            #
            # if refresh_fourier_freq is not None:
            #     all_inputs = torch.FloatTensor([x for x in itertools.product([1,-1], repeat=n)])
            #     all_inputs = all_inputs.to(device)


        # maxi = num_iter_limit
            dataloaderiter = iter(dataloader)
            print('Loading data')
            tot_loss = 0
            for iter_num in range(1):  # loop over the dataset multiple times

                data = next(dataloaderiter)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
        #        print(inputs)
                inputs = inputs.to(device)
                print(inputs.shape)
                labels = labels.to(device)
                labels = labels[:,0]
                print(labels.shape)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                running_coeffs.append([])
                for fourier_ind,fourier_tup in enumerate(track_fourier_coeffs_tuples):
                    fourier_tup_labels = eval_fourier_tup(inputs, fourier_tup)
                    fourier_tup_val = 0
                    fourier_tup_val = torch.mean(outputs * fourier_tup_labels)
                    running_coeffs[-1].append(fourier_tup_val)

                tot_loss += loss.item()
                print(tot_loss)
            running_losses.append(tot_loss)
        print(running_losses)
        pickle.dump((iter_range,running_losses),open(netparamsfolder + 'losses.pkl','wb'))
        pickle.dump((iter_range,running_coeffs),open(netparamsfolder + 'coeffs.pkl','wb'))


        return running_losses, running_coeffs

def get_staircase_fourier_coeff_tuples(n,d):
    track_fourier_coeffs_tuples = []
    for j in range(d+1):
        curr_coeff = []
        for i in range(n):
            if i < j:
                curr_coeff.append(-1)
            else:
                curr_coeff.append(1)
        curr_coeff = tuple(curr_coeff)
        track_fourier_coeffs_tuples.append(curr_coeff)
    return track_fourier_coeffs_tuples

def train_stair5_net():
    """
    Staircase, width of 100. n = d = 20.
    """
    refresh_save_rate=1000
    n = 30
    d = 10 # degree of staircase
    num_layers = 5
    layer_width = 40
    num_iter=100000
    learning_rate=0.01
    batch_size=20
    erm=True
    erm_num_samples = 60000
    net_type=ReLUResNet
    # net_type=ReLUResNetNormalized
    eval_fn = functools.partial(eval_staircase_fast,d=d) ####### LEARN STAIRCASE FUNCTION
    netparamsfolder = 'trained_wts/stair5_new/'
    train_net_and_save_params(n,d,num_layers,layer_width,num_iter,learning_rate,batch_size,net_type,eval_fn,netparamsfolder,refresh_save_rate,erm,erm_num_samples)





def train_parity5_net():
    """
    Staircase, width of 100. n = d = 20.
    """
    refresh_save_rate=1000
    n = 30
    d = 10 # degree of staircase
    num_layers = 5
    layer_width = 40
    num_iter=100000
    learning_rate=0.01
    batch_size=20
    erm=True
    erm_num_samples = 60000
    net_type=ReLUResNet
    # net_type=ReLUResNetNormalized
    eval_fn = functools.partial(eval_parity_fast,d=d) ####### LEARN STAIRCASE FUNCTION
    netparamsfolder = 'trained_wts/parity5_new/'
    train_net_and_save_params(n,d,num_layers,layer_width,num_iter,learning_rate,batch_size,net_type,eval_fn,netparamsfolder,refresh_save_rate,erm,erm_num_samples)


def eval_loss_and_coeffs_stair5():
    """
    Loads trained_wts/stair5/
    and computes loss over time and fourier coefficients
    """
    n = 30
    d = 10
    iter_range = range(0,100000,1000)
    batch_size=30000 
    eval_fn = functools.partial(eval_staircase_fast,d=d)
    netparamsfolder = 'trained_wts/stair5_new/'

    track_fourier_coeffs_tuples = get_staircase_fourier_coeff_tuples(n,d)

    losses, fourier_coeffs = output_losses_and_fourier_coeffs(eval_fn,n,iter_range,batch_size=batch_size,track_fourier_coeffs_tuples=track_fourier_coeffs_tuples,netparamsfolder=netparamsfolder)



def eval_loss_and_coeffs_parity5():
    """
    Loads trained_wts/parity5/
    and computes loss over time and fourier coefficients
    """
    n = 30
    d = 10
    iter_range = range(0,100000,1000)
    batch_size=30000 # number of samples to evaluate loss & estimate fourier coefficients at each point
    eval_fn = functools.partial(eval_parity_fast,d=d)
    netparamsfolder = 'trained_wts/parity5_new/'

    track_fourier_coeffs_tuples = get_staircase_fourier_coeff_tuples(n,d)

    losses, fourier_coeffs = output_losses_and_fourier_coeffs(eval_fn,n,iter_range,batch_size=batch_size,track_fourier_coeffs_tuples=track_fourier_coeffs_tuples,netparamsfolder=netparamsfolder)


def main():

     train_stair5_net()
     train_parity5_net()
     eval_loss_and_coeffs_stair5()
     eval_loss_and_coeffs_parity5()


if __name__ == '__main__':
    main()



