import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "DejaVu Sans",
#    "font.serif": ["Palatino"],
#})
"""
 Network architecture:
    ReLU resnet,
    Number of input variables: n = 20
    Degree of staircase/sparse parity: d = 20
    num_layers = 20
    layer_width = 50
    num_iter=1000000
    learning_rate=0.0001
    minibatch_size=100
"""

d = 10



datasetname = 'parity5_new'

(iter_range,losses) = pickle.load(open('trained_wts/' + datasetname + '/losses.pkl', 'rb'))
#print(iter_range) # Each of these is the loss checked at 1000 iterations.
#print(losses)

plt.figure(figsize=(7,5))
plt.plot(iter_range[1:],np.asarray(losses[1:]), color = 'red', label = 'Generalization Error for $\chi_{1:'+str(d)+'}$')
datasetname = 'stair5_new'
(iter_range,losses) = pickle.load(open('trained_wts/' + datasetname + '/losses.pkl', 'rb'))
plt.plot(iter_range[1:],np.asarray(losses[1:]), color = 'blue', label = 'Generalization Error for $S_{'+str(d)+'}$')
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.xlim([0,100000])
plt.ylim([0,1.5])
plt.legend(prop={'size': 20})
#plt.grid(axis = 'y')
plt.title("Learning $\chi_{1:"+str(d)+"}$ and $S_{"+str(d)+"}$", fontsize = 20)
plt.xlabel("SGD Iteration", fontsize = 20)
plt.ylabel("Generalization Error", fontsize = 20)
plt.show()




datasetname = 'parity5_new'
(iter_range, coeffs) = pickle.load(open('trained_wts/' + datasetname + '/coeffs.pkl','rb'))
#print(coeffs)

def plot_fourier_coeffs(stair_fourier_coeffs,title,location,ylimits,xlimits,line_style):
    if stair_fourier_coeffs:
        stair_fourier_coeffs = np.asarray(stair_fourier_coeffs).T
        print(stair_fourier_coeffs.shape)
        plt.figure(figsize=(7,5))
        for i in range(1,d+1):
            plt.plot(range(0,100000,1000), stair_fourier_coeffs[i,:], label= "$\hat{f}_{1:" + str(i)+ "}$",ls = line_style)
        plt.legend(loc = location, prop={'size': 18},ncol=2)
        plt.title(title, fontsize = 20)
        plt.xticks(fontsize= 14)
        plt.yticks(fontsize= 14)
        plt.ylim(ylimits)
        plt.xlim(xlimits)
        plt.xlabel("SGD Iteration", fontsize = 18)
        plt.ylabel("Fourier Coefficient", fontsize = 18)
        plt.show()

plot_fourier_coeffs(coeffs, "Fourier Coefficients for Learning $\chi_{1:"+str(d)+"}$", "upper right",[-0.02,0.20],[0,100000],'--')
datasetname = 'stair5_new'
(iter_range, coeffs) = pickle.load(open('trained_wts/' + datasetname + '/coeffs.pkl','rb'))
plot_fourier_coeffs(coeffs, "Fourier Coefficients for Learning $S_{"+str(d)+"}$", "lower right",[0.0,0.45],[0,70000],'--')

