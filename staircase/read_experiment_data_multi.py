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

"""
Load data from: staircase
\sum_{i=1}^20 \prod_{j=1}^i x_j
"""


d_1 = 7
d_2 = 7

plt.figure(figsize=(7,5))

datasetname = 'stair5_multi'
(iter_range,losses) = pickle.load(open('trained_wts/' + datasetname + '/losses.pkl', 'rb'))
plt.plot(iter_range[1:],np.asarray(losses[1:]), color = 'blue', label = 'Generalization Error for $S_{'+str(d_1)+','+str(d_2)+'}$')
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.xlim([0,30000])
plt.ylim([0,5])
plt.legend(prop={'size': 16}, loc = 'upper right')
#plt.grid(axis = 'y')
plt.title("Learning $S_{"+str(d_1)+","+str(d_2)+"}$", fontsize = 18)
plt.xlabel("SGD Iteration", fontsize = 18)
plt.ylabel("Generalization Error", fontsize = 18)
plt.show()






def plot_fourier_coeffs(stair_fourier_coeffs,title,location,ylimits,xlimits,line_style):
    if stair_fourier_coeffs:
        stair_fourier_coeffs = np.asarray(stair_fourier_coeffs).T
        print(stair_fourier_coeffs.shape)
        plt.figure(figsize=(7,5))
        for i in range(1,d_1 + 1):
            plt.plot(range(0,50000,1000), stair_fourier_coeffs[i,:], label= "$\hat{f}_{1:" + str(i)+ "}$",ls = line_style)
        for i in range(d_1 + 1,d_1 + d_2 -1):
            plt.plot(range(0,50000,1000), stair_fourier_coeffs[i,:], label= "$\hat{f}_{\{1\}\cup\{6:" + str(i)+ "\}}$",ls = line_style)
        plt.legend(loc = location, prop={'size': 16},ncol=2)
        plt.title(title, fontsize = 16)
        plt.xticks(fontsize= 12)
        plt.yticks(fontsize= 12)
        plt.ylim(ylimits)
        plt.xlim(xlimits)
        plt.xlabel("SGD Iteration", fontsize = 16)
        plt.ylabel("Fourier Coefficient", fontsize = 16)
        plt.show()


datasetname = 'stair5_multi'
(iter_range, coeffs) = pickle.load(open('trained_wts/' + datasetname + '/coeffs.pkl','rb'))
plot_fourier_coeffs(coeffs, "Fourier Coefficients for Learning $S_{"+str(d_1)+","+str(d_2)+"}$", "lower right",[0.0,1.2],[0,30000],'--')

