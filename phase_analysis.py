###
#Writer:  Koyo Miyamura
#Summary: Phase analysis using performance counter (CSV file).
#Input:   Performance counter from each programs and labels
#         (Input csv is got using program at https://github.com/koyo-miyamura/perf_analyze_rewrite)
#Output:  Plot feature value of each programs
#Remarks: I labeled CHANGE in the place which I thought you want to change. (you can search the word Chan to find the variable places)
#         I recomend you to read the document http://scikit-learn.org/stable/ to use the script
###

from time import time
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.decomposition import PCA
from sklearn.externals import joblib

###############################################################################
# Download from csv
#CHANGE
bzip2_train  = np.loadtxt("csv/bzip2_train.csv", delimiter = ";", skiprows=1) 
bzip2_2      = np.loadtxt("csv/bzip2_2_test.csv", delimiter = ";", skiprows=1)
gcc          = np.loadtxt("csv/gcc.csv", delimiter = ";", skiprows=1)
perlbench    = np.loadtxt("csv/perlbench.csv", delimiter = ";", skiprows=1)
hmmer        = np.loadtxt("csv/hmmer.csv", delimiter = ";", skiprows=1)
libquantum   = np.loadtxt("csv/libquantum.csv", delimiter = ";", skiprows=1)
sphinx3_train= np.loadtxt("csv/sphinx3_train.csv", delimiter = ";", skiprows=1)
bwaves       = np.loadtxt("csv/bwaves.csv", delimiter = ";", skiprows=1)
lbm          = np.loadtxt("csv/lbm.csv", delimiter = ";", skiprows=1)
wrf          = np.loadtxt("csv/wrf.csv", delimiter = ";", skiprows=1)
mcf          = np.loadtxt("csv/mcf.csv", delimiter = ";", skiprows=1)

X = np.concatenate([bzip2_train[:,1:-1],bzip2_2[:,1:-1], gcc[:,1:-1], perlbench[:,1:-1], hmmer[:,1:-1], libquantum[:,1:-1], sphinx3_train[:,1:-1], bwaves[:,1:-1], lbm[:,1:-1], mcf[:,1:-1], wrf[:,1:-1] ])
y = np.concatenate([map(int,bzip2_train[:,-1]), map(int,bzip2_2[:,-1]), map(int,gcc[:,-1]), map(int,perlbench[:,-1]), map(int,hmmer[:,-1]), map(int,libquantum[:,-1]), map(int,sphinx3_train[:,-1]),map(int,bwaves[:,-1]), map(int,lbm[:,-1]), map(int,mcf[:,-1]), map(int,wrf[:,-1])])


def plot_embedding(X,name,c=0):
    X = (X - x_min) / (x_max - x_min)
    plt.plot(X[:, 0], X[:, 1], marker="o", linestyle="none", markersize=3, label=name, color=cm.spectral(c))

    #To plot each program
    #CHANGE
    #(a)
    plt.xticks([0,1]), plt.yticks([0,1])
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("The first principal component")
    plt.ylabel("The second principal component")
    plt.title("Feature value projection by PCA")
    plt.rcParams["font.size"] = 24
    plt.legend(numpoints=1)
    plt.show()

    #Save the plot
    #CHANGE
    #plt.savefig(name + ".png")

    plt.clf()

#Decomposion using PCA
print("Computing PCA projection")
t0 = time()
#X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
PCA   = decomposition.TruncatedSVD(n_components=2).fit(X)
X_pca = PCA.transform(X)

#CHANGE
bzip2_tr      = PCA.transform(bzip2_train[:,1:-1])
bzip2_2_tr    = PCA.transform(bzip2_2[:,1:-1])
gcc_tr        = PCA.transform(gcc[:,1:-1])
perlbench_tr  = PCA.transform(perlbench[:,1:-1])
hmmer_tr      = PCA.transform(hmmer[:,1:-1])
libquantum_tr = PCA.transform(libquantum[:,1:-1])
sphinx3_tr    = PCA.transform(sphinx3_train[:,1:-1])
bwaves_tr     = PCA.transform(bwaves[:,1:-1])
lbm_tr        = PCA.transform(lbm[:,1:-1])
wrf_tr        = PCA.transform(wrf[:,1:-1])
mcf_tr        = PCA.transform(mcf[:,1:-1])

x_min, x_max = np.min(X_pca, 0), np.max(X_pca, 0)

#CHANGE
plot_embedding(bzip2_tr,name='bzip2',c=0)
plot_embedding(bzip2_2_tr,name='bzip2_2',c=0.1)
plot_embedding(gcc_tr,name='gcc',c=0.2)
plot_embedding(perlbench_tr,name='perlbench',c=0.3)
plot_embedding(hmmer_tr,name='hmmer',c=0.4)
plot_embedding(libquantum_tr,name='libquantum',c=0.5)
plot_embedding(sphinx3_tr,name='sphinx3',c=0.6)
plot_embedding(bwaves_tr,name='bwaves',c=0.7)
plot_embedding(lbm_tr,name='lbm',c=0.8)
plot_embedding(wrf_tr,name='wrf',c=0.9)
plot_embedding(mcf_tr,name='mcf',c=1.0)

#To plot all results of each programs, please comment out the part of (a) and remove # of (b)
#CHANGE
#(b)
#plt.xticks([0,1]), plt.yticks([0,1])
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.xlabel("The first principal component")
#plt.ylabel("The second principal component")
#plt.title("Feature value projection by PCA")
#plt.rcParams["font.size"] = 24
#plt.legend(markerscale=3, numpoints=1)
#plt.show()


"""
#Decomposion using t-SNE
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne)
print("time = %f",time()-t0)
                              
#plt.legend()
plt.show()
#plt.savefig('t-SNE.png')
"""
###############################################################################

