from numpy.random import randn
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib import pyplot
import pandas as pd
import scipy.io
import numpy as np

# mat = scipy.io.loadmat('lung.mat')
# X = mat['X']    # data
# X = X.astype(float)
# y = mat['Y']    # label
# y = y[:, 0]


df = pd.read_csv('ALLAML.csv')
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

n_samples, n_features = X.shape

for i in range(n_samples):
    data= X[i]
    data = data.reshape((len(data),-1))
    kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    data_trans = kbins.fit_transform(data)
    data_trans = data_trans.reshape(-1,)+1
    X[i]=data_trans


va=[]
for val in range(n_features):
    va.append("Att"+str(val))
va.append('Class')
y=y[:,None]
cd=np.append(X,y,axis=1)
df=pd.DataFrame(cd)
df.to_csv("dis_lung.csv",index=False,header=va)