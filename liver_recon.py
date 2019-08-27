import numpy as np
import scipy.io as sio
from NUFFT_funct import nufft_init, nufft_adj
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mat_cont = sio.loadmat("liver_radial.mat")
d = mat_cont["d"] # k-space data
k = mat_cont["k"] # k-space trajectory
w = mat_cont["w"] #density compensation
om_r = np.real(np.reshape(np.swapaxes(k,0,1), (np.shape(k)[0]*np.shape(k)[1], 1)))
om_i = np.imag(np.reshape(np.swapaxes(k,0,1), (np.shape(k)[0]*np.shape(k)[1], 1)))
om = np.concatenate((om_r,om_i),axis=1)*2*math.pi
print(np.shape(om))
Nd = np.array([380,380])
Jd = np.array([6,6])
Kd = np.array([[384,384]])
n_shift = np.array([180,180])
res = {}
res["st"] = nufft_init(om,Nd,Jd,Kd,n_shift, "kaiser")

res["adjoint"] = 0
res["imSize"] = [380,380]
res["imSize2"] = [np.shape(k)[0],np.shape(k)[0]]
res["dataSize"] = np.shape(k)
res["w"] = np.sqrt(w)
y = d
b = y*w
b = np.reshape(np.swapaxes(b,0,1),(np.shape(b)[0]*np.shape(b)[1],1))

c =np.reshape(nufft_adj(b,res["st"])/np.sqrt(np.prod(res["imSize2"])),(res["imSize"][0], res["imSize"][1]))

plt.imshow(np.abs(c),cmap="gray")
plt.savefig("Full_sampled(Liver).png")

