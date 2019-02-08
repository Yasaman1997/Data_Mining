import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA

u1 = np.transpose([10, 10])
u2 = np.transpose([22, 10])
cov = [[4, 4], [4, 9]]
#print(u1)
#print(u2)
#u1 = [10, 10]
#u2 = [22, 10]
#cov = [[4, 4], [4, 9]]

class1_sample = np.random.multivariate_normal(u1, cov, 1000).T
class2_sample = np.random.multivariate_normal(u2, cov, 1000).T


plt.figure(1)
#plt.axis('equal')
plt.plot(class1_sample[0, :], class1_sample[1, :], '+') #u1
plt.plot(class2_sample[0, :], class2_sample[1, :], '+') #u2


plt.figure(2)
plt.plot(class1_sample[ : ,:] , class1_sample[: ,: ] , '+')
plt.plot(class2_sample[:, :], class2_sample[:,:] , '+')

#pca

samples = np.concatenate((class1_sample, class2_sample), axis=1)
mlab_pca = mlabPCA(samples.T)




plt.figure(3)
plt.plot(mlab_pca.Y[0:1000,0],"^", markersize=7, color='green', alpha=0.5, label='class1')
plt.plot(mlab_pca.Y[1000:2000,0],"*" ,markersize=7, color='purple', alpha=0.5, label='class2')

#plt.figure(1)
new_pca = sklearnPCA(n_components=1)
transformed = new_pca.fit_transform(samples.T)



new_transform = new_pca.inverse_transform(transformed)
#draw PCA
plt.figure(1)
plt.plot(new_transform[0:1000, 0], new_transform[0:1000, 1],"*")
plt.plot(new_transform[1000:2000, 0], new_transform[1000:2000, 1],"*")


#reconstruction error
reconstruction_error = ((new_transform - samples.T) ** 2).mean()
print (np.sqrt(reconstruction_error))

plt.show()





