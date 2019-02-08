import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA


#np.random.seed(234234782384239784) # random seed for consistency

# A reader pointed out that Python 2.7 would raise a
# "ValueError: object of too small depth for desired array".
# This can be avoided by choosing a smaller random seed, e.g. 1
# or by completely omitting this line, since I just used the random seed for
# consistency.

u1 = [10, 10]
u2 = [22, 10]
cov = [[4, 4], [4, 9]]
#print(u1)

#print(cov_mat1)
#x1, y1= np.random.multivariate_normal(u1, cov, 1000).T
class1_sample = np.random.multivariate_normal(u1, cov, 1000).T

#print(u2)
#print(cov_mat2)
#x2, y2= np.random.multivariate_normal(u2, cov, 1000).T
class2_sample = np.random.multivariate_normal(u2, cov, 1000).T


#plot the random
#plt.plot(x1, y1, 'x')
#plt.plot(x2, y2, 'x')
#plt.figure(1)
#plt.plot(class1_sample[0, :], class1_sample[1, :], '+') #u1
#plt.plot(class2_sample[0, :], class2_sample[1, :], '+') #u2
#plt.axis('equal')
#plt.show()




#PCA
samples = np.concatenate((class1_sample, class2_sample), axis=1)
mlab_pca = mlabPCA(samples.T)
new_pca = sklearnPCA(n_components=1)
transformed = new_pca.fit_transform(samples.T)



new_transform = new_pca.inverse_transform(transformed)


#plot the random
#plt.plot(x1, y1, 'x')
#plt.plot(x2, y2, 'x')
plt.figure(1)
plt.plot(class1_sample[0, :], class1_sample[1, :], '+') #u1
plt.plot(class2_sample[0, :], class2_sample[1, :], '+') #u2
plt.axis('equal')
#plt.show()



#PLOTTING PCA
#plt.figure(1)
plt.plot(new_transform[0:1000, 0], new_transform[0:1000, 1],"+")
plt.plot(new_transform[1000:2000, 0], new_transform[1000:2000, 1],"+")
plt.axis('equal')
plt.show();