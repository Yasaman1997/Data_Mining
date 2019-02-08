# About Iris data set





**Data Set Information:**

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Predicted attribute: class of iris plant.

This is an exceedingly simple domain. 



**Attribute Information:**

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica

# An important note: 
iris.target is an array of integers used to represent the Iris species. 0=Setosa, 1=Versicolor, 2=Virginica. 
And the KMeans model object also assigns integer ids for the three clusters (n_clusters =3 above), namely 0, 1, 2. 
Its important to note that the KMeans model has no knowledge of the iris.target data, and the clusters being given ids 0,1,2 is just a coincidence.




Source:
[UCI](http://archive.ics.uci.edu/ml/datasets/Iris)
