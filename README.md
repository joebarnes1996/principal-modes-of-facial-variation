# Principal modes of facial variation

Principal component analysis (PCA) creates vectors which represent the rotation of a coordinate system, such that each successive principal component explains the variance of a data in a direction orthogonal to all other principal components. While this is a powerful technique in the remit of dimensionality reduction, it also has uses in shape analysis.

Principal components can be applied to shape data to find the modes in which shapes vary the most. Here, shape data is represented as an array of x and y coordinates, i.e.

S = [[x_1, y_1],
	 [x_2, y_2],
	 [..., ...],
	 [x_N, y_N]]

where N is the number of coordinates that make up the shapes. 

One can convert these shape arrays into vectors, i.e. 

S --> S_bar = [x_1, y_1, x_2, y_2, ..., x_N, y_N]

It is then possible to perform PCA on these vectors as you would any other data. The principal modes of variation, can be found by varying the mean shape in the directions of each principal component. In principal component space, the mean shape is a column vector of zeros. Hence each mode of variation can be evaluated by transforming a principal component vector, multiplied by some amount, back to a set of x and y coordinates in Euclidean space.

An example of the first mode of variation of the human face is shown below, at +/- 3 standard deviations from the mean.

![First mode of variation](httphttps://github.com/joebarnes1996/principal-modes-of-facial-variation/comparison.png?raw=True)