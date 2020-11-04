# Principal modes of facial variation

## 1 - Introduction and theory

Principal component analysis (PCA) is commonly used as a method of dimensionality reduction, essentially it creates the orthogonal vectors which represent the directions with which the data has the greatest variance. While a less common application, PCA can be used in order to understand the ways in which groups of shapes vary the most.

Shapes can be mathematically represented as an array of x and y pairs. For example, a shape made up of n points could be represented as

        [ x_1 y_1 ]
    S = [ ... ... ]
        [ x_n y_n ]

PCA relies on data being represented as column vectors, hence to perform PCA on shape data, one first has to convert them into column vectors of alternating x and y coordinates, as shown below

		[ x_1 ]
		[ y_1 ]
    S    = 	[ ... ]
		[ x_n ]
		[ y_n ]

With each shape represented by a vector, one can standardise the data before performing PCA following standard procedures. Having the principal components of shape data, it is easy to evaluate how shapes vary in the direction of each principal component. For example, the shape vectors across the first principal component can be evaluated as <img src="https://render.githubusercontent.com/render/math?math=\mu \pm v \sqrt{\lambda_i} \varphi_i">, where <img src="https://render.githubusercontent.com/render/math?math=\mu"> is the mean shape (equal to 0 in principal component space), <img src="https://render.githubusercontent.com/render/math?math=\varphi_i"> is the i-th principal component, <img src="https://render.githubusercontent.com/render/math?math=\lambda_i"> is the i-th eigenvalue (variance of the i-th principal component), and <img src="https://render.githubusercontent.com/render/math?math=v"> is the number of standard deviations from the mean in the direction of the i-th principal component that you wish to evaluate the shape. Of course the output of this is not a shape, but a vector in principal component space. However, one can unstandardise this vector, and then convert it back to a series of x and y coordinates, corresponding to a shape in original feature space.


## 2 - Implementation and results

The shape data that I demonstrate this technique on is the shape data of human faces, extracted from 350 facial images. Though I can't show the facial images for anonymity reasons, I can show the mean face, which was taken by morphing each face to the same shape, and taking the mean of the image tensors. The mean facial image and mean facial features are shown below.d

![First mode of variation](comparison.png?raw=True)
