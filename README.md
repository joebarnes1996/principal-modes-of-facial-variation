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

The shape data I use to demonstrate this technique was used within my MSc dissertation project, and contains the facial images of 86 individuals. Each individual had 2 lab photos (facial images taken under strict quality control, similar to passport photos), and 2 selfies. Due to privacy, I cannot add these images to this repository, but I can upload their facial features (facial shape), which was extracted using the dlib-ml package.

As well as the data relating to facial shape, I have also uploaded the mean facial shape, and the mean facial image (found by morphing all faces to the same shape, before taking the mean of the image tensors. The mean facial shape and image are shown below.


<img src="https://github.com/joebarnes1996/principal-modes-of-facial-variation/blob/master/images/example_features.png" width="400">
<img src="https://github.com/joebarnes1996/principal-modes-of-facial-variation/blob/master/Data/mean_image.png" width="400">

To demonstrate this technique, the below images show the mean facial image morphed in the directions of +/- 3 standard deviations from the mean in the direction of the first 5 principal components. Interestingly, I found that principal components 1 and 3 related to the vertical and horizontal positioning of the camera, while principal component 4 related to the distance with which each photo was taken, introducing a fish-eye effect. Principal components 2 and 4 and 6 onward are believed to relate purely to deviations in facial shape.

<img src="https://github.com/joebarnes1996/principal-modes-of-facial-variation/blob/master/images/mode_1.png" width="800">
<img src="https://github.com/joebarnes1996/principal-modes-of-facial-variation/blob/master/images/mode_2.png" width="800">
<img src="https://github.com/joebarnes1996/principal-modes-of-facial-variation/blob/master/images/mode_3.png" width="800">
<img src="https://github.com/joebarnes1996/principal-modes-of-facial-variation/blob/master/images/mode_4.png" width="800">
<img src="https://github.com/joebarnes1996/principal-modes-of-facial-variation/blob/master/images/mode_5.png" width="800">



