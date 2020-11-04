import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


#=============================================================================
"""
First run Image_warping_functions.py to get the necessary functions
"""






#=============================================================================
"""
1. Load features and convert to array
"""
#=============================================================================
# set the path to the data
path = r'C:\Users\joeba\github_projects\PCA_faces'

# load the data
facial_features = {}

for i in range(343):
    
    facial_features[i] = np.loadtxt(path + '/Data/{}.txt'.format(i))



# load the mean facial images
mean_image = cv2.imread(path + '\Data\mean_image.png')
mean_image = cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB)

mean_features = np.loadtxt(path + '/Data/mean_features.txt').astype(int)

# plot the mean features
plt.figure(figsize=(6, 8))
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid()
plt.scatter(mean_features[:,0], mean_features[:,1])
plt.xlabel('x-position')
plt.ylabel('y-position')
plt.savefig(path + '/images/example_features')
plt.show()



# convert all features into an array, X
X = dictToArray(facial_features)



#=============================================================================
# standardise the array
mean = X.mean()
std  = X.std()

# import stuff for PCA
from sklearn.preprocessing import StandardScaler

# standardise the data
sc = StandardScaler()
Z = sc.fit_transform(X)





#=============================================================================

#=============================================================================
"""
2. Perform PCA to the data
"""
#=============================================================================
# perform PCA on the standardised data
from sklearn.decomposition import PCA

pca = PCA()
dataRed = pca.fit_transform(Z)
vals = pca.explained_variance_ratio_
vecs = pca.components_
   







#=============================================================================
"""
3. Define function to assess modes of variation
"""
#=============================================================================
# create function to unflatten an array
def unFlatten(array):
    
    # intialise array
    height = int(len(array) / 2)    
    arrayOut = np.reshape(array, (height, 2))
    
    return arrayOut

#=============================================================================
# create function to flatten an array
def flatten(array):
    
    arrayOut = array.flatten()
    
    return arrayOut

#=============================================================================
# create a function to find the modes of variation
def modeOfVariation(mode, deviation=3):
    
    mode -= 1
    
    # get the parameters
    vec = vecs[mode]

    # get high and low shapes in PCA space
    highPCA = np.zeros(dataRed.shape[1])
    highPCA[mode] = deviation * dataRed.std(axis=0)[mode]
    
    lowPCA = np.zeros(dataRed.shape[1])
    lowPCA[mode] = - deviation * dataRed.std(axis=0)[mode]
    
    
    # convert from PCA space to standardised space
    highStd = np.matmul(highPCA, vecs)
    lowStd  = np.matmul(lowPCA,  vecs)
    
    # unstandardise
    high = highStd * X.std(axis=0) + X.mean(axis=0)
    low  = lowStd  * X.std(axis=0) + X.mean(axis=0)
    
    # convert to regular shape
    highShape = unFlatten(high).astype(int)
    lowShape  = unFlatten(low).astype(int)
    
    # get the images
    highIm = morphFace(mean_image, mean_features, highShape)
    lowIm =  morphFace(mean_image, mean_features, lowShape)
    
    
    return highIm, lowIm














#=============================================================================
"""
4. Visualise how the face varies in shape


NOTE - am demonstrating how the face varies over the first principal 
        component at +- 3 standard deviations from the mean.

"""

# find images for the first 5 modes of variation
for mode in range(1, 6):
    
    # get the high and low images
    high_image, low_image = modeOfVariation(mode, deviation=3)
    
    # visualise the results
    fig, axs = plt.subplots(1, 3, figsize=(15,8))
    axs[0].imshow(high_image)
    axs[0].set_title(r'$\mu+3\sqrt{\lambda_i}\varphi_i$', fontsize=20)
    
    axs[1].imshow(mean_image)
    axs[1].set_title(r'$\mu$', fontsize=20)
    
    
    axs[2].imshow(low_image)
    axs[2].set_title(r'$\mu-3\sqrt{\lambda_i}\varphi_i$', fontsize=20)
    
    fig.suptitle('i = {}'.format(mode), fontsize=30)
    plt.tight_layout()
    
    # save figure
    os.chdir(r'C:\Users\joeba\github_projects\PCA_faces')
    plt.savefig(path + '/images/mode_{}'.format(mode))
    
    plt.show()

















