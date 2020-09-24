"""
Accreditted to:
    
    https://github.com/wuhuikai/FaceSwap

"""

import cv2
import numpy as np
import scipy.spatial as spatial
import logging



#=============================================================================
""" 1. Examine warp_image_3d """
#=============================================================================
def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    
    rows, cols = dst_shape[:2]                                                  # does not mask
    result_img = np.zeros((rows, cols, 3), dtype=dtype)                         # does not mask

    delaunay = spatial.Delaunay(dst_points)                                     # does not mask
    
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))                           # does not mask

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)        # INVESTIGATING

    return result_img
#=============================================================================



#=============================================================================
""" 2. process_warp """
#=============================================================================
def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)                                 # POSSIBLY - generates a load of points within bounding box of features
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)                         # will only show the masked image within these coords

    for simplex_index in range(len(delaunay.simplices)):
        
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None




#=============================================================================
""" 3. triangular_affine_matrices """
#=============================================================================
def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat




#=============================================================================
""" 4. grid_coordinates """
#=============================================================================
def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)







#=============================================================================
""" 5. bilinear_interpolate """
#=============================================================================
def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T




#=============================================================================
""" 6. Border points """
#=============================================================================
def addBorderPoints(features, xMax=875, yMax=1250):
    
    # add an extra eight rows to the features
    featuresOut = np.zeros((features.shape[0]+ 8, features.shape[1]))
    featuresOut[:68] = features.copy()
    
    # add the additional features for the border poitns
    featuresOut[68:] = np.array([[0,0],
                                 [0, int(yMax/2)],
                                 [0, yMax-5],
                                 [xMax-5, 0],
                                 [xMax-5, int(yMax/2)],
                                 [xMax-5, yMax-5],
                                 [int(xMax/2), 0],
                                 [int(xMax/2), yMax-5]])
    
    # convert to integer
    featuresOut = featuresOut.astype(int)
    
    return featuresOut




#=============================================================================
""" 7. Compile functions"""
#=============================================================================
def morphFace(imgSrc, ftsSrc, ftsDst, borders=True):
    
    if borders == True:

        # add borders to features
        ftsSrc = addBorderPoints(ftsSrc)
        ftsDst = addBorderPoints(ftsDst)
    
    else:
        
        ftsSrc = ftsSrc.astype(int)
        ftsDst = ftsDst.astype(int)
            
    # warp the image
    imageWarped = warp_image_3d(imgSrc, ftsSrc, ftsDst, imgSrc.shape)
    
    return imageWarped
    
    
    
    
    
    
# convert dictionary to an array


def dictToArray(dictionary):
    
    # initialise an empty array of zeros
    array = np.zeros((len(dictionary),
                      dictionary[0].shape[0] * dictionary[0].shape[1]))
    
    # replace each row of the array with each flatten image in dictionary
    for i in range(len(dictionary)):
        
        array[i] = dictionary[i].flatten()
        
    return array


