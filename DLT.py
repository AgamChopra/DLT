# Author: Agamdeep S. Chopra, 02/09/2022
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


# DLT
def compute_A(X, X_):

    A_list = []
    zero = np.zeros(3)

    for i in range(X.shape[0]):
        w = X_[i, 2]
        y = X_[i, 1]
        x = X_[i, 0]
        XT = X[i]

        A_list.append(
            np.array([[zero, -w * XT, y * XT], [w * XT, zero, -x * XT]]))

    A = np.reshape(np.asarray(A_list), (2*X.shape[0], 9))

    return A


def compute_H(A):

    _, _, Vh = np.linalg.svd(A)
    H = np.reshape(Vh[-1], (3, 3))

    return H


def denorm_H(H, T, T_):
    
    H_ = np.linalg.inv(T_) @ H @ T
    
    return H_


def DLT(X, X_, h=980, w=500, h_=366, w_=488):
    # The DLT algorithm studied in lecture 3.
    
    T = np.linalg.inv(
        np.array([[w+h, 0, w/2], [0, w+h, h/2], [0, 0, 1]]))
    T_ = np.linalg.inv(
        np.array([[w_+h_, 0, w_/2], [0, w_+h_, h_/2], [0, 0, 1]]))

    Xn, Xn_ = np.zeros(X.shape), np.zeros(X.shape) # Normalized coordinates.

    for i in range(X.shape[0]):
        Xn[i] = T @ X[i]
        Xn_[i] = T_ @ X_[i]

    A = compute_A(Xn, Xn_)
    H_norm = compute_H(A)
    H = denorm_H(H_norm, T, T_)

    return H


def cords_warp(i_max, j_max, H, w=1):
    # maps discrete(int) coordinates of the warped image to the unwarped continous(float) space.

    warped_coords = []

    for i in range(i_max):
        for j in range(j_max):
            x = np.array([i, j, w])
            x_ = H @ x
            x_ = (x_ / x_[2]) * w # Project transformed coordinates back to orignal image plane.
            warped_coords.append(x_)

    Bmap = np.reshape(np.array(warped_coords), (i_max, j_max, 3))[:, :, :-1]

    return Bmap


# interpolation
def biliniar_interpolation_back(Bmap, image):
    # create a blank canvas. out of frame pixel locations of the source image will be rendered as R,G,B = 0,0,0.
    warped_image = np.zeros((Bmap.shape[0], Bmap.shape[1], image.shape[2])) 
    
    # bilinear interp. logic.
    for i in range(Bmap.shape[0]):
        for j in range(Bmap.shape[1]):

            x, y = int(Bmap[i, j, 0]), int(Bmap[i, j, 1])

            if x > 0 and y > 0 and x < image.shape[0]-1 and y < image.shape[1]-1:
                a, b = Bmap[i, j, 0] - x, Bmap[i, j, 1] - y
                warped_image[i, j] = (1-a)*(1-b)*image[x,y] + (1-a)*(b)*image[x, y+1] + (a)*(1-b)*image[x+1, y] + (a)*(b)*image[x+1, y+1]
                    
    return warped_image.astype('int32')


def main(img_path,out_path,coords=[250, 80, 250, 350, 530, 204, 536, 284],hw=[500,500]):
    '''
    Parameters
    ----------
    img_path : str, optional
        input path, see example. The default is "C:/Users/user/Desktop/example.jpg".
    out_path : str, optional
        output path, see example. The default is "C:/Users/user/Desktop/warped".
    coords : list, optional
        List of coordinates starting from top left pt1(y1,x1), 
        top right pt2(y2,x2), bottom left pt3(y3,x3), 
        bottom right pt4(y4,x4). Use software such as GIMP to 
        pinpoint coordinates of points.(Note: by default gimp will 
        give coordinates as x:y but this code uses y:x)
        The default is [250, 80, 250, 350, 530, 204, 536, 284].

    Returns
    -------
    None.

    '''
    im = Image.open(img_path)
    #im.show()
    
    img = np.asarray(im)    
    plt.imshow(img)
    plt.show()
    
    # The 3rd coordinate of the image in 3D space where the 2D image plane lies.
    z = 1
    
    # Scaling factor alpha. We dont need a scaling factor in this problem so it is set to 1.
    alpha = 1
    
    # Coordinates of landmarks on the known unwarped image. Since we're performing backward warping,
    # this will be the target image that we want to warp our blank new image to.
    X_ = np.array([[coords[0], coords[1], z], [coords[2], coords[3], z], [coords[4], coords[5], z], [coords[6], coords[7], z]])
    w_, h_ = 894,655
    
    # Corrosponding coordinates of the landmarks in the new blank image.
    X = np.array([[0, 0, z], [0, 499, z], [499, 0, z], [499, 499, z]])
    w, h =  hw[0], hw[1]
    
    # Calculate H using the DLT algorithm.
    H = DLT(X, X_, h, w, h_, w_)
    
    # Calculate the Backward mapping.    
    Bmap = alpha * cords_warp(w, h, H, z)
    
    # Apply Backward interpolation using the Backward map to obtain the warped image.
    img_ = biliniar_interpolation_back(Bmap, img)
    
    plt.imshow(img_)
    plt.show()
    
    im_ = Image.fromarray(img_.astype('uint8'), 'RGB')
    
    im_.show() 
    im_.save(out_path) #im_.save(out_path + ".ppm")
    
    
if __name__ == '__main__':
    main('/home/agam/Downloads/ringi.jpeg','/home/agam/Downloads/pingi.jpg',[0,200,20,580,800,100,820,500],[500,500])
