import numpy as np
import cv2


def normalize(uv):
    """
    https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
    """
    uv_ = np.mean(uv, axis=0)
    u_,v_ = uv_[0], uv_[1]
    u_cap, v_cap = uv[:,0] - u_, uv[:,1] - v_

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_],[0,1,-v_],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))  #[x,y,1]
    x_norm = (T.dot(x_.T)).T     #x_ = T.x

    return  x_norm, T

def EstimateFundamentalMatrix(pts1, pts2):
    normalised = True

    x1,x2 = pts1, pts2

    ## We need atleast 8 points
    if x1.shape[0] > 7:
        if normalised == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))     #Fundamental matrix 3x3 so 9columns and min 8 points So rows>=8 columns = 9
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3) 

        ## Reducing the rank of the matrix from 3 to 2
        u, s, vt = np.linalg.svd(F)
        s = np.diag(s) # since we get singular values as a list
        s[2,2] = 0                     #Due to Noise F can be full rank i.e 3, but we need to make it rank 2 by assigning zero to last diagonal element and thus we get the epipoles

        #Reconstructing the F matrix.
        F = np.dot(u, np.dot(s, vt))
        
        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))   #This is given in algorithm for normalization
        return F

    else:
        return None