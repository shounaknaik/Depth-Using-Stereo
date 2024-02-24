import cv2
import numpy as np


def stereo_rectification(img1,img2,keypoints1,keypoints2,F):
    '''
    For reason and how cv2 does rectification, see the following link
    https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf
    '''


    # Stereo rectification
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(keypoints1), np.float32(keypoints2), F, imgSize=(w1, h1))

    # Warp the image to rectify them.
    rectified_img1 = cv2.warpPerspective(img1,H1,(w1,h1))
    rectified_img2 = cv2.warpPerspective(img2,H2,(w2,h2))

    cv2.imwrite("./images/rectified1.png",rectified_img1)
    cv2.imwrite("./images/rectified2.png",rectified_img2)
    
    # Rectify the feature points
    rectified_pts1 = np.zeros((keypoints1.shape), dtype=int)
    rectified_pts2 = np.zeros((keypoints1.shape), dtype=int)

    
    for i in range(keypoints1.shape[0]):
        source1 = np.array([keypoints1[i][0], keypoints1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0]/new_point1[2])
        new_point1[1] = int(new_point1[1]/new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        rectified_pts1[i] = new_point1

        source2 = np.array([keypoints2[i][0], keypoints2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0]/new_point2[2])
        new_point2[1] = int(new_point2[1]/new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        rectified_pts2[i] = new_point2

    return rectified_img1,rectified_img2,rectified_pts1,rectified_pts2