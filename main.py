import math
import cv2
import random as random
import numpy as np  
import matplotlib.pyplot as plt

random.seed(42)
from fundamental_matrix import EstimateFundamentalMatrix
from essentialmatrix import getEssentialMatrix
from pose_disambiguation import ExtractCameraPose, DisambiguatePose
from lineartriangulation import LinearTriangulation
from rectification import stereo_rectification
from lineUtils import drawlines
from block_matching import ssd_correspondence

def estimate_F_matrix(keypoints1,keypoints2):
    '''
    Using Ransac along with the 8 point algorithm to get the fundamental matrix.
    '''
    pairs = list(zip(keypoints1, keypoints2))  
    # atleast 20 inliers needs to be there
    max_inliers = 20
    threshold = 0.02

    # print(keypoints1)
    # print(keypoints2)

    num_iters=1000
    print('Doing RANSAC ...')
    for i in range(num_iters):

        #Choosing 8 points randomly from the keypoint matches
        random_pairs = random.sample(pairs,8)
        # print(random_pairs)
        pts1,pts2 = zip(*random_pairs)
        pts1=np.array(pts1)
        pts2=np.array(pts2)
        F = EstimateFundamentalMatrix(pts1,pts2)

        inliers_img1=[]
        inliers_img2=[]
        # break
        for j in range(len(keypoints1)):

            img1_point = np.array([keypoints1[j][0], keypoints1[j][1],1])
            img2_point = np.array([keypoints2[j][0], keypoints2[j][1],1])
            distance = abs(np.dot(img2_point.T,np.dot(F,img1_point)))

            if distance < threshold:
                inliers_img1.append(keypoints1[j])
                inliers_img2.append(keypoints2[j])
            
        if len(inliers_img1) > max_inliers:
            # max_inliers = len(
            Best_F = F
            max_inliers = len(inliers_img1)
            # print(j)
            # break

    # print(Best_F)
    return Best_F
        
  
def get_draw_matches(img1,img2):
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1,descriptors2)

    # Lower the distance between descriptor better the match thus we are choosing first 75
    matches = sorted(matches, key = lambda x :x.distance)
    chosen_matches = matches[0:30]

    # Draw keypoints
    img_with_keypoints = cv2.drawMatches(img1,keypoints1,img2,keypoints2,chosen_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("./images/images_with_matching_keypoints.png", img_with_keypoints)

    # Getting x,y coordinates of the matches
    # Img0 will be query image and Img1 will be train image.
    list_keypoints1 = [list(keypoints1[match.queryIdx].pt) for match in chosen_matches] 
    list_keypoints2 = [list(keypoints2[match.trainIdx].pt) for match in chosen_matches]

    return list_keypoints1, list_keypoints2

def main():
    # number = int(input("Please enter the dataset number (1/2/3) to use for calculating the depth map\n"))

    number = 1
    img1 = cv2.imread(f"./Data/Project3/Dataset {number}/im0.png", 0)
    img2 = cv2.imread(f"./Data/Project3/Dataset {number}/im1.png", 0)

    width = int(img1.shape[1]* 0.3) # 0.4
    height = int(img1.shape[0]* 0.3) # 0.4

    img1 = cv2.resize(img1, (width, height), interpolation = cv2.INTER_AREA)
    # cv2.imshow('img1',img1)
    img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)
    # cv2.imshow('img2',img2)
    # cv2.waitKey(0)
    
    #__________________Camera Parameters -- taken from the Data folder.________________________________
    K11 = np.array([[5299.313,  0,   1263.818], 
                [0,      5299.313, 977.763],
                [0,          0,       1   ]])
    K12 = np.array([[5299.313,   0,    1438.004],
                [0,      5299.313,  977.763 ],
                [0,           0,      1     ]])

    K21 = np.array([[4396.869, 0, 1353.072],
                    [0, 4396.869, 989.702],
                    [0, 0, 1]])
    K22 = np.array([[4396.869, 0, 1538.86],
                [0, 4396.869, 989.702],
                [0, 0, 1]])
    
    K31 = np.array([[5806.559, 0, 1429.219],
                    [0, 5806.559, 993.403],
                    [ 0, 0, 1]])
    K32 = np.array([[5806.559, 0, 1543.51],
                    [ 0, 5806.559, 993.403],
                    [ 0, 0, 1]])
    camera_params = [(K11, K12), (K21, K22), (K31, K32)]

    keypoints1,keypoints2=get_draw_matches(img1,img2)

    # print(keypoints1)

    F_estimated=estimate_F_matrix(keypoints1,keypoints2)
    K1,K2 = camera_params[number-1]
    # print(K1,K2)
    print("Fundamental Matrix")
    print(F_estimated)

    Essential_matrix = getEssentialMatrix(K1,K2,F_estimated)
    print("Essential Matrix")
    print(Essential_matrix)
    
    ## Get Pose of Camera
    R_set,C_set = ExtractCameraPose(Essential_matrix)
    # print(C_set,R_set)

    # Get Linearly triangulated points
    # There will be 4 sets of points corresponding to each pose
    C1=np.zeros((3,1))
    R1=np.identity(3)
    points3D_4 =[]
    for i in range(len(C_set)):
        points3D = LinearTriangulation(K1,K2,C1, R1, C_set[i], R_set[i], keypoints1, keypoints2) 
        points3D_4.append(points3D)

    # print(points3D_4)

    ## Disambiguate Camera Poses by Depth Chirality
    R_best,C_best,points_3D_best = DisambiguatePose(R_set,C_set,points3D_4)
    # print(R_best,C_best)

    ## Rectification Step
    # print(F_estimated)
    rectified_img1,rectified_img2,rectified_pts1,rectified_pts2 = stereo_rectification(img1,img2,np.array(keypoints1),np.array(keypoints2),F_estimated)

    # Visualize Epipolar lines
    # In image1
    lines1 = cv2.computeCorrespondEpilines(rectified_pts2.reshape(-1, 1, 2), 2, F_estimated)
    lines1 = lines1.reshape(-1, 3)
    img1color, _= drawlines(rectified_img1, rectified_img2, lines1, rectified_pts1, rectified_pts2)

    # In image2
    lines2 = cv2.computeCorrespondEpilines(rectified_pts1.reshape(-1, 1, 2), 2, F_estimated)
    lines2 = lines2.reshape(-1, 3)
    img2color, _= drawlines(rectified_img2, rectified_img2, lines2, rectified_pts2, rectified_pts1)

    cv2.imwrite("./images/left_image.png", img1color)
    cv2.imwrite("./images/right_image.png", img2color)



    #------------------------Correspondance --------------------------#

    disparity_map_int= ssd_correspondence(rectified_img1, rectified_img2)


    # ------------------------Depth-----------------------------------#

    baseline1, f1 = 177.288, 5299.313
    baseline2, f2 = 144.049, 4396.869
    baseline3, f3 = 174.019, 5806.559
    
    params = [(baseline1, f1), (baseline2, f2), (baseline3, f3)]
    baseline, f = params[number-1]

    depth = (baseline * f) / (disparity_map_int + 1e-10)
    depth[depth > 100000] = 100000

    depth_map = np.uint8(depth * 255 / np.max(depth))
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('images/depth_hot.png')
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('./images/depth_gray.png')


if __name__ == "__main__":
    main()
    