import cv2
import numpy as np

def drawlines(img1, img2, lines, pts1src, pts2src):
    """This fucntion is used to visualize the epipolar lines on the images
        img1 - image on which we draw the epipolar lines for the points in img2
        lines - corresponding epilines """
    height, width = img1.shape

    ## Colorizing the image again
    img1color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)


    # Use the same random seed so that two images are comparable!
    np.random.seed(42)
    for line, pt1, pt2 in zip(lines, pts1src, pts2src):

        #Choose a random color
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # line is ax+by+c=0
        # Point 1 is (0,-c/b)
        #Point 2 is (width, -(c+a*width)/b)

        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [width, -(line[2]+line[0]*width)/line[1]])

        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        # print(pt1)
        img1color = cv2.circle(img1color, tuple(map(int, pt1)), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(map(int, pt2)), 5, color, -1)
    
    return img1color, img2color