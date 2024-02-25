import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def sum_of_squared_diff(pixel_vals_1, pixel_vals_2):

    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1
    
    return np.sum((pixel_vals_1 - pixel_vals_2)**2)

def block_comparison(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):
    """Block comparison function used for comparing windows on left and right images and find the minimum value ssd match the pixels"""
    
    # Get search range for the right image

    ## Basically searches for -50 and +50 in x and -1 to +1 in y
    ## We made the epipolar lines parallel and thus we don't have to have to search in the y much
    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)
    
    first = True
    min_ssd = None
    min_index = None

    ## Searching the corresponding pixels in second image
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            block_right = right_array[y: y+block_size, x: x+block_size]
            # print(block_right.shape)
            # print(y+block_size)
            ssd = sum_of_squared_diff(block_left, block_right)
            if first:
                min_ssd = ssd
                min_index = (y, x)
                first = False
            else:
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)

    return min_index


def ssd_correspondence(img1, img2):
    """Correspondence applied on the whole image to compute the disparity map and finally disparity map is scaled"""


    block_size = 15 
    x_search_block_size = 50 
    y_search_block_size = 1
    height, width = img1.shape
    disparity_map = np.zeros((height, width))

    print('Doing Block Matching ......')


    ## Basically we are iterating over each pixel, getting a window with the (y,x) as top left.
    for y in tqdm(range(block_size, height-block_size)):
        for x in range(block_size, width-block_size):
            block_left = img1[y:y + block_size, x:x + block_size]
            index = block_comparison(y, x, block_left, img2, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index[1] - x)


    # Normalizing the disparity map.
    disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))

    plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    plt.savefig('images/disparity_image_heat.png')

    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    plt.savefig('images/disparity_image_gray.png')

    return disparity_map_int
