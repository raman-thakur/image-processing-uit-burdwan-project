import random

# importing openCV
import cv2
import numpy as np

path = r'C:\Users\suman\Downloads\imgonline-com-ua-resize-7ysuwJ1kgHQBo.jpg'
# reading an image
image = cv2.imread(path)

# converting colour image to gray image
gray_image_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# adding noise function for salt and paper
def add_noise_sap(img,row,col):
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(row+col, (row*col)//50)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(row+col, (row*col)//50)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img

# adding noise function for gaussian noise
def add_gaussian_noise(img,row,col):
    gauss_noise = np.zeros((row, col), dtype=np.uint8)
    cv2.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
    gn_img = cv2.add(img, gauss_noise)
    return gn_img

# function for finding mean square error MSE
def mse_after_noise_removal(noisy_image,noise_removed_image,row,col):
    mse = 0
    for i in range(row):
        for j in range(col):
            mse += ((pow((int(noisy_image[i][j]) - int(noise_removed_image[i][j])), 2))/(row * col))
    return mse

def msnr_after_noise_removal(noisy_image,noise_removed_image,row,col):
    x = 0
    y = 0
    for i in range(row):
        for j in range(col):
            x += ((pow(int(noisy_image[i][j]), 2)))
    for i in range(row):
        for j in range(col):
            y += ((pow((int(noisy_image[i][j]) - int(noise_removed_image[i][j])), 2)))
    if y!=0:
        return x/y
    else:
        return -1

#taking mux size input
mux = 3

#defining windows for output console
window_gray = 'orignal_gray_image'
window_gray_noisy = 'orignal_noisy_gray_image'
window_max_filter = 'median_filtered_gray_image'

# storing the image matrix, height and width
image_gray_level_matrix = gray_image_original.copy()
image_height = len(image_gray_level_matrix)
image_width = len(image_gray_level_matrix[0])

# gray image after adding noise
gray_image = add_noise_sap(gray_image_original,image_height,image_width)

# showing original gray image
cv2.imshow(window_gray, image_gray_level_matrix)

# showing original noisy image
cv2.imshow(window_gray_noisy, gray_image)

adaptive_median_filtered_image_gray_level_matrix = gray_image.copy()
for i in range(image_height):
    for j in range(image_width):
        while 1:
            neighborhood_gray_array = []
            neighborhood_count = 0
            for p in range(i - int(mux // 2), 1 + i + int(mux // 2)):
                for q in range(j - int(mux // 2), 1 + j + int(mux // 2)):
                    if p >= 0 and p < image_height and q >= 0 and q < image_width:
                        neighborhood_gray_array.append(int(gray_image[p][q]))
                        neighborhood_count += 1
            neighborhood_gray_array.sort()
            # neighborhood_median_gray_value=neighborhood_gray_array[neighborhood_count//2];
            if (neighborhood_count % 2) == 1:
                neighborhood_median_gray_value = neighborhood_gray_array[neighborhood_count // 2]
            else:
                neighborhood_median_gray_value = (neighborhood_gray_array[neighborhood_count // 2] +
                                                  neighborhood_gray_array[(neighborhood_count // 2) - 1]) // 2

            neighborhood_max_gray_value = neighborhood_gray_array[neighborhood_count - 1];
            neighborhood_min_gray_value = neighborhood_gray_array[0];
            a1 = neighborhood_median_gray_value - neighborhood_min_gray_value;
            a2 = neighborhood_median_gray_value - neighborhood_max_gray_value;
            b1 = gray_image[i][j] - neighborhood_min_gray_value;
            b2 = gray_image[i][j] - neighborhood_max_gray_value;
            if (a1 > 0 and a2 < 0):
                if (b1 > 0 and b2 < 0):
                    adaptive_median_filtered_image_gray_level_matrix[i][j] = gray_image[i][j];
                    mux=3;
                    break
                else:
                    adaptive_median_filtered_image_gray_level_matrix[i][j] = neighborhood_median_gray_value;
                    mux=3;
                    break
            else:
                if(gray_image[i][j]>=neighborhood_max_gray_value):
                    adaptive_median_filtered_image_gray_level_matrix[i][j] = gray_image[i][j];
                    mux = 3;
                    break
                else:
                    mux=mux+2;


mse_adaptive_median_filter = mse_after_noise_removal(gray_image, adaptive_median_filtered_image_gray_level_matrix,image_height,image_width)
print("mse for weighted mean filter =" ,mse_adaptive_median_filter)
msnr_adaptive_median_ilter = msnr_after_noise_removal(gray_image, adaptive_median_filtered_image_gray_level_matrix,image_height,image_width)
print("msnr for weighted mean filter =" ,msnr_adaptive_median_ilter)

cv2.imshow(window_max_filter, adaptive_median_filtered_image_gray_level_matrix)

cv2.waitKey(0)
cv2.destroyAllWindows()
