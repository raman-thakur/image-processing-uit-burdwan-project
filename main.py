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
window_max_filter = 'max_filtered_gray_image'
window_min_filter = 'min_filtered_image'
window_mean_filter = 'mean_filtered_image'
window_weighted_mean_filter = 'weighted_mean_filtered_image'
window_median_filter = 'median_filtered_image'

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

max_filtered_image_gray_level_matrix = gray_image.copy()
for i in range(image_height):
    for j in range(image_width):
        for p in range(i-int(mux//2),1+i+int(mux//2)):
            for q in range(j - int(mux // 2), 1+j + int(mux // 2)):
                if p >= 0 and p < image_height and q >= 0 and q < image_width:
                    max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
                                                                     gray_image[i - 1][j - 1])
        # if i - 1 >= 0 and j - 1 >= 0:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i - 1][j - 1])
        # if i - 1 >= 0 and j >= 0:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i - 1][j])
        # if i - 1 >= 0 and j + 1 < image_width:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i - 1][j + 1])
        # if i >= 0 and j - 1 >= 0:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i][j - 1])
        # if i >= 0 and j + 1 < image_width:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i][j + 1])
        # if i + 1 < image_height and j - 1 >= 0:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i + 1][j - 1])
        # if i + 1 < image_height and j >= 0:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i + 1][j])
        # if i + 1 < image_height and j + 1 < image_width:
        #     max_filtered_image_gray_level_matrix[i][j] = max(max_filtered_image_gray_level_matrix[i][j],
        #                                                      gray_image[i + 1][j + 1])

mse_max_filter = mse_after_noise_removal(gray_image, max_filtered_image_gray_level_matrix,image_height,image_width)
print("mse for max filter =" ,mse_max_filter)
msnr_max_filter = msnr_after_noise_removal(gray_image, max_filtered_image_gray_level_matrix,image_height,image_width)
print("msnr for max filter =" ,msnr_max_filter)
cv2.imshow(window_max_filter, max_filtered_image_gray_level_matrix)

min_filtered_image_gray_level_matrix = gray_image.copy()
for i in range(image_height):
    for j in range(image_width):
        for p in range(i-int(mux//2),1+i+int(mux//2)):
            for q in range(j - int(mux // 2), 1+j + int(mux // 2)):
                if p >= 0 and p < image_height and q >= 0 and q < image_width:
                    max_filtered_image_gray_level_matrix[i][j] = min(max_filtered_image_gray_level_matrix[i][j],
                                                                     gray_image[i - 1][j - 1])
# for i in range(image_height):
#     for j in range(image_width):
#         if i - 1 >= 0 and j - 1 >= 0:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i - 1][j - 1])
#         if i - 1 >= 0 and j >= 0:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i - 1][j])
#         if i - 1 >= 0 and j + 1 < image_width:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i - 1][j + 1])
#         if i >= 0 and j - 1 >= 0:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i][j - 1])
#         if i >= 0 and j + 1 < image_width:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i][j + 1])
#         if i + 1 < image_height and j - 1 >= 0:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i + 1][j - 1])
#         if i + 1 < image_height and j >= 0:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i + 1][j])
#         if i + 1 < image_height and j + 1 < image_width:
#             min_filtered_image_gray_level_matrix[i][j] = max(min_filtered_image_gray_level_matrix[i][j],
#                                                              gray_image[i + 1][j + 1])

mse_min_filter = mse_after_noise_removal(gray_image, min_filtered_image_gray_level_matrix,image_height,image_width)
print("mse for min filter =" ,mse_min_filter)
msnr_min_filter = msnr_after_noise_removal(gray_image, min_filtered_image_gray_level_matrix,image_height,image_width)
print("msnr for min filter =" ,msnr_min_filter)
cv2.imshow(window_min_filter, min_filtered_image_gray_level_matrix)

mean_filtered_image_gray_level_matrix = gray_image.copy()
for i in range(image_height):
    for j in range(image_width):
        average = 0
        for p in range(i-int(mux//2),1+i+int(mux//2)):
            for q in range(j - int(mux // 2), 1+j + int(mux // 2)):
                if p >= 0 and p < image_height and q >= 0 and q < image_width:
                    average += int(gray_image[p][q])
        mean_filtered_image_gray_level_matrix[i][j] = average//(mux*mux)
# for i in range(image_height):
#     for j in range(image_width):
#         average = int(gray_image[i][j])
#         if i - 1 >= 0 and j - 1 >= 0:
#             average += int(gray_image[i - 1][j - 1])
#         if i - 1 >= 0 and j >= 0:
#             average += int(gray_image[i - 1][j])
#         if i - 1 >= 0 and j + 1 < image_width:
#             average += int(gray_image[i - 1][j + 1])
#         if i >= 0 and j - 1 >= 0:
#             average += int(gray_image[i][j - 1])
#         if i >= 0 and j + 1 < image_width:
#             average += int(gray_image[i][j + 1])
#         if i + 1 < image_height and j - 1 >= 0:
#             average += int(gray_image[i + 1][j - 1])
#         if i + 1 < image_height and j >= 0:
#             average += int(gray_image[i + 1][j])
#         if i + 1 < image_height and j + 1 < image_width:
#             average += int(gray_image[i + 1][j + 1])
#         average = average // 9
#         mean_filtered_image_gray_level_matrix[i][j] = average

mse_mean_filter = mse_after_noise_removal(gray_image, mean_filtered_image_gray_level_matrix,image_height,image_width)
print("mse for mean filter =" ,mse_mean_filter)
msnr_mean_filter = msnr_after_noise_removal(gray_image, mean_filtered_image_gray_level_matrix,image_height,image_width)
print("msnr for mean filter =" ,msnr_mean_filter)
cv2.imshow(window_mean_filter, mean_filtered_image_gray_level_matrix)


#here we took the generally used weight 1 2 1 , 1 10 1 , 1 2 1
weighted_mean_filtered_image_gray_level_matrix = gray_image.copy()
for i in range(image_height):
    for j in range(image_width):
        average = 0
        added_pixel=0
        for p in range(i-int(mux//2),1+i+int(mux//2)):
            for q in range(j - int(mux // 2), 1+j + int(mux // 2)):
                if p >= 0 and p < image_height and q >= 0 and q < image_width:
                    if p==i and q==j:
                        average += (10*int(gray_image[p][q]))
                        added_pixel += 10
                    elif p==i or q==j:
                        average += (2 * int(gray_image[p][q]))
                        added_pixel += 2
                    else:
                        average += int(gray_image[p][q])
                        added_pixel += 1
        weighted_mean_filtered_image_gray_level_matrix[i][j] = average//(added_pixel)

mse_weighted_mean_filter = mse_after_noise_removal(gray_image, weighted_mean_filtered_image_gray_level_matrix,image_height,image_width)
print("mse for weighted mean filter =" ,mse_weighted_mean_filter)
msnr_weighted_mean_filter = msnr_after_noise_removal(gray_image, weighted_mean_filtered_image_gray_level_matrix,image_height,image_width)
print("msnr for weighted mean filter =" ,msnr_weighted_mean_filter)
cv2.imshow(window_weighted_mean_filter, weighted_mean_filtered_image_gray_level_matrix)

median_filtered_image_gray_level_matrix = gray_image.copy()
for i in range(image_height):
    for j in range(image_width):
        neighborhood_gray_array = []
        neighborhood_count = 0
        for p in range(i-int(mux//2),1+i+int(mux//2)):
            for q in range(j - int(mux // 2), 1+j + int(mux // 2)):
                if p >= 0 and p < image_height and q >= 0 and q < image_width:
                    neighborhood_gray_array.append(int(gray_image[p][q]))
                    neighborhood_count += 1
        neighborhood_gray_array.sort()
        if (neighborhood_count % 2) == 1:
            median_filtered_image_gray_level_matrix[i][j] = neighborhood_gray_array[neighborhood_count//2]
        else:
            median_filtered_image_gray_level_matrix[i][j] = (neighborhood_gray_array[neighborhood_count//2] + neighborhood_gray_array[(neighborhood_count//2)-1])//2

mse_median_filter = mse_after_noise_removal(gray_image, median_filtered_image_gray_level_matrix,image_height,image_width)
print("mse for median filter =" ,mse_median_filter)
msnr_median_filter = msnr_after_noise_removal(gray_image, median_filtered_image_gray_level_matrix,image_height,image_width)
print("msnr for median filter =" ,msnr_median_filter)
cv2.imshow(window_median_filter, median_filtered_image_gray_level_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()
