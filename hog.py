import cv2
import numpy as np
import os

Config = {
    'Image_resize': (64, 64), # (width, height)
    'Image_convert': cv2.COLOR_BGR2GRAY, # cv2.COLOR_BGR2GRAY | cv2.COLOR_BGR2RGB | cv2.COLOR_BGR2HSV
    'Cell_size': (32, 32), # (width, height),
    'Magnitude_threshold': 40, # minimum magnitude limit,
    'n_bins': 18, # n_bins
}

def get_gradient(image):

    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_orientation = np.degrees(np.arctan2(gradient_y, gradient_x)) % 180

    return gradient_magnitude, gradient_orientation

def show_magnitude(IMAGE_PATH, image_name):
    image = cv2.imread(os.path.join(IMAGE_PATH, image_name))
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gradient_magnitude, gradient_orientation = get_gradient(gray)

    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    gradient_magnitude_colormap = cv2.applyColorMap(gradient_magnitude_normalized, cv2.COLORMAP_JET)

    cv2.imshow('Gradient Magnitude', gradient_magnitude_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return   

def get_histogram(magnitude, orientation, mag_threshold, n_bins):
    max_degree = 180
    degree_axis = list(range(0, max_degree, int(180 / n_bins)))
    histogram = [0] * len(degree_axis)
    cell_size = magnitude.shape
    for x in range(cell_size[0]):
        for y in range(cell_size[1]):
            if magnitude[x][y] < mag_threshold:
                continue
            index = int(orientation[x][y]//(180/n_bins))
            histogram[index] += magnitude[x][y]
    return histogram

def get_hog(IMAGE_PATH, image_name):
    image = cv2.imread(os.path.join(IMAGE_PATH, image_name))
    image = cv2.resize(image, Config['Image_resize'])

    image = cv2.cvtColor(image, Config['Image_convert'])

    cellSize = Config['Cell_size']
    mag_threshold = Config['Magnitude_threshold']
    n_bins = Config['n_bins']

    hog = []
    for x in range(0, image.shape[0], cellSize[0]):
        for y in range(0, image.shape[1], cellSize[1]):
            x_start = x
            y_start = y
            x_end = x + cellSize[0]
            y_end = y + cellSize[1]
            cell = image[x_start:x_end, y_start:y_end]
            magnitude, orientation = get_gradient(image=cell)
            histogram = get_histogram(magnitude, orientation, mag_threshold, n_bins)
            hog += histogram
            """
            print(list(histogram))
            cv2.imshow('cell image', cell)
            cv2.waitKey(0)
            """
    return hog

def get_config_string():
    config = str(Config).replace(' ', '')
    return config

def show_vector(IMAGE_PATH, image_name):
    image = cv2.imread(os.path.join(IMAGE_PATH, image_name), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))

    magnitude, orientation = get_gradient(image)
    magnitude = cv2.normalize(magnitude, None, 0, 20, cv2.NORM_MINMAX)

    gradient_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = 1 

    for i in range(0, gradient_vis.shape[0], 10):
        for j in range(0, gradient_vis.shape[1], 10):
            x = j
            y = i
            angle = orientation[i, j]
            mag = int(magnitude[i, j])
            dx = int(scale * mag * np.cos(angle))
            dy = int(scale * mag * np.sin(angle))
            cv2.arrowedLine(gradient_vis, (x, y), (x+dx, y+dy), (0, 0, 255), 1)
            
    cv2.imshow("Gradient Visualization", gradient_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(get_hog('./data/image/temp', 'h_3f507d02.jpg'))
    show_magnitude('./data/image/temp', 'white.jpg')
    show_vector('./data/image/temp', 'h_3f507d02.jpg')
    # show_magnitude('./data/image/temp', 'h_5d303feb.jpg')