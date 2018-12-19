import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import imageio
import math
from skimage.feature import peak_local_max

class ImageProcessing(object):

    def __init__(self, img):
        self.image = img
        self.peak_coordinates = None

    def get_image(self):
        return self.image

    def get_coordinates(self):
        if self.peak_coordinates is not None:
            return self.peak_coordinates
        else:
            raise Exception("Coordinates has not been calculated")

    def binarize_otsu(self):
        ret, binaryimage = cv.threshold(self.image, 0, 255, cv.THRESH_OTSU)
        self.image = binaryimage

    def draw_rectangle(self, p1, p2):
        cv.rectangle(self.image, p1, p2, (0, 255, 0), 10)

    def convert_to_rgb(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)

    def convert_to_gray(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def dilate(self, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation = cv.erode(self.image, kernel, iterations = 1)
        self.image = dilation

    def correlate(self, kernel):
        # following lines are for debug purposes
        # correlation =  signal.correlate2d(in1=self.image,
        #                                   in2=kernel,
        #                                   mode='full')
        # imageio.imsave('korelacja.jpg', correlation)
        correlation = cv.matchTemplate(kernel, self.image, method=cv.TM_CCORR_NORMED)
        self.image = correlation
        return correlation

    def find_peaks(self):
        distance = int(np.size(self.image, 0)/13)
        peaks = peak_local_max(self.image,
                                min_distance=1,
                                threshold_rel = 0.7)
        peak_coordinates = []
        for peak in peaks:
            min_distance = self.min_distance_between_peaks(peak_coordinates, peak)
            if (min_distance > distance):
                peak_coordinates.append(peak)
        self.peak_coordinates = np.array(peak_coordinates)

    def count_distance_between_peaks(self, peak_1, peak_2):
        distance = math.sqrt(math.pow(peak_2[0] - peak_1[0], 2) + math.pow(peak_2[1] - peak_1[1], 2))
        return distance

    def min_distance_between_peaks(self, peaks_list, peak_to_check):
        min_distance = 1000
        for peak in peaks_list:
            distance = self.count_distance_between_peaks(peak, peak_to_check)
            if (distance < min_distance):
                min_distance = distance
        return min_distance
        
    def plot_image(self):
        plt.plot()
        plt.imshow(self.image, 'gray')
        plt.title('Number of elements: ' + str(len(self.peak_coordinates)))
        # plt.xticks([]),plt.yticks([])
        plt.plot(self.peak_coordinates[:, 1], self.peak_coordinates[:, 0], 'r.')
        plt.show()

    def plot_custom_image(self, img, title):
        plt.plot()
        plt.imshow(img, 'gray')
        plt.title(title)
        plt.show()

    def display_two_images(self, imgRight):
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(self.image, 'gray')
        a.set_title('Before')
        plt.imshow(imgRight)
        a.set_title('After')

    def display(self, img):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img, 'gray')
        ax1.set_title('Original image')
        ax2.imshow(self.image, 'gray')
        plt.plot(self.peak_coordinates[:, 1], self.peak_coordinates[:, 0], 'r.')
        ax2.set_title('Processed image, number of elements: ' + str(len(self.peak_coordinates)))
        plt.show()
        


# class ImageOperations(object):
#     def display(self, image):
#         plt.plot()
#         plt.imshow(image, 'gray')
#         plt.title("ELO")
#         # plt.xticks([]),plt.yticks([])
#         plt.show()
