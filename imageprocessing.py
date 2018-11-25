import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import imageio
from skimage.feature import peak_local_max

class ImageProcessing(object):

    def __init__(self, img):
        self.image = img
        self.peak_coordinates = None

    def get_image(self):
        return self.image

    def get_coordinates(self):
        if self.peak_coordinatess is not None:
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
        # following lines are for debug purposesS
        # correlation =  signal.correlate2d(in1=self.image,
        #                                   in2=kernel,
        #                                   mode='full')
        # imageio.imsave('korelacja.jpg', correlation)
        correlation = cv.matchTemplate(kernel, self.image, method=cv.TM_CCORR_NORMED)
        self.image = correlation
        return correlation

    def find_peaks(self):
        self.peak_coordinates = peak_local_max(self.image,
                                           min_distance=1,
                                           threshold_rel = 0.7)
        
    def plot_image(self):
        plt.plot()
        plt.imshow(self.image, 'gray')
        plt.title("TEST")
        # plt.xticks([]),plt.yticks([])
        plt.plot(self.peak_coordinates[:, 1], self.peak_coordinates[:, 0], 'r.')
        plt.show()

# class ImageOperations(object):
#     def display(self, image):
#         plt.plot()
#         plt.imshow(image, 'gray')
#         plt.title("ELO")
#         # plt.xticks([]),plt.yticks([])
#         plt.show()
