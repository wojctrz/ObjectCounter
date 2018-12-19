import imageprocessing as ImageEvents
import cv2 as cv


img = cv.imread('testimages/realtest2.jpg', 0)
kernel = cv.imread('testimages/kernelreal4.jpg', 0)

kernproc = ImageEvents.ImageProcessing(kernel)
kernproc.binarize_otsu()
kernel = kernproc.get_image()
improc = ImageEvents.ImageProcessing(img)
improc.binarize_otsu()


corelation = improc.correlate(kernel)
improc.find_peaks()
improc.display(img)