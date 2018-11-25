import imageprocessing as ImageEvents
import cv2 as cv


img = cv.imread('testimages/testimage1.png', 0)
kernel = cv.imread('testimages/kernel1.png', 0)

improc = ImageEvents.ImageProcessing(img)
improc.binarize_otsu()
# improc.convert_to_rgb()
# improc.convert_to_gray()
# improc.draw_rectangle((10,10), (200, 200))
# improc.dilate(20)
imga = improc.get_image()
corelatis = improc.correlate(kernel)
improc.find_peaks()
improc.plot_image()

# imga = improc.get_image()

# ImageEvents.ImageOperations().display(imga)