import cv2 as cv2

def otsu(image):
    # cv2.cvtColor is applied over the image input with applied parameters to convert the image in grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # applying Otsu thresholding as an extra flag in binary
    # thresholding
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh1
