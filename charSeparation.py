import os
import glob
from imutils.contours import sort_contours
import imutils
import cv2


def removeAntigas():
	try:
		os.makedirs("./imagens/")
	except FileExistsError:
		# directory already exists
		pass
	files = glob.glob('./imagens/*')
	for f in files:
		os.remove(f)

def separa(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	# perform edge detection, find contours in the edge map, and sort the
	# resulting contours from left-to-right
	edged = cv2.Canny(blurred, 30, 150)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sort_contours(cnts, method="left-to-right")[0]

	# initialize the list of contour bounding boxes and associated
	# characters that we'll be OCR'ing
	chars = []
	removeAntigas()
	# loop over the contours
	for c in cnts:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)
		# filter out bounding boxes, ensuring they are neither too small
		# nor too large
		if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
			roi = gray[y:y + h, x:x + w]
			# extract the character and threshold it to make the character
			# appear as *white* (foreground) on a *black* background, then
			# grab the width and height of the thresholded image
			thresh = cv2.threshold(roi, 0, 255,
								   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			(tH, tW) = thresh.shape
			# if the width is greater than the height, resize along the
			# width dimension
			if tW > tH:
				thresh = imutils.resize(thresh, width=32)
			# otherwise, resize along the height
			else:
				thresh = imutils.resize(thresh, height=32)

			cv2.imwrite("./imagens/y=" + str(y) + "-h=" + str(h) + "-x=" + str(x) + "-w=" + str(w) + ".jpg", thresh)

			# re-grab the image dimensions (now that its been resized)
			# and then determine how much we need to pad the width and
			# height such that our image will be 32x32
			(tH, tW) = thresh.shape
			dX = int(max(0, 32 - tW) / 2.0)
			dY = int(max(0, 32 - tH) / 2.0)

			# pad the image and force 32x32 dimensions
			padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
										left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
										value=(0, 0, 0))
			padded = cv2.resize(padded, (32, 32))

			# prepare the padded image for classification via our
			# handwriting OCR model
			padded = padded.astype("float32") / 255.0
