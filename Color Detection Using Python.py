# import the necessary packages
import numpy as np
import cv2
import pandas as pd

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('C:\\Users\\Ay507tx\\Desktop\\n\\python-project-color-detection\\colors.csv', names=index, header=None)

# load the image, clone it, and setup the mouse callback function
image=cv2.imread('C:\\Users\\Ay507tx\\Desktop\\n\\python-project-color-detection\\colorpic.jpg')
roi=image

#Function for cropping Image
def click_and_crop(event, x, y, flags, param):
	# Grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

# Function to calculate minimum distance from all colors and get the most matching color
def getColorName(R, G, B):
	minimum = 10000
	for i in range(len(csv)):
		d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
		if (d <= minimum):
			minimum = d
			cname = csv.loc[i, "color_name"]
	# print('color name : ',cname)
	return cname

#Function to return R,G,B values
def Color_Recognition_Kmeans(image_cropped):
	print('Color Recognition.....')

	# reshape image from [X][Y][3] to [X*Y,3]
	img_reshape = [img.reshape((-1, 3)) for img in image_cropped]

	# reshape numpy 2D array to 1D array
	pixels_flat = np.concatenate(img_reshape)

	# change to float
	pixels_flat = np.float32(pixels_flat)

	# define criteria, number of clusters(K) and apply kmeans()
	# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 1
	ret, label, center = cv2.kmeans(pixels_flat, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# change to unit center
	center = np.uint8(center)

	# a list
	center = [tuple([int(c) for c in color]) for color in center]

	# return unique class label of an array and its coming counts
	l, c = np.unique(label, return_counts=True)

	# sorting list (l, c),reverse it from high to low
	order = sorted(zip(c, l), reverse=True)

	# just show class label in order
	order = [l for c, l in order]

	# convert list to array in order, then to list
	center = np.asarray(center)[order].tolist()

	print(center[0])
	rgb = center[0]
	return rgb
	# print(rgb)

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
clicked=False

while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	image = clone.copy()
	key = cv2.waitKey(1) & 0xFF

	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	rgb=Color_Recognition_Kmeans(roi)
	b = rgb[0]
	g = rgb[1]
	r = rgb[2]

	# cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle
	cv2.rectangle(image, (20, 20), (750, 60), (b, g, r), -1)

	# Creating text string to display( Color name and RGB values )
	text = getColorName(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
	print('color name : ', text)

	# cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
	cv2.putText(image, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

	# For very light colours we will display text in black colour
	if (r + g + b >= 600):
		cv2.putText(image, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
		# cv2.imshow("ROI", roi)

	# Break the loop when user hits 'esc' key
	if cv2.waitKey(0) & 0xFF == 27:
		break
# close all open windows
cv2.destroyAllWindows()