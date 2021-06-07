# -------------------------------------------------------------------------
# --- Author         : Pradeep Shinde
# --- Date           : 16th September 2020
# -------------------------------------------------------------------------

# import the necessary packages
import numpy as np
import cv2
import pandas as pd

# initialize the list of reference points and boolean indicating whether cropping is being performed or not
refPt = []
cropping = False

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

# load the image, clone it, and setup the mouse callback function
image=cv2.imread('colorpic.jpg')
roi=image


# Function for cropping Image
def click_and_crop(event, x, y, flags, param):

    # Grab references to the global variables
    global refPt, cropping, roi

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # to record the ending (x, y) coordinates and indicate that the cropping operation is finished
        refPt.append((x, y))
        cropping = True
        # to draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

        # calling other functions
        func_call()


# Function to call color_recognition_kmeans() and get_color_name()
def func_call():
    # Grab references to the global variables
    global refPt, roi, cropping
    if cropping:
        # if there are two reference points, then crop the region of interest
        if len(refPt) == 2:
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

            rgb = color_recognition_kmeans(roi)
            b, g, r = rgb[0], rgb[1], rgb[2]

            # cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle
            cv2.rectangle(image, (20, 20), (750, 60), (b, g, r), -1)

            # Creating text string to display( Color name and RGB values )
            text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
            print('color name : ', text)

            if r + g + b >= 600:
                # For very light colours we will display text in black colour
                cv2.putText(image, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cropping = False


# Function to return R,G,B values using k-means clustering Algorithm
def color_recognition_kmeans(image_cropped):
    print('Color Recognition.....')

    # reshape image from [X][Y][3] to [X*Y,3]
    img_reshape = [img.reshape((-1, 3)) for img in image_cropped]

    # reshape numpy 2D array to 1D array
    pixels_flat = np.concatenate(img_reshape)
    # change to float
    pixels_flat = np.float32(pixels_flat)

    # define number of clusters(K)
    k = 1
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # apply kmeans()
    center = cv2.kmeans(pixels_flat, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # extract rgb values
    rgb_values = [int(x) for x in center[2][0]]
    print(rgb_values)
    return rgb_values


# Function to calculate minimum distance from all colors and get the most matching color
def get_color_name(r, g, b):
    cname = ''
    minimum = 10000
    for i in range(len(csv)):
        d = abs(r - int(csv.loc[i, "r"])) + abs(g - int(csv.loc[i, "g"])) + abs(b - int(csv.loc[i, "b"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


clone = image.copy()
cv2.namedWindow("image")

while 1:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    image = clone.copy()
    cv2.setMouseCallback("image", click_and_crop)
    # Break the loop when user hits 'esc' key
    if cv2.waitKey(0) & 0xFF == 27:
        break

# close all open windows
cv2.destroyAllWindows()
