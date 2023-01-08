from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pandas as pd


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def show(name, var):
    scale_percent = 100
    width = int(var.shape[1] * scale_percent / 100)
    height = int(var.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(var, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow(name)
    cv2.moveWindow(name, 50, 50)
    cv2.imshow(name, resized)


# non euclid distance
# using too many pixel per meteric length
# ned stands for none Euclid Distance


def tolist(mydata):
    mydata = mydata[1:-1]
    data = mydata.split(',')
    mylist = [float(i) for i in data]
    return mylist


def read_csv(filename):
    my_df = pd.read_csv(filename)
    my_list = my_df.to_numpy()
    new_list = []
    for i in my_list:
        v = []
        for j in i:
            v.append(tolist(j))
        new_list.append(v)

    return new_list


horizontal_list = read_csv("horizontal.csv")
vertical_list = read_csv("vertical.csv")


def ned(point1, point2):

    x1 = point1[0]
    y1 = point1[1]
    wid = 0
    hid = 0

    for i in vertical_list[0]:
        if y1 <= i[0]:
            break
        wid += 1
    wid = min(wid, len(vertical_list[0])-1)
    x2 = point2[0]
    y2 = point2[1]

    for j in horizontal_list[0]:
        if x2 >= j[0]:
            break
        hid += 1
    hid = min(hid, len(horizontal_list[0])-1)

    clWidth = horizontal_list[wid]
    clHeight = vertical_list[hid]
    xdist = 0
    previous_point = False
    x = x1
    for point in clWidth:

        if previous_point == False:
            previous_point = point

        if x2 <= point[0]:
            xdist += (x2-x)/((previous_point[2]+point[2])/2)
            break
        if x <= point[0]:
            xdist += (point[0]-x)/((previous_point[2]+point[2])/2)
            x = point[0]
        if x2 <= point[1]:
            xdist += (x2-x)/point[2]
            break
        if x <= point[1]:
            xdist += (point[1]-x)/point[2]
            x = point[1]
        previous_point = point

    ydist = 0
    previous_point = False
    y = y1
    for point in clHeight:

        if previous_point == False:
            previous_point = point

        if y2 <= point[0]:
            ydist += (y2-y)/((previous_point[2]+point[2])/2)
            break
        if y <= point[0]:
            ydist += (point[0]-y)/((previous_point[2]+point[2])/2)
            y = point[0]
        if y2 <= point[1]:
            ydist += (y2-y)/point[2]
            break
        if y <= point[1]:
            ydist += (point[1]-y)/point[2]
            y = point[1]
        previous_point = point

    return dist.euclidean((0, 0), (xdist, ydist))


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int,
                required=False, help="the webcam id")
ap.add_argumeqnt("-i", "--image", required=False,
                 help="path to the input image")


args = vars(ap.parse_args())


vidcheck = False
vid = 0

run = True
original_image = ""
if args["webcam"]:
    vid = cv2.VideoCapture(args["webcam"])
    vidcheck = True
if args["image"]:
    original_image = cv2.imread(args['image'])
    print("yes")

while run:

    if vidcheck:
        ret, original_image = vid.read()

    if args["image"]:
        run = False

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    minimum_treshold = 50
    maximum_treshold = 100

    canny_image = cv2.Canny(blur_image, minimum_treshold, maximum_treshold)

    kernel_dilate = np.ones((5, 5))
    dilate_image = cv2.dilate(canny_image, kernel_dilate, iterations=3)

    kernel_erode = np.ones((5, 5))
    erode_image = cv2.erode(dilate_image, kernel_erode, iterations=2)

    cnts = cv2.findContours(erode_image, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts:
        (cnts, _) = contours.sort_contours(cnts)

    orig = original_image.copy()

    for c in cnts:

        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        drawingSize = orig.shape[0]//400
        cv2.drawContours(orig, [box.astype("int")], -
                         1, (0, 255, 0), 2*drawingSize)

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 3*drawingSize, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(orig, (int(tltrX), int(tltrY)),
                   3*drawingSize, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)),
                   3*drawingSize, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)),
                   3*drawingSize, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)),
                   3*drawingSize, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2*drawingSize)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2*drawingSize)

        dimA = ned((tltrX, tltrY), (blbrX, blbrY))
        dimB = ned((tlblX, tlblY), (trbrX, trbrY))

        cv2.putText(orig, "{:.2f}cm".format(dimB),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    drawingSize/2, (0, 0, 0), 2*drawingSize)
        cv2.putText(orig, "{:.2f}cm".format(dimA),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    drawingSize/2, (0, 0, 0), 2*drawingSize)

    show("output", orig)
    if args["image"]:
        show("output", orig)
        cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        run = False
        break


if args["webcam"]:
    vid.release()
