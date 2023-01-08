from imutils import perspective
import numpy as np
import imutils
import cv2
from functools import cmp_to_key
import argparse
import pandas as pd


def show(name, var):
    scale_percent = 900/var.shape[1]
    width = int(var.shape[1] * scale_percent)
    height = int(var.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(var, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow(name)
    cv2.moveWindow(name, 50, 50)
    cv2.imshow(name, resized)


def sortbycond1(a, b):
    f1 = a[0][1]//50
    s1 = b[0][1]//50

    f2 = a[0][0]
    s2 = b[0][0]

    if f1 != s1:
        return (f1-s1)
    else:
        return (f2-s2)


def sortbycond2(a, b):
    f1 = a[0][0]//50
    s1 = b[0][0]//50

    f2 = a[0][1]
    s2 = b[0][1]

    if f1 != s1:
        return (f1-s1)
    else:
        return (f2-s2)


def write_to_csv(contours_list, scale):
    # horizontal_data
    m = -1
    horizontal_list = []
    v = []
    for box in contours_list:
        if box[0][1]//50 == m:
            v.append([box[0][0], box[1][0], (box[1][0]-box[0][0])/scale])
        else:
            if len(v) != 0:
                horizontal_list.append(v)
            m = box[0][1]//50
            v = [[box[0][0], box[1][0], (box[1][0]-box[0][0])/scale]]

    horizontal_list.append(v)

    horizontal_df = pd.DataFrame(horizontal_list)
    horizontal_df.to_csv('horizontal.csv', index=False)

    # vertical data
    contours_list.sort(key=cmp_to_key(sortbycond2))
    m = -1
    vertical_list = []
    v = []
    for box in contours_list:
        if box[0][0]//50 == m:
            v.append([box[0][1], box[2][1], (box[2][1]-box[0][1])/scale])
        else:
            if len(v) != 0:
                vertical_list.append(v)
            m = box[0][0]//50
            v = [[box[0][1], box[2][1], (box[2][1]-box[0][1])/scale]]

    vertical_list.append(v)
    vertical_df = pd.DataFrame(vertical_list)
    vertical_df.to_csv('vertical.csv', index=False)


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--calibration", type=float,
                required=False, help="enter the legnth of calibration grid ")
ap.add_argument("-i", "--image", required=False,
                help="path to the input image")

ap.add_argument("-w", "--webcam", type=int,
                required=False, help="the webcam id")
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

    contours_list = []
    for c in cnts:

        if cv2.contourArea(c) < 1000:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        contours_list.append(box)
        # cv2.drawContours(orig, [box.astype("int")], -
        #                  1, (0, 0, 255), 2*drawingSize)

    if len(contours_list) < 4:
        continue
    contours_list.sort(key=cmp_to_key(sortbycond1))
    orig = original_image.copy()
    drawingSize = orig.shape[0]//400
    flag = 0
    for box in contours_list:
        flag += 1
        cv2.drawContours(orig, [box.astype("int")], -
                         1, (0, 0, 255), 2*drawingSize)

        text = "calibrating is in process "
        output = orig.copy()
        cv2.putText(output, text+"."*(flag % 4),
                    (orig.shape[1]//10, orig.shape[0] //
                     2), cv2.FONT_HERSHEY_SIMPLEX,
                    drawingSize, (0, 0, 0), 2*drawingSize)
        show("calibration", output)
        cv2.waitKey(400)

    text = "calibratation has completed !! "
    output = orig.copy()
    cv2.putText(output, text+"."*(flag % 4),
                (orig.shape[1]//10, orig.shape[0] //
                 2), cv2.FONT_HERSHEY_SIMPLEX,
                drawingSize, (0, 255, 0), 2*drawingSize)
    show("calibration", output)
    cv2.waitKey(1000)
    write_to_csv(contours_list, args["calibration"])
    print("calibartion has completed successfully now you can you the object measurement")
    run = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        run = False
        break


if args["webcam"]:
    vid.release()
