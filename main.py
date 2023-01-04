from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def show(name, var):
    scale_percent = 20
    width = int(var.shape[1] * scale_percent / 100)
    height = int(var.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(var, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow(name)
    cv2.moveWindow(name, 50, 50)
    cv2.imshow(name, resized)
    cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-c", "--calibre", type=float, required=True,
                help="the length of the calibration square")

args = vars(ap.parse_args())

original_image = cv2.imread(args['image'])
show('original_image', original_image)

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
show('gray_image', gray_image)


blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
show('blur_image', blur_image)


minimum_treshold = 50
maximum_treshold = 100


canny_image = cv2.Canny(blur_image, minimum_treshold, maximum_treshold)
show('canny_image', canny_image)


kernel_dilate = np.ones((5, 5))
dilate_image = cv2.dilate(canny_image, kernel_dilate, iterations=3)
show('dilate_image', dilate_image)


kernel_erode = np.ones((5, 5))
erode_image = cv2.erode(dilate_image, kernel_erode, iterations=2)
show('erode_image', erode_image)


cnts = cv2.findContours(erode_image, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetricA = None
pixelsPerMetricB = None

orig = original_image.copy()
for c in cnts:

    if cv2.contourArea(c) < 1000:
        continue

    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 8)

    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 10, (0, 0, 255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 10, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 10, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 10, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 10, (255, 0, 0), -1)

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 8)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 8)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetricB is None:
        pixelsPerMetricB = dB / args["calibre"]

    if pixelsPerMetricA is None:
        pixelsPerMetricA = dA / args["calibre"]

    dimA = dA / pixelsPerMetricA
    dimB = dB / pixelsPerMetricB

    cv2.putText(orig, "{:.2f}cm".format(dimB),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                3, (0, 0, 0), 8)
    cv2.putText(orig, "{:.2f}cm".format(dimA),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                3, (0, 0, 0), 8)

    show("output", orig)
