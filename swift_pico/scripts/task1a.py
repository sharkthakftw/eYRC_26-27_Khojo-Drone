#!/usr/bin/env python3


# This script is used to run the first task 1A of Khojo Drone.

# You will be building the image processing pipeline for the drone. The pipeline will take in an image and output a txt file. 
import cv2
import numpy as np
import os
import sys

img_name = "image_1.jpg"
needed_ids = [80, 85, 90, 95]
size = 900
cells = 12
min_area = 100

folder = os.path.dirname(os.path.abspath(__file__))


# Part 1
img = cv2.imread(os.path.join(folder, img_name))
if img is None:
    print("cant read", img_name)
    sys.exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
corners, ids, rejected = detector.detectMarkers(gray)

if ids is None:
    print("no markers found")
    sys.exit()

ids = ids.flatten()
print("ids:", ids.tolist())

markers = {}
for i, c in zip(ids, corners):
    if int(i) in needed_ids:
        markers[int(i)] = c.reshape(4, 2)

for i in needed_ids:
    if i not in markers:
        print("marker", i, "is missing")
        sys.exit()

det_img = img.copy()
cv2.aruco.drawDetectedMarkers(det_img, corners, ids.reshape(-1, 1))


# Part 2
centers = {}
for i in needed_ids:
    centers[i] = markers[i].mean(axis=0)
mid = np.mean(list(centers.values()), axis=0)

# corner of each marker that is closest to the middle
inner = {}
for i in needed_ids:
    d = np.linalg.norm(markers[i] - mid, axis=1)
    inner[i] = markers[i][np.argmin(d)]

order = sorted(needed_ids, key=lambda i: centers[i][1])
top = sorted(order[:2], key=lambda i: centers[i][0])
bottom = sorted(order[2:], key=lambda i: centers[i][0])

src = np.float32([inner[top[0]], inner[top[1]], inner[bottom[1]], inner[bottom[0]]])
dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (size, size))
print("warped:", warped.shape)


# Part 3
cell = size / cells
lines = [int(round(cell * k)) for k in range(1, cells)]


# Part 4
letters = "ABCDEFGHIJK"
points = {}
for r in range(11):
    for c in range(11):
        points[letters[c] + str(r + 1)] = (lines[c], lines[r])
print("intersections:", len(points))


# Part 5
hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
kernel = np.ones((5, 5), np.uint8)

red1 = cv2.inRange(hsv, (0, 120, 100), (10, 255, 255))
red2 = cv2.inRange(hsv, (170, 120, 100), (179, 255, 255))
red_mask = cv2.bitwise_or(red1, red2)
yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
yellow_cnts, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

red_cnts = [c for c in red_cnts if cv2.contourArea(c) > min_area]
yellow_cnts = [c for c in yellow_cnts if cv2.contourArea(c) > min_area]
print("red:", len(red_cnts), "yellow:", len(yellow_cnts))


# Part 6
def get_center(cnt):
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(cnt)
        return x + w / 2, y + h / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


# Part 7
def get_name(x, y):
    col = int(x / cell + 0.5)
    row = int(y / cell + 0.5)
    col = max(1, min(col, 11))
    row = max(1, min(row, 11))
    return letters[col - 1] + str(row)


found = []
for cnt in red_cnts:
    x, y = get_center(cnt)
    found.append(("red", x, y, get_name(x, y)))
for cnt in yellow_cnts:
    x, y = get_center(cnt)
    found.append(("yellow", x, y, get_name(x, y)))

for colour, x, y, name in found:
    print(colour, round(x, 1), round(y, 1), name)


# draw results
out = warped.copy()
cv2.drawContours(out, red_cnts + yellow_cnts, -1, (0, 255, 0), 2)

for p in lines:
    cv2.line(out, (p, 0), (p, size - 1), (255, 0, 255), 1)
    cv2.line(out, (0, p), (size - 1, p), (255, 0, 255), 1)

for colour, x, y, name in found:
    pt = (int(x), int(y))
    cv2.circle(out, pt, 5, (0, 0, 0), -1)
    cv2.putText(out, name, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, name, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

cv2.imwrite(os.path.join(folder, "detections.png"), det_img)
cv2.imwrite(os.path.join(folder, "warped.png"), warped)
cv2.imwrite(os.path.join(folder, "red_mask.png"), red_mask)
cv2.imwrite(os.path.join(folder, "yellow_mask.png"), yellow_mask)
cv2.imwrite(os.path.join(folder, "survivors.png"), out)

cv2.imshow("survivors", out)
cv2.waitKey(0)
cv2.destroyAllWindows()