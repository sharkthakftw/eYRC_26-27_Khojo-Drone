#!/usr/bin/env python3


# This script is used to run the first task 1A of Khojo Drone.

# You will be building the image processing pipeline for the drone. The pipeline will take in an image and output a txt file.
import argparse
import os
import sys

import cv2
import numpy as np

# colour that counts as critical (swap these if the task page says the opposite)
CRITICAL_COLOUR = "red"
STABLE_COLOUR = "yellow"

needed_ids = [80, 85, 90, 95]
size = 900
cells = 12
min_area = 100

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()


def detect_markers(gray):
    """Detect ArUco markers; works on both old (<4.7) and new (>=4.7) OpenCV."""
    # new API (OpenCV >= 4.7)
    if hasattr(cv2.aruco, "ArucoDetector"):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        p = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(d, p).detectMarkers(gray)

    # old API (OpenCV < 4.7)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    else:
        d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters()
    return cv2.aruco.detectMarkers(gray, d, parameters=p)


# Part 1
img = cv2.imread(args.image)
if img is None:
    print("cant read", args.image)
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners, ids, rejected = detect_markers(gray)

if ids is None:
    print("no markers found")
    sys.exit(1)

ids = ids.flatten()
detected_ids = sorted(int(i) for i in ids)
print("ids:", detected_ids)

markers = {}
for i, c in zip(ids, corners):
    if int(i) in needed_ids:
        markers[int(i)] = c.reshape(4, 2)

for i in needed_ids:
    if i not in markers:
        print("marker", i, "is missing")
        sys.exit(1)


# Part 2
centers = {i: markers[i].mean(axis=0) for i in needed_ids}
mid = np.mean(list(centers.values()), axis=0)

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


# Part 3 and 4
cell = size / cells
letters = "ABCDEFGHIJK"


# Part 5
hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
kernel = np.ones((5, 5), np.uint8)

red1 = cv2.inRange(hsv, (0, 120, 100), (10, 255, 255))
red2 = cv2.inRange(hsv, (170, 120, 100), (179, 255, 255))
masks = {
    "red": cv2.bitwise_or(red1, red2),
    "yellow": cv2.inRange(hsv, (20, 100, 100), (35, 255, 255)),
}

contours = {}
for colour, mask in masks.items():
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # [-2] works for both OpenCV 3 (3 return values) and 4 (2 return values)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    contours[colour] = [c for c in cnts if cv2.contourArea(c) > min_area]


# Part 6
def get_center(cnt):
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(cnt)
        return x + w / 2, y + h / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


# Part 7
def get_name(x, y):
    col = max(1, min(int(x / cell + 0.5), 11))
    row = max(1, min(int(y / cell + 0.5), 11))
    return letters[col - 1] + str(row)


def names_for(colour):
    names = [get_name(*get_center(c)) for c in contours[colour]]
    return sorted(names, key=lambda n: (n[0], int(n[1:])))


critical = names_for(CRITICAL_COLOUR)
stable = names_for(STABLE_COLOUR)


# results file
stem = os.path.splitext(os.path.basename(args.image))[0]
out_file = stem + "_results.txt"

with open(out_file, "w") as f:
    f.write("Detected marker IDs: " + str(detected_ids) + "\n")
    f.write("Critical Survivors: " + ", ".join(critical) + "\n")
    f.write("Stable Survivors: " + ", ".join(stable) + "\n")

print("wrote", out_file)