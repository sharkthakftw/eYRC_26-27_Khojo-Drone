#!/usr/bin/env python3


# This script is used to run the first task 1A of Khojo Drone.

# You will be building the image processing pipeline for the drone. The pipeline will take in an image and output a txt file. 
import cv2
import numpy as np
import string
import sys
import os

# ============ Settings (change these if needed) ============
IMAGE_NAME = "image_1.jpg"          # the arena photo (keep the quotes)
REQUIRED_IDS = [80, 85, 90, 95]     # the four corner markers
OUT_SIZE = 900                      # final arena size: 900 x 900
GRID_CELLS = 12                     # arena is 12 x 12 cells
EDGE_MARGIN = 0                     # set to 1-3 if a thin marker/white strip shows on the edges
MIN_AREA = 100                      # ignore coloured blobs smaller than this (noise)
SHOW_WINDOWS = True                 # set False if you have no display; PNGs are still saved

# Colour ranges in HSV (OpenCV hue goes 0-179). Tune these if a survivor is missed.
RED_RANGES = [((0, 120, 100), (10, 255, 255)),      # red sits at both ends of the hue scale
              ((170, 120, 100), (179, 255, 255))]
YELLOW_RANGES = [((20, 100, 100), (35, 255, 255))]


# ============ Find the image ============
# Look in the folder you run from first, then in the folder this script lives in.
script_dir = os.path.dirname(os.path.abspath(__file__))
candidates = [IMAGE_NAME, os.path.join(script_dir, IMAGE_NAME)]
IMAGE_PATH = next((p for p in candidates if os.path.exists(p)), None)

if IMAGE_PATH is None:
    print("Could not find", IMAGE_NAME)
    print("Looked in:", os.getcwd(), "and", script_dir)
    print("Put the image in one of those folders, or set IMAGE_NAME to its full path.")
    sys.exit(1)

# Output files go next to the script so they are easy to find
out_dir = script_dir


# ============ Part 1: find the corner markers ============
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Found the file but could not read it as an image:", IMAGE_PATH)
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
corners, ids, rejected = detector.detectMarkers(gray)

if ids is None:
    print("No markers detected at all.")
    sys.exit(1)

ids = ids.flatten()
print("Detected IDs:", ids.tolist())

markers = {}
for marker_id, c in zip(ids, corners):
    if int(marker_id) in REQUIRED_IDS:
        markers[int(marker_id)] = c.reshape(4, 2)

missing = [i for i in REQUIRED_IDS if i not in markers]
if missing:
    print("Missing marker(s):", missing)
    sys.exit(1)
print("All four markers found.")

detections_img = img.copy()
cv2.aruco.drawDetectedMarkers(detections_img, corners, ids.reshape(-1, 1))


# ============ Part 2: straighten the arena (perspective transform) ============
centres = {mid: markers[mid].mean(axis=0) for mid in REQUIRED_IDS}
arena_centre = np.mean(list(centres.values()), axis=0)

# Inner corner of each marker = the corner closest to the arena centre
inner = {}
for mid in REQUIRED_IDS:
    dists = np.linalg.norm(markers[mid] - arena_centre, axis=1)
    inner[mid] = markers[mid][np.argmin(dists)]

# Which marker is top-left, top-right, bottom-right, bottom-left
by_y = sorted(REQUIRED_IDS, key=lambda m: centres[m][1])
tl_id, tr_id = sorted(by_y[:2], key=lambda m: centres[m][0])
bl_id, br_id = sorted(by_y[2:], key=lambda m: centres[m][0])
print("TL, TR, BR, BL IDs:", tl_id, tr_id, br_id, bl_id)

src = np.array([inner[tl_id], inner[tr_id], inner[br_id], inner[bl_id]],
               dtype=np.float32)
m = EDGE_MARGIN
src += np.array([[m, m], [-m, m], [-m, -m], [m, -m]], dtype=np.float32)

dst = np.array([[0, 0],
                [OUT_SIZE - 1, 0],
                [OUT_SIZE - 1, OUT_SIZE - 1],
                [0, OUT_SIZE - 1]], dtype=np.float32)

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (OUT_SIZE, OUT_SIZE))
print("Warped shape:", warped.shape)          # must be (900, 900, 3)

for p in src:
    cv2.circle(detections_img, tuple(int(v) for v in p), 6, (0, 0, 255), -1)


# ============ Part 3: grid lines and the 121 intersections ============
cell = OUT_SIZE / GRID_CELLS                  # 75 px per cell
line_positions = [int(round(cell * k)) for k in range(1, GRID_CELLS)]   # 75 ... 825


# ============ Part 4: name every intersection ============
# Columns A-K (left to right), rows 1-11 (top to bottom), e.g. "A1", "C2", "K11"
column_letters = string.ascii_uppercase[:GRID_CELLS - 1]    # "ABCDEFGHIJK"

intersections = {}      # name -> (x, y) in the 900 x 900 image
for row_index, y in enumerate(line_positions):
    for col_index, x in enumerate(line_positions):
        name = column_letters[col_index] + str(row_index + 1)
        intersections[name] = (x, y)
        # ============ Part 5: find the survivors (red and yellow regions) ============
import os

MIN_AREA = 100                      # ignore coloured blobs smaller than this (noise)
SHOW_WINDOWS = True                 # set False if you have no display; PNGs are still saved

# Colour ranges in HSV (OpenCV hue goes 0-179). Tune these if a survivor is missed.
RED_RANGES = [((0, 120, 100), (10, 255, 255)),      # red sits at both ends of the hue scale
              ((170, 120, 100), (179, 255, 255))]
YELLOW_RANGES = [((20, 100, 100), (35, 255, 255))]

out_dir = os.path.dirname(os.path.abspath(__file__))    # save images next to the script

hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)     # use the clean warped image
kernel = np.ones((5, 5), np.uint8)


def colour_mask(ranges):
    """Black/white mask for one colour (white = that colour)."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove tiny specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes
    return mask


def find_regions(mask):
    """Outline of every region big enough to be a survivor."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_AREA]


red_mask = colour_mask(RED_RANGES)
yellow_mask = colour_mask(YELLOW_RANGES)

red_regions = find_regions(red_mask)
yellow_regions = find_regions(yellow_mask)

print("Red survivors found   :", len(red_regions))
print("Yellow survivors found:", len(yellow_regions))
print("Total                 :", len(red_regions) + len(yellow_regions))


# ============ Part 6: reduce each survivor to one point (its centre) ============
def region_centre(contour):
    """Centre of mass of the filled region (works for circles and triangles).
    Falls back to the bounding-box centre if the region has zero area."""
    mo = cv2.moments(contour)
    if mo["m00"] > 0:
        return mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    x, y, w, h = cv2.boundingRect(contour)
    return x + w / 2, y + h / 2


# ============ Part 7: match each centre to the nearest intersection ============
def nearest_intersection(cx, cy):
    """Nearest grid crossing. Crossings sit at multiples of `cell`, from 1 to 11."""
    col = int(cx / cell + 0.5)
    row = int(cy / cell + 0.5)
    col = min(max(col, 1), GRID_CELLS - 1)
    row = min(max(row, 1), GRID_CELLS - 1)
    return column_letters[col - 1] + str(row)


survivors = []      # (colour, centre_x,