#!/usr/bin/env python3

import cv2
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import atexit



## DEPENDENCIES
# opencv3, numpy
# pip3 install matplotlib
# sudo apt install python3-tk

DEBUG = 1
MIN_MATCH_COUNT = 10

IN_DIR  = "./input/"
OUT_DIR = "./output/"

def get_next_framve_from_video():
    cap = cv2.VideoCapture(0)

    mypath  = IN_DIR
    outpath = OUT_DIR

    videos_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    videos_paths.sort()

    counter = 0
    for vidpath in videos_paths:
        cap = cv2.VideoCapture(mypath + vidpath)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            yield frame, "frame{}".format(counter)
            counter += 1

def get_next_frame_from_file():
    mypath  = IN_DIR
    outpath = OUT_DIR

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    images_paths.sort()

    counter = 0
    for imgpath in images_paths:
        yield cv2.imread( mypath + imgpath), imgpath
        counter += 1


def compute_homography(img1, base):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(base,None)

    # initialize matcher
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # matching
    matches = bfm.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    print("    > found {} matches".format(len(matches)))

    # filtering matches
    good = matches[0:int(len(matches)/2)]
    print("    > after filtration {} matches left\n".format(len(good)))

    # getting source & destination points
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # finding homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    return H, mask


def transform_image_to_base_image(img1, base):
    H, mask = compute_homography(img1, base)
    mask = np.ones_like(img1)

    h,w,b = img1.shape

    dst = cv2.warpPerspective(img1, H, (w, h))
    mask = cv2.warpPerspective(mask, H, (w, h))

    return dst, mask


def make_mask_from_image(img):
    map1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, map1 = cv2.threshold(map1, 0.000001, 1, cv2.THRESH_BINARY)
    map1 = cv2.cvtColor(map1, cv2.COLOR_GRAY2BGR)

    return map1


def exit_handler():
    print('My application is ending!')
    global stabilized_average, divider_mask, imagenames, outname, stabilized_average

    stabilized_average /= divider_mask # dividing sum by number of images
 
    outname = OUT_DIR+imagenames[0]+'-'+imagenames[-1]+'.png'

    print("Saving file " + outname)

    cv2.imwrite(outname, np.uint16(stabilized_average*256))

# main
if __name__ == '__main__':
    atexit.register(exit_handler)

    base = None
    stabilized_average = None
    divider_mask = None
    base

    counter = 0
    imagenames = []
    for img1, imgname in get_next_framve_from_video():
        imagenames.append(imgname)

        print("IMAGE NR {}".format(counter))
        
        imgpath = IN_DIR + imgname
        
        if counter == 0:
            base = img1
            stabilized_average = np.float32(img1.copy())
            divider_mask = np.ones_like(stabilized_average)

            if DEBUG == 2:
                cv2.imwrite(OUT_DIR + imgname + '_DEBUG.jpg', stabilized_average)
                cv2.imwrite(OUT_DIR + imgname + '_divider_maskDEBUG.png', np.uint16(divider_mask*65535/(counter+1)))

        else:
            transformed_image, mask = transform_image_to_base_image(img1, base)
            stabilized_average += np.float32(transformed_image)
            divider_mask += mask

            if DEBUG == 1:
                cv2.imshow( "stabilized_average/divider_mask", stabilized_average/256/divider_mask );
                cv2.imshow( "divider_mask", divider_mask/(counter+1) );
                cv2.waitKey(1);

            if DEBUG == 2:
                cv2.imwrite(OUT_DIR + imgname + '_DEBUG.jpg', transformed_image)
                cv2.imwrite(OUT_DIR + imgname + '_divider_maskDEBUG.png', np.uint16(divider_mask*65535/(counter+1)))
        counter += 1
        
    exit_handler()

