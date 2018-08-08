#!/usr/bin/env python3

import cv2
import os
import re
from matplotlib import pyplot as plt
import numpy as np

## DEPENDENCIES
# opencv3, numpy
# pip3 install matplotlib
# sudo apt install python3-tk

DEBUG = False
MIN_MATCH_COUNT = 10


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

    h,w,b = img1.shape

    dst = cv2.warpPerspective(img1, H, (w, h))

    return dst


# main
if __name__ == '__main__':

    mypath = "./input/"
    outpath = "./output/"
    images = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    images.sort()
    
    print("\n".join(images))
    
    firstimgpath = mypath + images[0]
    print("Base image is {}".format(firstimgpath))
    
    base = cv2.imread(firstimgpath)
    stabilized_average = np.float32(base.copy())
    
    divider = len(images)
    
    if DEBUG:
        cv2.imwrite(outpath + images[0] + '_DEBUG.jpg', base)

    for i in range(len(images) - 1):
        print(str(i + 1) + "/" + str(len(images) - 1))
        
        imgpath = mypath + images[i + 1]
        
        img1 = cv2.imread(imgpath)
        
        transformed_image = transform_image_to_base_image(img1, base)
        stabilized_average += np.float32(transformed_image)
        
        if DEBUG:
            cv2.imwrite(outpath + images[i + 1] + '_DEBUG.jpg', transformed_image)
    
    print("divider = {}".format(divider))
    stabilized_average /= divider # dividing sum by number of images
        
    outname = outpath+images[0]+'-'+images[-1]+'.png'
    
    print("Saving file " + outname)

    cv2.imwrite(outname, np.uint16(stabilized_average*256))
