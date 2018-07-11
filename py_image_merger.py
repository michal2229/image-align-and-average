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

ADD_IMAGES_FAILED_TO_ALIGN = False
ADAPT_BRIGHTNESS = True
DEBUG = True


def transform_image_to_base_image(img1, base):

    MIN_MATCH_COUNT = 10

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
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    # M[0][2] = 0 # if you want to find disparity or sth...
    print("    > found homography: \n{}\n".format(M))

    h,w,b = img1.shape

    # warping perspective
    dst = cv2.warpPerspective(img1, M, (w, h))
    
    if ADAPT_BRIGHTNESS:
        im1_brightness = 0.0
        im2_brightness = 0.0
        
        for match in good:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx

            im1_x, im1_y = np.int32(kp1[img1_idx].pt)
            im2_x, im2_y = np.int32(kp2[img2_idx].pt)
            
            im1_brightness += img1[im1_y][im1_x]
            im2_brightness += base[im2_y][im2_x]
        
        img1_to_base_brightness_ratio = im1_brightness/im2_brightness
        
        print("    > brightness ratio is {}".format(cv2.mean(img1_to_base_brightness_ratio)[0]))
        
        dst = dst/cv2.mean(img1_to_base_brightness_ratio)[0]

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
        
        try:
            transformed_image = transform_image_to_base_image(img1, base)
            stabilized_average += np.float32(transformed_image)
            
            if DEBUG:
                cv2.imwrite(outpath + images[i + 1] + '_DEBUG.jpg', transformed_image)
            
        except Exception as e:
            print("not processed: {} \nbecause: {}".format(imgpath, e))
            divider -= 1
    
    print("divider = {}".format(divider))
    stabilized_average /= divider # dividing sum by number of images
        
    outname = outpath+images[0]+'-'+images[-1]+'.png'
    
    if ADAPT_BRIGHTNESS:
        max_brightness = np.max(stabilized_average)
        if max_brightness > 255:
            stabilized_average *= 255/max_brightness # making sure every value is below 255
    
    print("Saving file " + outname)

    cv2.imwrite(outname, np.uint16(stabilized_average*256))
