#!/usr/bin/env python2

import cv2
import os
import re
from matplotlib import pyplot as plt
import numpy as np


ADD_IMAGES_FAILED_TO_ALIGN = True


def transform_image_to_base_image(img1, base):

    MIN_MATCH_COUNT = 10


    # Initiate ORB detector
    # orb = cv2.ORB()      # OpenCV2
    orb = cv2.ORB_create() # OpenCV3

    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(base,None)

    # initialize matcher
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bfm.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # filtering matches
    good = matches[0:len(matches)/2]

    # getting source & destination points
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # finding homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,b = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # warping perspective
    dst = cv2.warpPerspective(img1, M, (w, h))

    return dst


# main
if __name__ == '__main__':

    mypath = "./input/"
    outpath = "./output/"
    images = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    images.sort()
    print("\n".join(images))

    base = cv2.imread(mypath + images[0])
    stabilized_average = np.float32(base.copy())/len(images)

    for i in range(len(images) - 1):
        print(str(i + 1) + "/" + str(len(images) - 1))
        
        imgpath = mypath + images[i + 1]
        
        img1 = cv2.imread(imgpath)
        
        try:
            stabilized_average += np.float32(transform_image_to_base_image(img1, base))/len(images)
            
        except Exception as e:
            print("not processed: {} \nbecause: {}".format(imgpath, e))
            
            if ADD_IMAGES_FAILED_TO_ALIGN:
                stabilized_average += np.float32(img1)/len(images)
        
    outname = outpath+images[0]+'-'+images[-1]+'.png'
    
    print("Saving file " + outname)

    cv2.imwrite(outname, np.uint16(stabilized_average*256))
    
    
    
    
    
