#!/usr/bin/env python3

import cv2
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import atexit
import rawpy
import imageio


## DEPENDENCIES
# python3
# opencv3, numpy
# pip3 install matplotlib rawpy imageio
# sudo apt install python3-tk

INPUT_MODE = 2 # 0 - webcam, 1 - video, 2 - image sequence, 3 - raw files sequence
ALIGN = True # if False, just average
DEBUG = 1
RESIZE=1.0 # 0.5 gives half of every dim, and so on...

IN_DIR  = "./input/"
DARK_IN_DIR  = "./input_darkframe/"
DARKFRAME_COEFF = 1.0
OUT_DIR = "./output/"

## Returns 0.0 ... 65535.0 darkframe loaded from 16bpp image
def get_darkframe():
    mypath  = DARK_IN_DIR

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    
    if len(images_paths) == 0:
        return None
    
    darkframe = cv2.imread( mypath + images_paths[0])
    
    return np.float32(darkframe)*DARKFRAME_COEFF

def get_frame_from_main_camera():
    cap = cv2.VideoCapture(0)

    counter = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        yield frame, "frame{}".format(counter)
        counter += 1


def get_next_frame_from_video():
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

def get_next_frame_from_raws():
    mypath  = IN_DIR
    outpath = OUT_DIR

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    images_paths.sort()

    counter = 0
    for imgpath in images_paths:
        with rawpy.imread(mypath + imgpath) as raw:
            rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
            yield rgb
        counter += 1

def get_next_frame_from_chosen_source(src = 1):
    if src == 0:
        return get_frame_from_main_camera()
    if src == 1:
        return get_next_frame_from_video()
    if src == 2:
        return get_next_frame_from_file()
    if src == 3:
        return get_next_frame_from_raws()


def compute_homography(img1, base):
    # Initiate ORB detector
    # alg = cv2.ORB_create()
    alg = cv2.AKAZE_create()

    # find the keypoints and descriptors with orb
    kp1 = alg.detect(img1,None)
    kp2 = alg.detect(base,None)

    print("    > found {} kp1".format(len(kp1)))
    print("    > found {} kp2".format(len(kp2)))

    # kp1 = sorted(kp1, key=lambda x: -x.response) #[0:int(len(kp1)/2)]
    # kp2 = sorted(kp2, key=lambda x: -x.response) #[0:int(len(kp2)/2)]

    kp1, des1 = alg.compute(img1, kp1)
    kp2, des2 = alg.compute(base, kp2)

    # initialize matcher
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # matching
    matches = bfm.match(des1, des2)
    print("    > found {} matches".format(len(matches)))
    
    # filtering matches
    avg_dist = 0
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        im1_x, im1_y = np.int32(kp1[img1_idx].pt)
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)
        
        a = np.array((im1_x ,im1_y))
        b = np.array((im2_x, im2_y))
        
        avg_dist += np.linalg.norm(a - b)
    avg_dist /= len(matches)
    
    dist_diffs = []
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        im1_x, im1_y = np.int32(kp1[img1_idx].pt)
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)
        
        a = np.array((im1_x ,im1_y))
        b = np.array((im2_x, im2_y))
        
        dist = np.linalg.norm(a - b)
        dist_diff = np.linalg.norm(avg_dist - dist)
        dist_diffs.append(dist_diff)
        
    sorted_y_idx_list = sorted(range(len(dist_diffs)), key=lambda x:dist_diffs[x])
    good = [matches[i] for i in sorted_y_idx_list ][0:int(len(matches)/2)]
    good = sorted(good, key = lambda x:x.distance)
    good = good[0:int(len(good)/2)]
    print("    > after filtration {} matches left\n".format(len(good)))

    # getting source & destination points
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    if DEBUG >= 2:
        matchesimg = cv2.drawMatches(img1, kp1, base, kp2, good, flags=4, outImg=None)
        cv2.imshow( "matchesimg", matchesimg );
        cv2.waitKey(1);
    # finding homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, mask


def transform_image_to_base_image(img1, H):
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
    global stabilized_average, divider_mask, imagenames, outname, stabilized_average, darkframe_average

    stabilized_average /= divider_mask # dividing sum by number of images
 
    outname = OUT_DIR+imagenames[0]+'-'+imagenames[-1]+'.png'

    print("Saving file " + outname)
    
    stabilized_average[stabilized_average < 0] = 0
    stabilized_average[stabilized_average > 255] = 255
    
    cv2.imwrite(outname, np.uint16(stabilized_average*256))


# main
if __name__ == '__main__':
    atexit.register(exit_handler)

    stabilized_average = None
    darkframe_image = get_darkframe()
    mask_image = None
    divider_mask = None

    counter = 0
    imagenames = []
    for img1_, imgname in get_next_frame_from_chosen_source(INPUT_MODE):
        img1 = None

        try:
            img1 = img1_
            img1 = cv2.resize(img1, (0, 0), fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_LINEAR)
            print("IMAGE NR {} of size {}".format(counter, img1.shape))
        except Exception as e:
            print("Nope")
            continue

        imagenames.append(imgname)
        
        imgpath = IN_DIR + imgname
        
        if counter == 0:
            stabilized_average = np.float32(img1)
            if darkframe_image is not None:
                darkshape = img1.shape
                darkframe_image = cv2.resize(darkframe_image, (darkshape[1], darkshape[0]), interpolation=cv2.INTER_LINEAR)
                stabilized_average = stabilized_average - darkframe_image
                
            mask_image = np.ones_like(img1, dtype=np.float32)
            divider_mask = mask_image.copy()

            if DEBUG >= 3:
                cv2.imwrite(OUT_DIR + imgname + '_DEBUG.jpg', stabilized_average)
                cv2.imwrite(OUT_DIR + imgname + '_divider_maskDEBUG.png', np.uint16(divider_mask*65535/(counter+1)))

        else:
            transformed_image = img1.copy()
            H = None
            
            if darkframe_image is not None:
                transformed_image = transformed_image - darkframe_image
                
            if ALIGN:
                base = (stabilized_average/divider_mask)  
                base[base <= 0] = 0
                base[base >= 255] = 255
                H, _ = compute_homography(img1, np.array(base, dtype=img1.dtype)  )
                transformed_image, mask_image = transform_image_to_base_image(transformed_image, H)
                
            stabilized_average += np.float32(transformed_image)
            divider_mask += mask_image

            if DEBUG >= 1:
                cv2.imshow( "stabilized_average/256/divider_mask", stabilized_average/256/divider_mask );
                cv2.imshow( "transformed_image/255", transformed_image/255 );
                cv2.imshow( "divider_mask", divider_mask/(counter+1) );
                cv2.waitKey(1);

            if DEBUG >= 3:
                cv2.imwrite(OUT_DIR + imgname + '_DEBUG.jpg', transformed_image)
                cv2.imwrite(OUT_DIR + imgname + '_divider_maskDEBUG.png', np.uint16(divider_mask*65535/(counter+1)))
        counter += 1
        
    #exit_handler()

