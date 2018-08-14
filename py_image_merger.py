#!/usr/bin/env python3

import cv2
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import math as ma
import atexit



## DEPENDENCIES
# python3
# opencv3, numpy
# pip3 install matplotlib
# sudo apt install python3-tk

INPUT_MODE = 1 # 0 - webcam, 1 - video, 2 - image sequence
ALIGN = True # if False, just average
PREDICTION = False
DEBUG = 1
RESIZE = 1 #.0 0.5 gives half of every dim, and so on...
BORDER = 0.1
KEYPOINTS_NEIGHBORHOOD_ONLY = True

IN_DIR  = "./input/"
DARK_IN_DIR  = "./input_darkframe/"
DARKFRAME_COEFF = 1.0
OUT_DIR = "./output/"

stabilized_average, divider_mask, divider_mask_sane, imagenames, outname = None, None, None, None, None

## Returns 0.0 ... 65535.0 darkframe loaded from 16bpp image
def get_darkframe():
    mypath  = DARK_IN_DIR

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    
    if len(images_paths) == 0:
        return None
    
    darkframe = cv2.imread( mypath + images_paths[0])
    
    print("get_darkframe(): loaded darkframe.")
    
    return np.float32(darkframe)/255*DARKFRAME_COEFF

def get_frame_from_main_camera():
    cap = cv2.VideoCapture(0)

    counter = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        yield np.float32(frame)/255, "webcam_{}".format(counter)
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

            yield np.float32(frame)/255, "{}_{}".format(vidpath, counter)
            counter += 1

def get_next_frame_from_file():
    mypath  = IN_DIR
    outpath = OUT_DIR

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    images_paths.sort()

    counter = 0
    for imgpath in images_paths:
        yield np.float32(cv2.imread( mypath + imgpath)), imgpath
        counter += 1

def get_next_frame_from_chosen_source(_src = 1):
    if _src == 0:
        return get_frame_from_main_camera()
    if _src == 1:
        return get_next_frame_from_video()
    if _src == 2:
        return get_next_frame_from_file()


def compute_homography(_input_image, _base, pxdistcoeff = 0.9, dscdistcoeff = 0.9, _cache = None):
    k_cnt  = "counter"
    k_kp2  = "kp2"
    k_des2 = "des2"
    if _cache is not None:
        if k_cnt in _cache:
            _cache[k_cnt] += 1
        else:
            _cache[k_cnt] = 0

    # Initiate ORB detector
    # alg = cv2.ORB_create()
    alg = cv2.AKAZE_create()

    # find the keypoints and descriptors with orb
    kp1 = alg.detect(_input_image,None)
    kp1, des1 = alg.compute(_input_image, kp1)
    
    
    kp2, des2 = None, None
    if _cache is not None:
        if _cache[k_cnt] < 10 or _cache[k_cnt] % 10 == 0:
            kp2 = alg.detect(_base,None)
            kp2, des2 = alg.compute(_base, kp2)
            _cache[k_kp2] = kp2
            _cache[k_des2] = des2
            print("compute_homography(): found and cached {} kp2".format(len(kp2)))    
        else:
            kp2, des2 = _cache[k_kp2], _cache[k_des2]
            print("compute_homography(): reused {} kp2".format(len(kp2)))
    else:
        kp2 = alg.detect(_base,None)
        kp2, des2 = alg.compute(_base, kp2)
        print("compute_homography(): found {} kp2".format(len(kp2)))

    print("compute_homography(): found {} kp1".format(len(kp1)))

    # initialize matcher
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # matching
    matches = bfm.match(des1, des2)
    # matches = bfm.knnMatch(des1, des2, k=2)
    print("compute_homography(): found {} matches".format(len(matches)))
    
    # filtering matches
    maxresponse = -1
    maxdistance = -1
    avg_dist = 0
    for match in matches:
        input_image_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        im1_x, im1_y = np.int32(kp1[input_image_idx].pt)
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)
        
        a = np.array((im1_x ,im1_y))
        b = np.array((im2_x, im2_y))
        
        avg_dist += np.linalg.norm(a - b)
        
        maxresponse = kp1[input_image_idx].response if kp1[input_image_idx].response > maxresponse else maxresponse
        maxresponse = kp2[img2_idx].response if kp2[img2_idx].response > maxresponse else maxresponse
        maxdistance = match.distance if match.distance > maxdistance else maxdistance
        
        # print("compute_homography(): distance = {}".format(match.distance))
    avg_dist /= len(matches)
    
    dist_diffs = []
    for match in matches:
        input_image_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        im1_x, im1_y = np.int32(kp1[input_image_idx].pt)
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)
        
        a = np.array((im1_x ,im1_y))
        b = np.array((im2_x, im2_y))
        
        dist = np.linalg.norm(a - b)
        dist_diff = np.linalg.norm(avg_dist - dist)
        dist_diffs.append(dist_diff)
        
        
    sorted_y_idx_list = sorted(range(len(dist_diffs)), key=lambda x:dist_diffs[x])
    good = [matches[i] for i in sorted_y_idx_list ][0:int(len(matches)*pxdistcoeff)]
    
    good = sorted(good, key = lambda x:x.distance)
    good = good[0:int(len(good)*dscdistcoeff)]
    
    kp2_xyd = []
    dist_diffs = []
    for match in matches:
        img2_idx = match.trainIdx
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)
        kp2_xyd.append(((im2_x, im2_y), match.distance,  kp1[input_image_idx].response, kp2[img2_idx].response))
    
    print("compute_homography(): after filtration {} matches left".format(len(good)))

    # getting source & destination points
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    if DEBUG >= 2:
        matchesimg = cv2.drawMatches(_input_image, kp1, _base, kp2, good, flags=4, outImg=None)
        cv2.imshow( "compute_homography(): matchesimg", matchesimg );
        cv2.waitKey(1);
    # finding homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, [kp2_xyd, maxresponse, maxdistance]


def transform_image_to_base_image(_input_image, _H):
    h,w,b = _input_image.shape
    dst = cv2.warpPerspective(_input_image, _H, (w, h))

    return dst

def make_mask_for_image(_img, _border_coeff = 0):
    return make_border_for_image(np.ones_like(_img), _border_coeff)

def make_border_for_image(_img, _border_coeff = 0):
    h, w, b = _img.shape
    hborder = int(h*_border_coeff)
    wborder = int(w*_border_coeff)
    map1 = cv2.copyMakeBorder(_img, hborder, hborder, wborder, wborder,  borderType = cv2.BORDER_CONSTANT, value = (0, 0, 0) )
    return map1
    
def make_mask_from_points(match_data, transformed_mask, _border_coeff = 0):
    h, w, b = transformed_mask.shape
    d = ma.sqrt(h**2 + w**2)
    transmaskmean = np.mean(transformed_mask) # percentage of area white
    relative_size_transformed_to_original =  ma.sqrt(transmaskmean)/(1 - 2*_border_coeff)
    print("make_mask_from_points(): transmaskmean = {}".format(transmaskmean))
    print("make_mask_from_points(): relative_size_transformed_to_original = {}".format(relative_size_transformed_to_original))
    radius = 1
    kernel_size = 127
    k_resize = (kernel_size + 1)/w
    blurradius = int(w * k_resize)
    blurradius = blurradius - blurradius % 2 + 1
    # match_good_over_thresh = 5
    
    mask = np.zeros_like(transformed_mask)
    
    print("make_mask_from_points(): match_data = {}".format(match_data[1:]))
    
    kp2_xyd, maxresponse, maxdistance = match_data
    for xykp2, dkp12, rkp1, rkp2 in kp2_xyd:
        #k_ = max(k, match_good_over_thresh) 
        kd = (1 - dkp12/maxdistance)**5
        #kr1 = rkp1/maxresponse
        #kr2 = rkp2/maxresponse
        #print("kd = {}, kr1 = {}, kr2 = {}".format(kd, kr1, kr2))
        
        k = 64*(kd) # * kr1 * kr2
        color = (k, k, k)
        #cv2.circle(mask, xykp2, radius, color, thickness = -1)
        mask[xykp2[1], xykp2[0]] += color
        
    kcenter = (kernel_size - 1)/2 + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32) # cv2.getGaussianKernel((blurradius, blurradius), 0)
    for kx in range(kernel.shape[1]):
        for ky in range(kernel.shape[0]):
            dist = (kcenter - np.linalg.norm(np.array((kcenter - kx, kcenter - ky))))/kcenter
            kernel[ky, kx] = ma.pow(dist, int(8/relative_size_transformed_to_original) )
    kernel/np.max(kernel)
    mask *= transformed_mask
    mask = cv2.blur(mask, (int(1/k_resize), int(1/k_resize))) # because without it details are missing after downscaling
    cv2.imshow( "make_mask_from_points(): mask", mask );
    cv2.imshow( "make_mask_from_points(): kernel", kernel );
    cv2.waitKey(1);
    mask = cv2.resize(mask, (0, 0), fx=k_resize, fy=k_resize, interpolation=cv2.INTER_LINEAR)
    #mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow( "make_mask_from_points(): mask after resize", mask );
    cv2.waitKey(1);
    mask = cv2.filter2D(mask, cv2.CV_32F, kernel)
    #mask = cv2.GaussianBlur(mask, (blurradius, blurradius), 0)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # mask = cv2.blur(mask, (blurradius, blurradius))
    maxval = np.max(mask)
    mask /= maxval
    
    return mask

def compute_predicted_H(H_new, H_state_old):
    H_state_current = [H_new, 0, 0]
    H_pred = None
    k = 0.5 # denoising
    
    if H_state_old[0] is not None:
        H_state_current[1] = (H_state_current[0] - H_state_old[0])*k + H_state_old[1]*(1-k)
    
    if H_state_old[1] is not None:
        H_state_current[2] = (H_state_current[1] - H_state_old[1])*k + H_state_old[1]*(1-k)
        H_pred = H_state_current[0] + H_state_current[1] + H_state_current[2]/2
    
    return H_pred, H_state_current

def exit_handler():
    global stabilized_average, divider_mask_sane, imagenames, outname

    stabilized_average /= divider_mask_sane # dividing sum by number of images
 
    outname = OUT_DIR+imagenames[0]+'-'+imagenames[-1]+'.png'

    print("exit_handler(): Saving file " + outname)
    
    stabilized_average[stabilized_average < 0] = 0
    stabilized_average[stabilized_average > 1] = 1
    
    cv2.imwrite(outname, np.uint16(stabilized_average*65535))

    cv2.waitKey();




## MAIN
# 
#
def main():
    global stabilized_average, divider_mask_sane, imagenames, outname
    
    atexit.register(exit_handler)
    
    shape_original = None
    shape_no_border = None
    shape = None

    stabilized_average = None
    darkframe_image_no_border = None # get_darkframe()
    #darkframe_image = None
    mask_image_no_border = None
    mask_image = None
    divider_mask_no_border = None
    divider_mask = None
    divider_mask_sane = None
    transformed_image = None
    transformed_mask  = None
    
    H_state = [None, None, None] # x, v, a
    H_state_x_old = None
    H_state_v_old = None
    H_state_a_old = None
    
    H_pred = None
    
    keypoint_cache = dict()

    counter = 0
    imagenames = []
    for input_image_, imgname in get_next_frame_from_chosen_source(INPUT_MODE):
        input_image_no_border = None
        input_image = None

        try:
            shape_original = input_image_.shape
        except Exception as e:
            print("main(): Nope")
            if counter > 0:
                return
            continue

        print("\nmain(): image nr {}".format(counter + 1))

        if darkframe_image_no_border is None:
            darkframe_image_no_border = get_darkframe()
            darkframe_image_no_border = cv2.resize(darkframe_image_no_border, (shape_original[1], shape_original[0]), interpolation=cv2.INTER_LINEAR)

        if darkframe_image_no_border is not None:
            input_image_no_border = input_image_ - darkframe_image_no_border
            
        if shape is None:
            input_image_no_border = cv2.resize(input_image_no_border, (0, 0), fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_LINEAR)
        else:
            input_image_no_border = cv2.resize(input_image_no_border, (shape_no_border[1], shape_no_border[0]), interpolation=cv2.INTER_LINEAR)

        input_image = make_border_for_image(input_image_no_border, BORDER)
        
        shape_no_border = input_image_no_border.shape
        shape = input_image.shape

        if stabilized_average is None:
            stabilized_average = np.float32(input_image)
            
        if mask_image is None:
            mask_image_no_border = make_mask_for_image(input_image_no_border)
            mask_image = make_border_for_image(mask_image_no_border, BORDER)

        if divider_mask is None:
            divider_mask = make_mask_for_image(mask_image_no_border, BORDER)
            divider_mask_sane = divider_mask.copy()
            divider_mask_sane[divider_mask_sane == 0] = 1
            # divider_mask[divider_mask == 0] = 2**(-100)

        imagenames.append(imgname)
        imgpath = IN_DIR + imgname
        
        if counter == 0:
            counter += 1
            continue
        
        input_image = cv2.resize(input_image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        transformed_image = input_image
        transformed_mask = mask_image
        H = None
        
        
        if ALIGN:
            base = (stabilized_average/divider_mask_sane)  
            base[base <= 0] = 0
            base[base >= 1] = 1
            H, match_data = compute_homography(input_image, np.array(base, dtype=input_image.dtype), pxdistcoeff = 0.9, dscdistcoeff = 0.9, _cache = keypoint_cache )
            
            if H_pred is not None:
                transformed_image_nopred = transform_image_to_base_image(transformed_image, H)
                transformed_image_pred   = transform_image_to_base_image(transformed_image, H_pred)
                cv2.imshow( "main(): transformed_image_nopred", transformed_image_nopred );
                cv2.imshow( "main(): transformed_image_pred",   transformed_image_pred );
                cv2.waitKey(1);
                
                H = (H + H_pred)/2
            
            transformed_image = transform_image_to_base_image(transformed_image, H)
            transformed_mask  = transform_image_to_base_image(transformed_mask, H)
            
            if PREDICTION and (INPUT_MODE == 1 or INPUT_MODE == 0):
                H_pred, H_state = compute_predicted_H(H, H_state)
            
            if KEYPOINTS_NEIGHBORHOOD_ONLY:
                mask_of_kp2 = make_mask_from_points(match_data, transformed_mask, BORDER)
                transformed_mask *= mask_of_kp2
                transformed_image *= mask_of_kp2
                if DEBUG >= 1:
                    cv2.imshow( "main(): maks_of_kp2", mask_of_kp2 );
                    cv2.waitKey(1);
        stabilized_average += np.float32(transformed_image)
        divider_mask += transformed_mask
        divider_mask_sane = divider_mask.copy()
        divider_mask_sane[divider_mask_sane == 0] = 1
        
        if DEBUG >= 1:
            cv2.imshow( "main(): stabilized_average/divider_mask", stabilized_average/divider_mask );
            cv2.imshow( "main(): transformed_mask", transformed_mask );
            cv2.imshow( "main(): transformed_image", transformed_image );
            cv2.imshow( "main(): divider_mask", divider_mask/(counter+1) );
            cv2.waitKey(1);

        if DEBUG >= 3:
            cv2.imwrite(OUT_DIR + imgname + '_DEBUG.jpg', transformed_image)
            cv2.imwrite(OUT_DIR + imgname + '_divider_maskDEBUG.png', np.uint16(divider_mask*255/(counter+1)))
        counter += 1
        
    #exit_handler()

# main
if __name__ == '__main__':
    main()

