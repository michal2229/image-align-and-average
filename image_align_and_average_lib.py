#!/usr/bin/env python3

import cv2
import os
import numpy as np
import math as ma
import atexit
import pickle
import gzip
import bz2

## DEPENDENCIES
# python3
# opencv3, numpy
# pip3 install matplotlib
# sudo apt install python3-tk

stabilized_average, divider_mask_sane, imagenames, outname = None, None, None, None


def get_darkframe(_path, _coeff):
    mypath = _path

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]

    if len(images_paths) == 0:
        return None

    darkframe = cv2.imread(mypath + images_paths[0])

    print("get_darkframe(): loaded darkframe.")

    return np.float32(darkframe) / 255 * _coeff


def get_frame_from_main_camera(_maxframes=-1):
    cap = cv2.VideoCapture(0)

    counter = 0
    while (counter < _maxframes or _maxframes < 0):
        # Capture frame-by-frame
        ret, frame = cap.read()

        yield np.float32(frame) / 255, "webcam_{}".format(counter), -1
        counter += 1


def get_next_frame_from_video(_indir, _maxframes=-1, _file_not_dir=False):
    cap = cv2.VideoCapture(0)

    mypath = _indir
    videos_paths = None

    if _file_not_dir:
        videos_paths = [mypath]
    else:
        videos_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
        videos_paths.sort()

    counter = 0
    for vidpath in videos_paths:
        if _file_not_dir:
            cap = cv2.VideoCapture(mypath + vidpath)
        else:
            cap = cv2.VideoCapture(vidpath)

        l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        while (counter < _maxframes or _maxframes < 0):
            ret, frame = cap.read()

            if ret == False:
                break

            yield np.float32(frame) / 255, vidpath, l
            counter += 1


def get_next_frame_from_file(_indir, _maxframes=-1, _file_not_dir=False):
    mypath = _indir

    if _file_not_dir:
        yield np.float32(cv2.imread(_indir)) / 255

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    images_paths.sort()

    counter = 0
    for imgpath in images_paths:
        if _maxframes > -1 and counter >= _maxframes:
            break

        yield np.float32(cv2.imread(mypath + imgpath)) / 255, imgpath, len(images_paths)
        counter += 1


def get_next_frame_from_chosen_source(_src, _indir, _maxframes=-1, _file_not_dir=False):
    if _src == 0:
        return get_frame_from_main_camera()
    if _src == 1:
        return get_next_frame_from_video(_indir, _maxframes, _file_not_dir)
    if _src == 2:
        return get_next_frame_from_file(_indir, _maxframes, _file_not_dir)


def compute_homography(_input_image, _base, _pxdistcoeff, _dscdistcoeff, _cache, _debug):
    k_cnt = "counter"
    k_kp2 = "kp2"
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
    kp1 = alg.detect(_input_image, None)
    kp1, des1 = alg.compute(_input_image, kp1)

    kp2, des2 = None, None
    if _cache is not None:
        if _cache[k_cnt] < 10 or _cache[k_cnt] % 10 == 0:
            kp2 = alg.detect(_base, None)
            kp2, des2 = alg.compute(_base, kp2)
            _cache[k_kp2] = kp2
            _cache[k_des2] = des2
            print("compute_homography(): found and cached {} kp2".format(len(kp2)))
        else:
            kp2, des2 = _cache[k_kp2], _cache[k_des2]
            print("compute_homography(): reused {} kp2".format(len(kp2)))
    else:
        kp2 = alg.detect(_base, None)
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

        a = np.array((im1_x, im1_y))
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

        a = np.array((im1_x, im1_y))
        b = np.array((im2_x, im2_y))

        dist = np.linalg.norm(a - b)
        dist_diff = np.linalg.norm(avg_dist - dist)
        dist_diffs.append(dist_diff)

    sorted_y_idx_list = sorted(range(len(dist_diffs)), key=lambda x: dist_diffs[x])
    good = [matches[i] for i in sorted_y_idx_list][0:int(len(matches) * _pxdistcoeff)]

    good = sorted(good, key=lambda x: x.distance)
    good = good[0:int(len(good) * _dscdistcoeff)]

    kp2_xyd = []
    dist_diffs = []
    for match in matches:
        img2_idx = match.trainIdx
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)
        kp2_xyd.append(((im2_x, im2_y), match.distance, kp2[img2_idx].response))

    print("compute_homography(): after filtration {} matches left".format(len(good)))

    # getting source & destination points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if _debug >= 2:
        matchesimg = cv2.drawMatches(_input_image, kp1, _base, kp2, good, flags=4, outImg=None)
        cv2.imshow("compute_homography(): matchesimg", matchesimg)
        cv2.waitKey(1)
    # finding homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, [kp2_xyd, maxresponse, maxdistance]


def transform_image_to_base_image(_input_image, _h):
    h, w, b = _input_image.shape
    dst = cv2.warpPerspective(_input_image, _h, (w, h))

    return dst


def make_mask_for_image(_img, _border_coeff):
    return make_border_for_image(np.ones_like(_img), _border_coeff)


def make_border_for_image(_img, _border_coeff):
    h, w, b = _img.shape
    hborder = int(h * _border_coeff)
    wborder = int(w * _border_coeff)
    map1 = cv2.copyMakeBorder(_img, hborder, hborder, wborder, wborder, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return map1

def make_me_a_kernel(_kernel_size):
    kcenter = (_kernel_size - 1) / 2 + 1

    kernel = np.zeros((_kernel_size, _kernel_size),
                      dtype=np.float32)  # cv2.getGaussianKernel((blurradius, blurradius), 0)

    for kx in range(kernel.shape[1]):
        for ky in range(kernel.shape[0]):
            dist = (kcenter - np.linalg.norm(np.array((kcenter - kx, kcenter - ky)))) / kcenter
            kernel[ky, kx] = ma.pow(dist, 8)
    kernel / np.max(kernel)

    return kernel

def make_mask_from_points(_match_data, _transformed_mask, _border_coeff, _kernel):
    h, w, b = _transformed_mask.shape
    d = ma.sqrt(h ** 2 + w ** 2)
    transmaskmean = np.mean(_transformed_mask)  # percentage of area white
    relative_size_transformed_to_original = ma.sqrt(transmaskmean) / (1 - 2 * _border_coeff)
    print("make_mask_from_points(): transmaskmean = {}".format(transmaskmean))
    print("make_mask_from_points(): relative_size_transformed_to_original = {}".format(
        relative_size_transformed_to_original))
    radius = 1
    kernel_size = 127
    k_resize = (kernel_size + 1) / w
    blurradius = int(w * k_resize)
    blurradius = blurradius - blurradius % 2 + 1
    kernel = None
    # match_good_over_thresh = 5

    mask = np.zeros_like(_transformed_mask)

    print("make_mask_from_points(): match_data = {}".format(_match_data[1:]))

    kp2_xyd, maxresponse, maxdistance = _match_data
    for xykp2, dkp12, rkp2 in kp2_xyd:
        # k_ = max(k, match_good_over_thresh)
        maxdistance = max(1, maxdistance)
        kd = (1 - dkp12 / maxdistance) ** 5
        # kr2 = rkp2/maxresponse
        # print("kd = {}, kr1 = {}, kr2 = {}".format(kd, kr2))

        k = 64 * (kd)  # * kr1 * kr2
        color = (k, k, k)
        # cv2.circle(mask, xykp2, radius, color, thickness = -1)
        mask[xykp2[1], xykp2[0]] += color

    if _kernel is None:
        _kernel = make_me_a_kernel(kernel_size)
    kernel = _kernel

    mask *= _transformed_mask
    mask = cv2.blur(mask,
                    (int(1 / k_resize), int(1 / k_resize)))  # because without it details are missing after downscaling
    cv2.imshow("make_mask_from_points(): mask", mask)
    cv2.imshow("make_mask_from_points(): kernel", kernel)
    cv2.waitKey(1)
    mask = cv2.resize(mask, (0, 0), fx=k_resize, fy=k_resize, interpolation=cv2.INTER_LINEAR)
    # mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("make_mask_from_points(): mask after resize", mask)
    cv2.waitKey(1)
    mask = cv2.filter2D(mask, cv2.CV_32F, kernel)
    # mask = cv2.GaussianBlur(mask, (blurradius, blurradius), 0)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # mask = cv2.blur(mask, (blurradius, blurradius))
    maxval = np.max(mask)
    mask /= maxval

    return mask, kernel


def compute_predicted_H(_h_new, _h_state_old):
    h_state_current = [_h_new, 0, 0]
    h_pred = None
    k = 0.5  # denoising

    if _h_state_old[0] is not None:
        h_state_current[1] = (h_state_current[0] - _h_state_old[0]) * k + _h_state_old[1] * (1 - k)

    if _h_state_old[1] is not None:
        h_state_current[2] = (h_state_current[1] - _h_state_old[1]) * k + _h_state_old[1] * (1 - k)
        h_pred = h_state_current[0] + h_state_current[1] + h_state_current[2] / 2

    return h_pred, h_state_current


def save_my_pickle(_object, _filename):
    filename = "{}.pickle.compressed".format(_filename)
    print("save_my_pickle(): saving pickle: {}".format(filename))

    with gzip.open(filename, 'wb') as f:
        pickle.dump(_object, f)

    print("save_my_pickle(): saved pickle: {}".format(filename))


def load_my_pickle(_filename):
    object = None
    filename = "{}.pickle.compressed".format(_filename)
    print("load_my_pickle(): loading pickle: {}".format(filename))

    try:
        with gzip.open(filename, 'rb') as f:
            object = pickle.load(f)

    except Exception as e:
        print("load_my_pickle(): couldn't load pickle: {}".format(filename))
        print(e)

    print("load_my_pickle(): loaded pickle: {}".format(filename))

    return object