#!/usr/bin/env python3

import cv2
import os
import numpy as np
import math as ma
import atexit
import pickle
import image_align_and_average_lib as lib

## DEPENDENCIES
# python3
# opencv3, numpy
# pip3 install matplotlib
# sudo apt install python3-tk

INPUT_MODE = 2  # 0 - webcam, 1 - video, 2 - image sequence
ALIGN = True  # if False, just average
PREDICTION = False
DEBUG = 1
RESIZE = 0.25 # 0.5 gives half of every dim, and so on...
BORDER = 0.1
KEYPOINTS_NEIGHBORHOOD_ONLY = True
MAXFRAMES = 250
ALWAYS_PICKLE_FRAMES = False # serialize frames even when available from files
PICKLE_FILE_NAME="MyPickle"
HOMOGRAPHY_AVERAGING = 5 # only for loaded from pickle

IN_DIR = "./input/"
DARK_IN_DIR = "./input_darkframe/"
PICKLE_DIR = "./pickles/"
DARKFRAME_COEFF = 1.0
OUT_DIR = "./output/"

stabilized_average, divider_mask_sane, imagenames, outname, homographies_data, homographies_loaded_from_pickle = None, None, None, None, None, None


def exit_handler_standard():
    global stabilized_average, divider_mask_sane, imagenames, outname, homographies_data, homographies_loaded_from_pickle

    if homographies_loaded_from_pickle is None and len(homographies_data["homographies_infos"]) > 0:
        lib.save_my_pickle(homographies_data, PICKLE_FILE_NAME)

    stabilized_average /= divider_mask_sane  # dividing sum by number of images

    outname = OUT_DIR + imagenames[0] + '-' + imagenames[-1] + '.png'

    print("exit_handler_standard(): Saving file " + outname)

    stabilized_average[stabilized_average < 0] = 0
    stabilized_average[stabilized_average > 1] = 1

    cv2.imwrite(outname, np.uint16(stabilized_average * 65535))





    # cv2.waitKey()


## MAIN
# 
#
def main():
    global stabilized_average, divider_mask_sane, imagenames, outname, homographies_data, homographies_loaded_from_pickle


    shape_original = None
    shape_no_border = None
    shape = None

    stabilized_average = None
    darkframe_image_no_border = None  # get_darkframe()

    mask_image_no_border = None
    mask_image = None

    divider_mask_no_border = None
    divider_mask = None
    divider_mask_sane = None

    transformed_image = None
    transformed_mask = None

    homography_state = [None, None, None]  # x, v, a
    homography_pred = None

    keypoint_cache = dict()
    convkernel_cache = None





    homographies_loaded_from_pickle = lib.load_my_pickle(PICKLE_FILE_NAME)

    if homographies_loaded_from_pickle is not None:
        video_input_generator = None
        for homography_info_from_pickle in homographies_loaded_from_pickle["homographies_infos"]:

            counter_from_pickle = homography_info_from_pickle["counter"]
            imgpath_from_pickle = homography_info_from_pickle["imgpath"]
            homography_from_pickle = homography_info_from_pickle["homography"]
            match_data_from_pickle = homography_info_from_pickle["match_data"]
            convkernel_cache_from_pickle = homography_info_from_pickle["convkernel_cache"]
            input_image_shape_from_pickle = homography_info_from_pickle["shape"]
            image_mode_from_pickle = homography_info_from_pickle["input_image_mode"]

            print("main(): counter_from_pickle = {}".format(counter_from_pickle))

            input_image_from_pickle = None

            if image_mode_from_pickle == 0: # FROM WEBCAM, SO PICKLED
                input_image_from_pickle = lib.load_my_pickle(imgpath_from_pickle)
            if image_mode_from_pickle == 1:
                if video_input_generator is None:
                    video_input_generator = lib.get_next_frame_from_chosen_source(image_mode_from_pickle, imgpath_from_pickle, _file_not_dir=True)
                input_image_from_pickle, _, _ = next(video_input_generator)
                input_image_from_pickle = lib.make_border_for_image(input_image_from_pickle, BORDER)
                input_image_from_pickle = cv2.resize(input_image_from_pickle,
                                                     (input_image_shape_from_pickle[1],
                                                      input_image_shape_from_pickle[0]),
                                                     interpolation=cv2.INTER_LINEAR)
            if image_mode_from_pickle == 2:
                input_image_from_pickle = next(lib.get_next_frame_from_chosen_source(image_mode_from_pickle, imgpath_from_pickle, _file_not_dir=True))
                input_image_from_pickle = cv2.resize(input_image_from_pickle,
                                                     (input_image_shape_from_pickle[1],
                                                      input_image_shape_from_pickle[0]),
                                                     interpolation=cv2.INTER_LINEAR)



            cv2.imshow("main(): input_image_from_pickle", input_image_from_pickle)

            if convkernel_cache_from_pickle is not None:
                cv2.imshow("main(): convkernel_cache_from_pickle", convkernel_cache_from_pickle)
            else:
                convkernel_cache_from_pickle = lib.make_me_a_kernel(127)

            transformed_image_pickle = lib.transform_image_to_base_image(input_image_from_pickle, homography_from_pickle)
            cv2.imshow("main(): transformed_image_pickle", transformed_image_pickle)

            cv2.waitKey(100)

    else:
        atexit.register(exit_handler_standard)

        homographies_data = dict()
        homographies_data["homographies_infos"] = []
        homographies_data["darkframe_image"] = None

        counter = 0
        imagenames = []
        for input_image_, imgname, frames_num in lib.get_next_frame_from_chosen_source(INPUT_MODE, IN_DIR, MAXFRAMES):
            homography_info = dict()

            input_image_no_border = None
            input_image = None
            match_data = None
            cv2.imshow("main(): input_image_", input_image_)

            try:
                shape_original = input_image_.shape
            except Exception as e:
                print("main(): Nope")
                if counter > 0:
                    return
                continue

            print("\nmain(): image nr {} of {}".format(counter + 1, frames_num))

            if darkframe_image_no_border is None:
                darkframe_image_no_border = lib.get_darkframe(DARK_IN_DIR, DARKFRAME_COEFF)
                darkframe_image_no_border = cv2.resize(darkframe_image_no_border, (shape_original[1], shape_original[0]),
                                                       interpolation=cv2.INTER_LINEAR)

            if darkframe_image_no_border is not None:
                input_image_no_border = input_image_ - darkframe_image_no_border

            if shape is None:
                input_image_no_border = cv2.resize(input_image_no_border, (0, 0), fx=RESIZE, fy=RESIZE,
                                                   interpolation=cv2.INTER_LINEAR)
            else:
                input_image_no_border = cv2.resize(input_image_no_border, (shape_no_border[1], shape_no_border[0]),
                                                   interpolation=cv2.INTER_LINEAR)

            input_image = lib.make_border_for_image(input_image_no_border, BORDER)

            shape_no_border = input_image_no_border.shape
            shape = input_image.shape

            if stabilized_average is None:
                stabilized_average = np.float32(input_image)

            if mask_image is None:
                mask_image_no_border = lib.make_mask_for_image(input_image_no_border, 0.0)
                mask_image = lib.make_border_for_image(mask_image_no_border, BORDER)

            if divider_mask is None:
                divider_mask = lib.make_mask_for_image(mask_image_no_border, BORDER)
                divider_mask_sane = divider_mask.copy()
                divider_mask_sane[divider_mask_sane == 0] = 1
                # divider_mask[divider_mask == 0] = 2**(-100)

            imagenames.append(imgname)
            imgpath = IN_DIR + imgname

            #if counter == 0:
            #    counter += 1
            #    continue

            input_image = cv2.resize(input_image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            transformed_image = input_image
            transformed_mask = mask_image
            homography = None

            if ALIGN:
                base = (stabilized_average / divider_mask_sane)
                base[base <= 0] = 0
                base[base >= 1] = 1
                homography, match_data = lib.compute_homography(input_image, np.array(base, dtype=input_image.dtype), _pxdistcoeff=0.9,
                                                            _dscdistcoeff=0.9, _cache=keypoint_cache, _debug=DEBUG)

                if homography_pred is not None:
                    transformed_image_nopred = lib.transform_image_to_base_image(transformed_image, homography)
                    transformed_image_pred = lib.transform_image_to_base_image(transformed_image, homography_pred)
                    cv2.imshow("main(): transformed_image_nopred", transformed_image_nopred)
                    cv2.imshow("main(): transformed_image_pred", transformed_image_pred)
                    cv2.waitKey(1)

                    homography = (homography + homography_pred) / 2

                transformed_image = lib.transform_image_to_base_image(transformed_image, homography)
                transformed_mask = lib.transform_image_to_base_image(transformed_mask, homography)

                if PREDICTION and (INPUT_MODE == 1 or INPUT_MODE == 0):
                    homography_pred, homography_state = lib.compute_predicted_H(homography, homography_state)

                if KEYPOINTS_NEIGHBORHOOD_ONLY:
                    mask_of_kp2, convkernel_cache = lib.make_mask_from_points(match_data, transformed_mask, BORDER, convkernel_cache)
                    transformed_mask *= mask_of_kp2
                    transformed_image *= mask_of_kp2
                    if DEBUG >= 1:
                        cv2.imshow("main(): maks_of_kp2", mask_of_kp2)
                        cv2.waitKey(1)

            if counter == 0: # initial conditions
                stabilized_average *= 0
                divider_mask *= 0

            stabilized_average += np.float32(transformed_image)
            divider_mask += transformed_mask
            divider_mask_sane = divider_mask.copy()
            divider_mask_sane[divider_mask_sane == 0] = 1

            if DEBUG >= 1:
                cv2.imshow("main(): stabilized_average/divider_mask", stabilized_average / divider_mask_sane)
                cv2.imshow("main(): transformed_mask", transformed_mask)
                cv2.imshow("main(): transformed_image", transformed_image)
                cv2.imshow("main(): divider_mask", divider_mask / (counter + 1))
                cv2.waitKey(1)

            if DEBUG >= 3:
                cv2.imwrite(OUT_DIR + imgname + '_DEBUG.jpg', transformed_image)
                cv2.imwrite(OUT_DIR + imgname + '_divider_maskDEBUG.png',
                            np.uint16(divider_mask_sane * 255 / (counter + 1)))

            homography_info["counter"] = counter
            homography_info["shape"] = shape
            homography_info["homography"] = homography
            homography_info["match_data"] = match_data
            homography_info["convkernel_cache"] = convkernel_cache
            homography_info["input_image_mode"] = INPUT_MODE

            if INPUT_MODE == 0 or ALWAYS_PICKLE_FRAMES: # if webcam we cant read it again :(
                pickle_imgname = "{}{}_{}".format(PICKLE_DIR, PICKLE_FILE_NAME, counter)
                lib.save_my_pickle(input_image, pickle_imgname)
                homography_info["imgpath"] = pickle_imgname
            else:
                homography_info["imgpath"] = imgpath

            homographies_data["homographies_infos"].append(homography_info)

            counter += 1

    # exit_handler()


# main
if __name__ == '__main__':
    main()
