## image-align-and-average

Python application for image noise removal by aligning and averaging many images.

It uses OpenCV and NumPy.

OpenCV is used for image analysis (feature detection, matching, finding homography, transforming).

NumPy is used for image processing and averaging. 

Images are processed with 16bit per channel accuracy.

Samples included.


### usage

1. place images in *input* directory
1. optionally put 16bpp raster darkframe into *input_darkframe* dir
1. adjust some params in script (at the moment this is the only way to customize processing)
1. run application
1. wait until it finishes or press Ctrl + C, it should exit gracefully and save output image
1. check *output* directory

### sample results

YouTube videos:
  * [Image alignment](https://www.youtube.com/watch?v=hhg6CC40sQQ)
  * [Image denoising by averaging](https://www.youtube.com/watch?v=gHukdX8MLIU)

Divided view, upper-left - original, bottom-right - averaged, center boosted 8x. Images were taken with smartphone camera:
![result](https://raw.githubusercontent.com/michal2229/image-align-and-average/master/results/IMG_20160402_210944_40-images_boost-8x_comparision_small.jpg)
