import cv2, os, shutil, time
import numpy as np
from random import shuffle
import random
from config import configData


classes = configData["classes"]
trainImagesPath = configData["trainImagesPath"]
trainAnnotationsPath = configData["trainAnnotationsPath"]
trainPatchesPath = configData["trainPatchesPath"]
valImagesPath = configData["valImagesPath"]
valAnnotationsPath = configData["valAnnotationsPath"]
valPatchesPath = configData["valPatchesPath"]

# Extract training image patches
def extractImagePatches(imagesPath, annotationsPath, imagePatchesPath, numSamples):

    for className in classes:
        if(os.path.isdir(os.path.join(imagePatchesPath, className)) == True):
            shutil.rmtree(os.path.join(imagePatchesPath, className))
        os.makedirs(os.path.join(imagePatchesPath, className))

    totalSamples = numSamples["background"] + numSamples["cell_center"] + numSamples["cell_innerboundary"] + numSamples["cell_outerboundary"]

    width = 1280
    height = 960
    (patchWidth, patchHeight) = (configData["patchWidth"], configData["patchHeight"])  # window size
    (padSizeW, padSizeH) = (int(patchWidth / 2), int(patchHeight / 2))
    erosionKernel = np.ones((10, 10), np.uint8)
    dilationKernel = np.ones((30, 30), np.uint8)

    images = []
    gsImages = []
    gsErodedImages = []
    gsDilatedImages = []

    imageFiles = sorted(os.listdir(imagesPath))
    annotationFiles = sorted(os.listdir(annotationsPath))
    for ii in range(len(imageFiles)):
        imageFile = os.path.join(imagesPath, imageFiles[ii])
        annotationFile = os.path.join(annotationsPath, annotationFiles[ii])

        im = cv2.imread(imageFile)  # image
        gsIm = cv2.imread(annotationFile)  # gold standard image
        gsImEroded = cv2.erode(gsIm, erosionKernel, iterations=1)  # eroded gold standard image
        gsImDilated = cv2.dilate(gsIm, dilationKernel, iterations=1)  # dilated gold standard image

        images.append(im)
        gsImages.append(gsIm)
        gsErodedImages.append(gsImEroded)
        gsDilatedImages.append(gsImDilated)

    # Candidate center coordinates of image patches which are to be extracted from training image files
    imageIds = np.arange(0, len(imageFiles), 1)
    x = np.arange(patchWidth, width - patchWidth, 1)
    y = np.arange(patchHeight, height - patchHeight, 1)
    shuffle(imageIds)
    shuffle(x)
    shuffle(y)

    imageCount = 0
    selectedPatchCoords = []
    meanImage = np.zeros((patchHeight, patchWidth, 3))

    print("Image patches are being saved ...")

    # Random patch extraction from an whole image for training dataset
    while (numSamples["background"] > 0 or
           numSamples["cell_center"] > 0 or
           numSamples["cell_innerboundary"] > 0 or
           numSamples["cell_outerboundary"] > 0):

        ii = random.choice(imageIds)
        xx = random.choice(x)
        yy = random.choice(y)

        # If patch coordinate is already selected, pick another random coordinate
        if ((ii, xx, yy) in selectedPatchCoords):
            continue

        selectedPatchCoords.append((ii, xx, yy))

        im = images[ii]
        gsIm = gsImages[ii]
        gsImEroded = gsErodedImages[ii]
        gsImDilated = gsDilatedImages[ii]

        top = yy - padSizeH
        bottom = yy + padSizeH
        left = xx - padSizeW
        right = xx + padSizeW
        imPatch = im[top:bottom, left:right]
        gsImPatch = gsIm[top:bottom, left:right]
        gsImErodedPatch = gsImEroded[top:bottom, left:right]
        gsImDilatedPatch = gsImDilated[top:bottom, left:right]
        centerPixelValGsEroded = gsImErodedPatch[(int)(patchHeight / 2), (int)(patchWidth / 2), 0]
        centerPixelValGsDilated = gsImDilatedPatch[(int)(patchHeight / 2), (int)(patchWidth / 2), 0]
        centerPixelValGs = gsImPatch[(int)(patchHeight / 2), (int)(patchWidth / 2), 0]

        # Classifying image patches as "background", "cell center", "cell innerboundary", and "cell outerboundary" based on their center pixel value
        if (centerPixelValGsEroded == 255 and numSamples["cell_center"] != 0):
            cv2.imwrite(os.path.join(imagePatchesPath, classes[1], "patch") + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamples["cell_center"] -= 1
            imageCount += 1

        elif (centerPixelValGs == 255 and numSamples["cell_innerboundary"] != 0):
            cv2.imwrite(os.path.join(imagePatchesPath, classes[2], "patch") + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamples["cell_innerboundary"] -= 1
            imageCount += 1

        elif (centerPixelValGsDilated == 255 and numSamples["cell_outerboundary"] != 0):
            cv2.imwrite(os.path.join(imagePatchesPath, classes[3], "patch") + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamples["cell_outerboundary"] -= 1
            imageCount += 1

        elif (numSamples["background"] != 0):
            cv2.imwrite(os.path.join(imagePatchesPath, classes[0], "patch") + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamples["background"] -= 1
            imageCount += 1

    meanImage /= totalSamples
    meanVals = np.mean(meanImage, axis=(0, 1))
    print("Image patches are saved to a file. Mean value:", meanVals)


def main():
    numSamplesTrain = configData["numSamplesTrain"]
    numSamplesVal = configData["numSamplesVal"]
    extractImagePatches(trainImagesPath, trainAnnotationsPath, trainPatchesPath, numSamplesTrain)
    extractImagePatches(valImagesPath, valAnnotationsPath, valPatchesPath, numSamplesVal)


if __name__ == "__main__":

    start_time = time.perf_counter()

    main()

    elapsed_time = (time.perf_counter() - start_time) * 1000

    print("Elapsed time: %.3f" % elapsed_time, "ms")
