import os
from config import configData


classes = configData["classes"]
trainImagesPath = configData["trainImagesPath"]
trainAnnotationsPath = configData["trainAnnotationsPath"]
trainPatchesPath = configData["trainPatchesPath"]
valImagesPath = configData["valImagesPath"]
valAnnotationsPath = configData["valAnnotationsPath"]
valPatchesPath = configData["valPatchesPath"]


def createLabels(fileName, imagePath, classes):

    fileName = open(fileName, "w+")

    for ii in range(0, len(classes)):
        className = classes[ii]
        imageFiles = sorted(os.listdir(os.path.join(imagePath, className)))
        for jj in range(0, len(imageFiles)):
            fileName.write(os.path.join(imagePath, className, imageFiles[jj]) + " " + str(ii) + "\n")

    fileName.close()


createLabels("trainLabels.txt", trainPatchesPath, classes)
createLabels("valLabels.txt", valPatchesPath, classes)