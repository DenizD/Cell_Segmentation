import caffe
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.misc
from config import configData

patchWidth = configData["patchWidth"]
patchHeight = configData["patchHeight"]

def initModel(modelFile, deployFile):

    caffe.set_device(0)
    caffe.set_mode_gpu()

    # load the model
    net = caffe.Net(deployFile, modelFile, caffe.TEST)

    net.blobs["data"].reshape(1, 3, patchWidth, patchHeight)

    return net

def preprocessIm(im, meanData):
    im = np.array(im, dtype=np.float32)
    im = im[:, :, (2, 1, 0)] # RGB to BGR conversion
    im -= meanData # mean subtraction
    im = im.transpose((2, 0, 1)) # set channel order from (H,W,C) to (C,H,W)

    return im

def classifyIm(im, net):
    net.blobs["data"].data[...] = im

    # forward propagation
    out = net.forward()

    # predict class index and prob
    class_index = out["prob"].argmax()
    class_probs = out["prob"][0]

    return class_index, class_probs

def main():

    deployFile = configData["deployProtoFile"]
    meanData = configData["meanVal"]
    modelFile = configData["weightFile"]
    imageFile = configData["testImageFile"]

    net = initModel(modelFile, deployFile)

    (padSizeW, padSizeH) = (int(patchWidth / 2), int(patchHeight / 2))

    im = Image.open(imageFile)
    im = np.array(im, dtype=np.float32)

    im = cv2.copyMakeBorder(im, padSizeH, padSizeH, padSizeW, padSizeW, cv2.BORDER_REFLECT)
    imHeight = im.shape[0]
    imWidth = im.shape[1]

    estimatedLabels = np.zeros((imHeight-patchHeight, imWidth-patchWidth)) # estimated class labels

    # classify image patches using sliding window approach for a sample imagefile, and generate estimated labels at the output
    for y in range(0, imHeight-patchHeight, 1):
        print(y)
        for x in range(0, imWidth-patchWidth, 1):
            imPatch = im[y:y + patchHeight, x:x + patchWidth, :]
            imPatch = preprocessIm(imPatch, meanData)
            [class_index, class_probs] = classifyIm(imPatch, net)
            estimatedLabels[y, x] = class_index

    scipy.misc.imsave("estimatedLabels.jpg", estimatedLabels)

if __name__ == "__main__":
    main()
