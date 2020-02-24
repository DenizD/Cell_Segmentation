import caffe

import numpy as np
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter

import random

class customDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        # Prototxt params
        # Image params
        params = eval(self.param_str)

        self.labelFile = params["labelFile"]
        self.batchSize = params["batchSize"]
        self.imageWidth = params["imageWidth"]
        self.imageHeight = params["imageHeight"]
        self.numChannels = params["numChannels"]
        self.meanData = params["meanData"]

        # Data Augmentation params
        self.train = params["train"]
        self.rotateProb = params["rotateProb"]
        self.rotateAngle = params["rotateAngle"]
        self.mirrorProb = params["mirrorProb"]
        self.applyRandomFilter = params["applyRandomFilter"]
        self.jitterProb = params["jitterProb"]
        self.jitterVal = params["jitterVal"]

        self.numSamples = sum(1 for line in open(self.labelFile))
        self.iterNo = 0
        self.batchCount = 0

        print("labelFile:", self.labelFile)
        print("batchSize:", self.batchSize)
        print("numChannels:", self.numChannels)
        print("meanData:", self.meanData)
        print("train:", self.train)
        print("rotateProb:", self.rotateProb)
        print("rotateAngle:", self.rotateAngle)
        print("mirrorProb:", self.mirrorProb)
        print("applyRandomFilter:", self.applyRandomFilter)
        print("jitterProb:", self.jitterProb)
        print("jitterVal:", self.jitterVal)

        self.allIds = np.arange(self.numSamples)
        random.shuffle(self.allIds)
        self.allImageFiles = ["" for x in range(self.numSamples)]
        self.allLabels = np.zeros(self.numSamples)
        with open(self.labelFile, "r") as annsfile:
            for c, i in enumerate(annsfile):
                data = i.split(" ")
                self.allImageFiles[c] = data[0]
                self.allLabels[c] = int(data[1])

        self.batchIds = np.zeros((self.batchSize))
        self.batchData = np.zeros((self.batchSize, self.numChannels, self.imageWidth, self.imageHeight))
        self.batchLabels = np.zeros((self.batchSize))

        top[0].reshape(self.batchSize, self.numChannels, self.imageWidth, self.imageHeight)
        top[1].reshape(self.batchSize)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):

        # Load image batches and their labels
        if (self.batchSize * self.batchCount >= self.numSamples):
            random.shuffle(self.allIds)
            self.batchCount = 0

        for ii in range(0, self.batchSize):
            self.batchIds[ii] = self.allIds[(self.iterNo * self.batchSize + ii) % len(self.allIds)]

        for ii in range(0, self.batchSize):
            self.batchData[ii] = self.loadImage(self.allImageFiles[(int)(self.batchIds[ii])])
            self.batchLabels[ii] = self.allLabels[(int)(self.batchIds[ii])]

        # assign output
        top[0].data[...] = self.batchData
        top[1].data[...] = self.batchLabels

        self.iterNo += 1
        self.batchCount += 1

    def backward(self, top, propagate_down, bottom):
        pass

    def loadImage(self, imageFile):
        im = Image.open(imageFile)

        if self.train:

            # Data Augmentation
            if (self.rotateProb is not 0):
                im = self.rotateIm(im)

            if (self.mirrorProb is not 0):
                im = self.mirrorIm(im)

            if (self.applyRandomFilter):
                im = self.applyFilter(im)

            if (self.jitterProb is not 0):
                im = self.rgbJitter(im)

        # Preprocessing (RGB to BGR, mean subtraction, reshape)
        im = np.array(im, dtype=np.float32)
        im = im[:, :, (2,1,0)]
        im -= self.meanData
        im = im.transpose((2, 0, 1))

        return im

    def mirrorIm(self, im):
        if (random.random() > self.mirrorProb):
            return im
        return ImageOps.mirror(im)

    def rotateIm(self, im):
        if (random.random() > self.rotateProb):
            return im
        return im.rotate(random.randint(-self.rotateAngle, self.rotateAngle))

    def applyFilter(self, im):
        filterType = random.randint(0,5)
        if (filterType == 0):
            im = im.filter(ImageFilter.GaussianBlur)
        elif (filterType == 1):
            im = im.filter(ImageFilter.MinFilter)
        elif (filterType == 2):
            im = im.filter(ImageFilter.MaxFilter)
        elif (filterType == 3):
            im = im.filter(ImageFilter.MedianFilter)
        elif (filterType == 4):
            im = im.filter(ImageFilter.ModeFilter)
        elif (filterType == 5):
            im = im.filter(ImageFilter.UnsharpMask)
        return im

    def rgbJitter(self, im):

        if (random.random() > self.jitterProb):
            return im

        im = np.array(im)
        ch = random.randint(0,2)
        im[:,:,ch] += random.randint(0, self.jitterVal)
        im = Image.fromarray(im, "RGB")

        return im