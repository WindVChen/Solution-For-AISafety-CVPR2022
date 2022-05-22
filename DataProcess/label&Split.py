import os
import numpy as np
import glob
import shutil
import random

def split(totalPath, ratio):
    finedir = os.path.join(totalPath, "images")
    if os.path.isdir(finedir):
        print("Starting Split {} ********".format(os.path.basename(finedir)))
        trainCal = 0
        validCal = 0
        trainCreate = finedir.replace("images", "train")
        removeAug(trainCreate)
        validCreate = finedir.replace("images", "val")
        removeAug(validCreate)


        allImages = os.listdir(finedir)
        with open(os.path.join(totalPath, "label.txt"), "r") as f:
            labels = f.readlines()
            for i in range(0, len(labels)):
                labels[i] = labels[i].rstrip('\n')
        labelsDict = {}
        for i in range(0, len(labels)):
            k, v = labels[i].split(" ")
            labelsDict.update({k:v})

        random.shuffle(allImages)

        splitRatio = [int(x*len(allImages)) for x in ratio]
        with open(os.path.join(trainCreate, "train.txt"), 'w') as f:
            for index in range(splitRatio[0]):
                shutil.copy(os.path.join(finedir, allImages[index]), trainCreate)
                trainCal+=1
                f.write("{{\"filename\": \"{}\", \"label\": {}, \"label_name\": \"{}\"}}\n".format(allImages[index], labelsDict[allImages[index]], labelsDict[allImages[index]]))

        with open(os.path.join(validCreate, "val.txt"), 'w') as f:
            for index in range(splitRatio[0], len(allImages)):
                shutil.copy(os.path.join(finedir, allImages[index]), validCreate)
                validCal+=1
                f.write("{{\"filename\": \"{}\", \"label\": {}, \"label_name\": \"{}\"}}\n".format(allImages[index], labelsDict[allImages[index]], labelsDict[allImages[index]]))


        print("Total:{} Train:{} Valid:{} \n".format(len(allImages), trainCal, validCal))

    pass

def removeAug(path):
    if(os.path.isdir(path)):
        images = glob.glob(path+"/*.JPEG")
        for image in images:
            os.remove(image)
            print("del ", image)
        labs = glob.glob(path + "/*.txt")
        for lab in labs:
            os.remove(lab)
            print("del ", lab)
    pass


if __name__ == "__main__":
    totalPath = r"D:\Track"
    ratio = [0.75, 0.]  # train, valid, test

    split(totalPath, ratio)