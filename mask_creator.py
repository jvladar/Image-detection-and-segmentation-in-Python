from pycocotools.coco import COCO
import numpy as np
import os
import matplotlib.pyplot as plt

class MaskCreator():
    def __init__(self):
        self.dataDir = 'coco'
        self.dataType='images'
        self.annFile='{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)

        self.coco = COCO(self.annFile)
        # Load the categories in a variable
        catIDs = self.coco.getCatIds()
        self.categories = self.coco.loadCats(catIDs)
        # self.images = os.listdir(image_dir)

    def __findImageCocoObj__(self, className, imgName):
        catIds  = self.coco.getCatIds(catNms=className)
        imgIds  = self.coco.getImgIds(catIds=catIds)
        images  = []
        images += self.coco.loadImgs(imgIds)

        relevantImg = None
        for i in range(len(images)):
            # print("images", images[i])
            if images[i]["file_name"] == imgName:
                relevantImg = images[i]
                break
        return relevantImg

    def __createMaskInner__(self, className, imgObj, generate_files=False):
        catIds = self.coco.getCatIds(catNms=className)
        annIds = self.coco.getAnnIds(imgIds=imgObj['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        mask = np.zeros((imgObj['height'], imgObj['width']))
        for m in range(len(anns)):
            mask = np.maximum(self.coco.annToMask(anns[m]), mask)
        # plt.imshow(mask)

        if generate_files:
            plt.imshow(mask)
            plt.axis('off')
            if not os.path.exists("Masks"):
                os.mkdir("Masks")
            plt.savefig("Masks" + '/' + imgObj["file_name"] + "Ms.png", bbox_inches='tight',pad_inches = 0)
            plt.close()

        return mask

    # returns mask and className
    def __createMask__(self, imageName, generate_files=False):
        imgObj = None
        className = None
        # classIndex = None
        # find imageCocoObj
        for i in range (len(self.categories)):
            imgObj = self.__findImageCocoObj__(self.categories[i]["name"], imageName)
            if imgObj != None:
                # classIndex = i
                className = self.categories[i]["name"]
                break
        
        return self.__createMaskInner__(className, imgObj, generate_files), className
