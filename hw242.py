from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## For visualizing results
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

dataDir='coco'
dataType='val'  # todo, change for train here
annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)

ukazka_part_1 = False
if ukazka_part_1:
    # initialize the COCO api for instance annotations
    coco=COCO(annFile)

    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)

    nms=[cat['name'] for cat in cats]
    print(len(nms),'COCO categories: {}'.format(' '.join(nms)))

    def getClassName(classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"

    print('The class name is', getClassName(1, cats))  # should produce Mytilus

    filterClasses = ['Mytilus']#, 'Zostera']
    # Fetch class IDs only corresponding to the filterClasses
    catIds = coco.getCatIds(catNms=filterClasses)
    # Get all images containing the above Category IDs
    imgIds = coco.getImgIds(catIds=catIds)
    print("Number of images containing all required classes:", len(imgIds))

    # load and show random img
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    # I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # Load and display instance annotations
    # plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)

    classes = ['Mytilus', 'Zostera']
    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given class
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
        
    # Now, filter out the repeated images    
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    dataset_size = len(unique_images)

    print("Number of images containing the filter classes:", dataset_size)

    """
    4. Image Segmentation Mask Generation
    """
    filterClasses = ['Mytilus', 'Zostera']
    mask = np.zeros((img['height'],img['width']))
    for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)
        pixel_value = filterClasses.index(className)+1
        mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
    # plt.imshow(mask)
    # plt.show()

    print('Unique pixel values in the mask are:', np.unique(mask))

    # # Binary Semantic Segmentation Mask
    # mask = np.zeros((img['height'],img['width']))
    # for i in range(len(anns)):
    #     mask = np.maximum(coco.annToMask(anns[i]), mask)
    # plt.imshow(mask)
    # print('Unique pixel values in the mask are:', np.unique(mask))

def filterDataset(folder, classes=None, mode='train'):    
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(annFile)
    
    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
    
    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])
            
    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    
    return unique_images, dataset_size, coco

folder = 'coco'
classes = ['Zostera', 'Mytilus']
mode = 'val'
images, dataset_size, coco = filterDataset(folder, classes, mode)

"""
(b) Generate the images and masks
"""

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    # train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    print(img_folder + '/' + imageObj['file_name'])
    train_img = cv2.imread(img_folder + '/' + imageObj['file_name'])#/255.0
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img
    
def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)+1
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask  
    
def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask

def dataGeneratorCoco(images, classes, coco, folder, 
                      input_image_size=(224,224), batch_size=4, mode='train', mask_type='binary'):
    
    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    
    c = 0
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            
            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)
            
            ### Create Mask ###
            if mask_type=="binary":
                train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)
            
            elif mask_type=="normal":
                train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)                
            
            # Add to respective batch sized arrays
            img[i-c] = train_img
            mask[i-c] = train_mask
            
        c+=batch_size
        if(c + batch_size >= dataset_size):
            c=0
            random.shuffle(images)
        yield img, mask

batch_size = 4
input_image_size = (224,224)
mask_type = 'normal'
def visualizeGenerator(gen):
    img, mask = next(gen)
    
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
    
    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:,:,0])
                
            ax.axis('off')
            fig.add_subplot(ax)        
    plt.show()

val_gen = dataGeneratorCoco(images, classes, coco, folder,
                            input_image_size, batch_size, mode, mask_type)
# visualizeGenerator(val_gen)

def augmentationsGenerator(gen, augGeneratorArgs, seed=None):
    # Initialize the image data generator with args provided
    image_gen = ImageDataGenerator(**augGeneratorArgs)
    
    # Remove the brightness argument for the mask. Spatial arguments similar to image.
    augGeneratorArgs_mask = augGeneratorArgs.copy()
    _ = augGeneratorArgs_mask.pop('brightness_range', None)
    # Initialize the mask data generator with modified args
    mask_gen = ImageDataGenerator(**augGeneratorArgs_mask)
    
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    
    for img, mask in gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation of the images 
        # will end up different from the augmentation of the masks
        g_x = image_gen.flow(255*img, 
                             batch_size = img.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = mask_gen.flow(mask, 
                             batch_size = mask.shape[0], 
                             seed = seed, 
                             shuffle=True)
        
        img_aug = next(g_x)/255.0
        
        mask_aug = next(g_y)
                   

        yield img_aug, mask_aug

augGeneratorArgs = dict(featurewise_center = False, 
                        samplewise_center = False,
                        rotation_range = 5, 
                        width_shift_range = 0.01, 
                        height_shift_range = 0.01, 
                        brightness_range = (0.8,1.2),
                        shear_range = 0.01,
                        zoom_range = [1, 1.25],  
                        horizontal_flip = True, 
                        vertical_flip = False,
                        fill_mode = 'reflect',
                        data_format = 'channels_last')

aug_gen = augmentationsGenerator(val_gen, augGeneratorArgs)
# visualizeGenerator(aug_gen)

val_gen_train = dataGeneratorCoco(images, classes, coco, dataDir, input_image_size, batch_size, 'train')
val_gen_val = dataGeneratorCoco(images, classes, coco, dataDir, input_image_size, batch_size, 'val')

aug_gen_train = augmentationsGenerator(val_gen_train, augGeneratorArgs)
aug_gen_val = augmentationsGenerator(val_gen_val, augGeneratorArgs)

# raise "chyba"
# Create filtered train dataset (using filterDataset()) 
# Create filtered val dataset (using filterDataset()) 

# Create train generator (using dataGeneratorCoco()) 
# Create train generator (using dataGeneratorCoco()) 

# Set your parameters
n_epochs = 2
steps_per_epoch = 15 // batch_size
validation_steps = 6 // batch_size

m = tf.keras.applications.MobileNetV2(input_shape=[224,224,3],
                                      include_top=False)
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile your model first
m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Start the training process
history = m.fit(x = aug_gen_train,
                validation_data = aug_gen_val,
                steps_per_epoch = steps_per_epoch,
                validation_steps = validation_steps,
                epochs = n_epochs,
                verbose = True)
