from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

dataDir='coco'
dataType='images'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# Initialize the COCO api for instance annotations
coco = COCO(annFile)

# Load the categories in a variable
catIDs = coco.getCatIds()
categories = coco.loadCats(catIDs)

print ("Categories:", categories)

def saving(className, generate_files=True):
    images = []
    if className != None:
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
    dataset_size = len(imgIds)

    if not generate_files:
        return unique_images

    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        I = io.imread('{}/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0

        plt.imshow(I)
        plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        if not os.path.exists("Frames"):
            os.mkdir("Frames")
        plt.savefig("Frames" + '/' + str(i) + className + "Fr.png", bbox_inches='tight',pad_inches = 0)
        plt.close()

        print("Number of images containing the filter classes:", dataset_size)

        mask = np.zeros((img['height'],img['width']))
        for m in range(len(anns)):
            mask = np.maximum(coco.annToMask(anns[m]), mask)
        plt.imshow(mask)
        plt.axis('off')
        if not os.path.exists("Masks"):
            os.mkdir("Masks")
        plt.savefig("Masks" + '/' + str(i) + className + "Ms.png", bbox_inches='tight',pad_inches = 0)
        plt.close()
    return unique_images

uniq_im = None
for i in range (len(categories)):
    uniq_im_tmp = saving(categories[i]["name"], generate_files=False)
    if uniq_im is None:
        uniq_im = uniq_im_tmp
    else:
        uniq_im.append(uniq_im_tmp)

#print('Unique pixel values in the mask are:', np.unique(mask))

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    # train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    train_img = cv2.imread(img_folder + '/' + imageObj['file_name'])#/255.0
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)

    if (len(train_img.shape) == 3 and train_img.shape[2] == 3): # If it's a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img

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
                      input_image_size=(224,224), batch_size=4,  # todo edit image size according to selected model
                      mode='', mask_type='binary'):  # todo odebrat mask_type
    img_folder = '{}/images/{}'.format(folder, mode)
    # print("zacatek generator", images[0], len(images))
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    
    c = 0
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            
            ### Retrieve Image
            if not os.path.exists(img_folder + '/' + imageObj['file_name']):
                continue

            train_img = getImage(imageObj, img_folder, input_image_size)
            
            ### Create Mask
            # if mask_type=="binary":
            train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)              
            
            # Add to respective batch sized arrays
            img[i-c] = train_img
            mask[i-c] = train_mask
            
        c += batch_size
        if (c + batch_size >= dataset_size):  # todo nejspis zbytecne
            c = 0
            random.shuffle(images)
        yield img, mask

def visualizeGenerator(gen):
    # Iterate the generator to get image and mask batches
    img, mask = next(gen)
 
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
   
    for i in range(2):        
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:,:,0])
                
            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()

batch_size = 8
input_image_size = (224, 224)
# mode = 'train'

# val_gen = dataGeneratorCoco(uniq_im, categories, coco, dataDir, input_image_size, batch_size)
val_gen_train = dataGeneratorCoco(uniq_im, categories, coco, dataDir, input_image_size, batch_size, 'train')
val_gen_val = dataGeneratorCoco(uniq_im, categories, coco, dataDir, input_image_size, batch_size, 'val')

# visualizeGenerator(val_gen)

"""
AUGMENTATION
"""

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

aug_gen_train = augmentationsGenerator(val_gen_train, augGeneratorArgs)
aug_gen_val = augmentationsGenerator(val_gen_val, augGeneratorArgs)

# visualizeGenerator(aug_gen)

""""
APPLY ON MODEL
"""
NUMBER_OF_EPOCHS = 2
n_epochs = NUMBER_OF_EPOCHS
IMG_SHAPE = input_image_size + (3,)

dataset_size_train = 15
dataset_size_val = 8
steps_per_epoch = dataset_size_train // batch_size
validation_steps = dataset_size_val // batch_size
model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                          include_top=False,
                                          weights='imagenet')
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x = aug_gen_train,
                    validation_data = aug_gen_val,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps = validation_steps,
                    epochs = n_epochs,
                    verbose = True)