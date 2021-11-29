import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from mask_creator import MaskCreator

class SeaFlowersDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.maskCreator = MaskCreator()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask, className = self.maskCreator.__createMask__(self.images[index])

        if className == "Mytilus":
            mask[mask == 1.0] = 2.0

        # check unique values
        # uniq_values = []
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i][j] not in uniq_values:
        #             uniq_values.append(mask[i][j])
        # print(uniq_values)
        # print("delka", len(uniq_values))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
