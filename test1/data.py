import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Dataset


class ImageDataset(Dataset):

    def __init__(
        self,
        root: str,
        # istrain: bool,
        # data_size: int,
        return_index: bool = False,
    ):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """basic information"""
        self.root = root
        # self.data_size = data_size
        self.return_index = return_index
        # self.istrain = istrain

        """ declare data augmentation """
        # normalize = transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225]
        #         )
        #
        # 448:600
        # 384:510
        # 768:
        # if istrain:
        #     # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
        #     # RandAugment(n=2, m=3, img_size=sub_data_size)
        #     self.transforms = transforms.Compose([
        #                 transforms.Resize((510, 510), Image.BILINEAR),
        #                 transforms.RandomCrop((data_size, data_size)),
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
        #                 transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        #                 transforms.ToTensor(),
        #                 normalize
        #         ])
        # else:
        #     self.transforms = transforms.Compose([
        #                 transforms.Resize((510, 510), Image.BILINEAR),
        #                 transforms.CenterCrop((data_size, data_size)),
        #                 transforms.ToTensor(),
        #                 normalize
        #         ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)
        self.transform = transforms.ToTensor()
        # print(self.data_infos)

    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort()  # sort by alphabet
        # print("[dataset] class number:", len(folders))
        eye = np.eye(len(folders), dtype=np.float32)
        for class_id, folder in enumerate(folders):
            files = os.listdir(root + folder)
            class_files = root + folder
            for file in files:
                data_path = class_files + "/" + file
                data_infos.append({"path": data_path, "label": eye[class_id]})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        # print("istrain: ", self.istrain)
        # img = cv2.imread(image_path)
        # print("DEBUG:", img.shape)
        # img = img[:, :, ::-1] # BGR to RGB.
        # print("RGB:  ", img.shape)

        # to PIL.Image
        # img = Image.fromarray(img)
        img = self.transform(np.load(image_path).astype(np.float32))
        # print(img.shape)
        # print("befortransforms:  ", img.size)
        # img = self.transforms(img)
        # print("after transforms: ", img.shape)

        # istrain:  True
        # DEBUG: (354, 500, 3)
        # RGB:   (354, 500, 3)
        # befortransforms:   (500, 354)
        # after transforms:  torch.Size([3, 384, 384])

        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label

        # return img, sub_imgs, label, sub_boundarys
        return img, label
