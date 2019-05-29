import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class ChestXrayDataset(Dataset):
    def __init__(self, root, image_list_file, train=True, transform=None,
                 age_thresh=20,
                 synth_root="",
                 oversample=False,
                 undersample=False,
                 class_indices=None):
        """
        there are two types of image_list_file
        (train_val, test)_list.txt: image name + 14 labels for diseases
        (train_val, test)_list2.txt: image name + 14 labels for diseases
                                     + 3 additional labels
        (additional labels are Patient Age, Patient Gender (M or F),
                               View Position (PA or AP))
        Refer to
        https://www.kaggle.com/nih-chest-xrays/data/version/3#Data_Entry_2017.csv
        for more details

        train: 78468
        val:   11219
        test:  22433
        ------------
        sum:  112120
        """
        self.val_num = 11219
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                if len(items) == 15:
                    label = items[1:]
                    label = [int(i) for i in label]
                else:
                    label = self.label_transform_(items[1:], age_thresh)
                image_name = os.path.join(root, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = np.array(image_names)
        self.labels = np.array(labels)
        self.transform = transform

        if train:
            self.train_image_names = self.image_names[:-self.val_num]
            self.val_image_names = self.image_names[-self.val_num:]
            self.train_labels = self.labels[:-self.val_num]
            self.val_labels = self.labels[-self.val_num:]
            if synth_root != "":
                self.add_synth_images(class_indices, synth_root)
            elif oversample:
                self.oversample(class_indices)
            elif undersample:
                self.undersample(class_indices)
            self.image_names = np.append(self.train_image_names,
                                         self.val_image_names)
            self.labels = np.append(self.train_labels, self.val_labels, axis=0)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

    def label_transform_(self, label, age_thresh):
        """
        0-13: diseases
        14: Age > age_thresh -> 0
        15: M -> 0
        16: PA -> 0
        """
        for i, e in enumerate(label):
            if i <= 13:
                label[i] = int(e)
            elif i == 14:
                label[i] = 0 if int(e) > age_thresh else 1
            else:
                label[i] = 0 if e == 'M' or e == 'PA' else 1
        return label

    def calc_class_sum(self, class_indices):
        """
        class_sum (return value) is expected to be like below
        [(# of images with class_indices[0]),
         (# of images with class_indices[1]),
         ...
         (# of images with class_indices[len(class_indices)-1]),
         (# of images without any class in class_indices)]
        """

        class_sum = np.zeros(len(class_indices) + 1)
        # no label indices for undersample
        self.no_label_indices = []
        for i, label in enumerate(self.train_labels):
            if sum(label[class_indices]) == 0:
                self.no_label_indices.append(i)
                class_sum[-1] += 1
            else:
                class_sum[:-1] += label[class_indices]
        print(class_sum)
        return class_sum

    def oversample(self, class_indices):
        """
        assuming no disease images are the most
        oversample images so that # of images with every label in class_indices
        is equal.
        """
        class_sum = self.calc_class_sum(class_indices)
        num_adding_images = max(class_sum) - class_sum
        assert class_sum[-1] == 0
        selected_indices = list(range(len(self.train_image_names)))
        for i, num in enumerate(num_adding_images[:-1]):
            images_with_label_indices = \
                np.where(self.train_labels[:, class_indices[i]] == 1)[0]
            print(i, len(images_with_label_indices))
            selected_indices += list(np.random.choice(
                                        images_with_label_indices,
                                        int(num)))
        self.train_image_names = self.train_image_names[selected_indices]
        self.train_labels = self.train_labels[selected_indices]

    def undersample(self, class_indices):
        """
        undersample so that # of images with every label in class_indices
        including no label images is equal
        """
        class_sum = self.calc_class_sum(class_indices)
        selected_indices = []
        for i, num in enumerate(class_sum[:-1]):
            images_with_label_indices = \
                np.where(self.train_labels[:, class_indices[i]] == 1)[0]
            print(i, len(images_with_label_indices))
            selected_indices += list(np.random.choice(
                                        images_with_label_indices,
                                        int(min(class_sum)),
                                        replace=False))
        selected_indices += list(np.random.choice(
                                    self.no_label_indices,
                                    int(min(class_sum)),
                                    replace=False))
        self.train_image_names = self.train_image_names[selected_indices]
        self.train_labels = self.train_labels[selected_indices]

    def add_synth_images(self, class_indices, synth_root):
        """
        assuming directory tree is like below
        synth_root
        |- 01 (images with class_indices[0])
        |- 02 (images with class_indices[1])
        ...
        |- mn (no label images)
        where mn is 2 digit number and mn = len(class_indices) + 1
        """
        class_sum = self.calc_class_sum(class_indices)
        num_adding_images = max(class_sum) - class_sum
        for i, num in enumerate(num_adding_images):
            # get images from dir i
            img_dir = os.path.join(synth_root, '%02d' % (i+1))
            files = os.listdir(img_dir)
            files_file = [os.path.join(img_dir, f)
                          for f in files
                          if os.path.isfile(os.path.join(img_dir, f))]
            self.train_image_names = np.append(self.train_image_names,
                                               files_file[:int(num)])
            labels = np.zeros((int(num), len(self.train_labels[0])))
            if i != len(num_adding_images)-1:
                labels[:, class_indices[i]] = 1
            self.train_labels = np.append(self.train_labels, labels, axis=0)
