import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd

class Tiny_Imagenet:
    def __init__(self, root, train_transform=None, test_transform=None):
        super(Tiny_Imagenet, self).__init__()
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None
        self.root = root

    def get_data(self):
        train_list_img, train_list_label, test_list_img, test_list_label = [], [], [], []
        train_path = os.path.join(self.root, 'train/')
        class_path = os.listdir(train_path)

        test_path = os.path.join(self.root, 'val/')
        val_data = pd.read_csv(test_path + 'val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])


        for i in range(len(class_path)):
            class_temp = os.path.join(train_path, class_path[i], 'images/')
            img_path = os.listdir(class_temp)
            for j in range(len(img_path)):
                img_path_temp = os.path.join(class_temp, img_path[j])
                img = cv2.imread(img_path_temp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if j < 450:
                    train_list_img.append(img)
                    train_list_label.append(i)
                else:
                    test_list_img.append(img)
                    test_list_label.append(i)

        ## testing image loading
        '''test_img_path = os.path.join(test_path, 'images')
        test_class_path = os.listdir(test_img_path)

        test_header = []

        for idx, row in val_data.iterrows():
            test_header.append(row['Class'])

        for i in range(len(test_class_path)):
            img_path_temp = os.path.join(test_img_path, test_class_path[i])
            img = cv2.imread(img_path_temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_list_img.append(img)

            test_label = 0
            class_header = test_header[i]

            for label in class_path:
                if label == class_header:
                    test_list_label.append(test_label)
                    break
                else:
                    test_label += 1'''

        train_list_img, test_list_img = np.asarray(train_list_img), np.asarray(test_list_img)

        train_list_label, test_list_label = np.asarray(train_list_label), np.asarray(test_list_label)

        self.train_data, self.test_data = train_list_img, test_list_img
        self.train_targets, self.test_targets= train_list_label, test_list_label

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label,labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.test_data[np.array(self.test_targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))

        self.TestData, self.TestLabels = self.concatenate(datas, labels)
        self.TrainData, self.TrainLabels = [], []

    def getTrainData(self, classes, exemplar_set, exemplar_label_set):
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        for label in classes:
            data = self.train_data[np.array(self.train_targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.train_transform:
            img = self.train_transform(img)

        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_targets) == label]


