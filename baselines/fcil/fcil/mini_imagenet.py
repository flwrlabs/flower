import numpy as np
from PIL import Image
import cv2
import os

class Mini_Imagenet:
    def __init__(self, root, train_transform=None, test_transform=None):
        super(Mini_Imagenet, self).__init__()
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
        class_path = os.listdir(self.root)
        for i in range(len(class_path)):
            class_temp = os.path.join(self.root, class_path[i])
            img_path = os.listdir(class_temp)
            for j in range(len(img_path)):
                img_path_temp = os.path.join(class_temp, img_path[j])
                img = cv2.imread(img_path_temp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if j < 500:
                    train_list_img.append(img)
                    train_list_label.append(i)
                else:
                    test_list_img.append(img)
                    test_list_label.append(i)

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


