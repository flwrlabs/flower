import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random
from Fed_utils import * 

def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class GLFC_model:

    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, train_set, device, encode_model):

        super(GLFC_model, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.encode_model = encode_model


        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = 0
        self.learned_numclass = 0
        self.learned_classes = []
        self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None
        self.train_dataset = train_set
        self.start = True
        self.signal = False

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1
        self.device = device
        self.last_entropy = 0

    # get incremental train data
    def beforeTrain(self, task_id_new, group):
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            self.numclass = self.task_size * (task_id_new + 1)
            if group != 0:
                if self.current_class != None:
                    self.last_class = self.current_class
                self.current_class = random.sample([x for x in range(self.numclass - self.task_size, self.numclass)], 6)
                # print(self.current_class)
            else:
                self.last_class = None

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, False)
        
    def update_new_set(self):
        self.model = model_to_device(self.model, False, self.device)
        self.model.eval()
        self.signal = False
        self.signal = self.entropy_signal(self.train_loader)

        if self.signal and (self.last_class != None):
            self.learned_numclass += len(self.last_class)
            self.learned_classes += self.last_class
        
            m = int(self.memory_size / self.learned_numclass)
            self._reduce_exemplar_sets(m)
            for i in self.last_class: 
                images = self.train_dataset.get_image_class(i)
                self._construct_exemplar_set(images, m)

        self.model.train()

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, True)


    def _get_train_and_test_dataloader(self, train_classes, mix):
        if mix:
            self.train_dataset.getTrainData(train_classes, self.exemplar_set, self.learned_classes)
        else:
            self.train_dataset.getTrainData(train_classes, [], [])

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  num_workers=8,
                                  pin_memory=True)

        return train_loader

    # train model
    def train(self, ep_g, model_old):
        self.model = model_to_device(self.model, False, self.device)
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)

        if model_old[1] != None:
            if self.signal:
                self.old_model = model_old[1]
            else:
                self.old_model = model_old[0]
        else:
            if self.signal:
                self.old_model = model_old[0]
        
        if self.old_model != None:
            print('load old model')
            self.old_model = model_to_device(self.old_model, False, self.device)
            self.old_model.eval()
        
        for epoch in range(self.epochs):
            loss_cur_sum, loss_mmd_sum = [], []
            if (epoch + ep_g * 20) % 200 == 100:
                if self.numclass==self.task_size:
                     opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, weight_decay=0.00001)
                else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate / 5
            elif (epoch + ep_g * 20) % 200 == 150:
                if self.numclass>self.task_size:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate / 25
                else:
                     opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
            elif (epoch + ep_g * 20) % 200 == 180:
                if self.numclass==self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] =self.learning_rate / 125
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.cuda(self.device), target.cuda(self.device)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()

    def entropy_signal(self, loader):
        self.model.eval()
        start_ent = True
        res = False

        for step, (indexs, imgs, labels) in enumerate(loader):
            imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = entropy(softmax_out)

            if start_ent:
                all_ent = ent.float().cpu()
                all_label = labels.long().cpu()
                start_ent = False
            else:
                all_ent = torch.cat((all_ent, ent.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.long().cpu()), 0)

        overall_avg = torch.mean(all_ent).item()
        print(overall_avg)
        if overall_avg - self.last_entropy > 1.2:
            res = True
        
        self.last_entropy = overall_avg

        self.model.train()

        return res

    def _compute_loss(self, indexs, imgs, label):
        output = self.model(imgs)

        target = get_one_hot(label, self.numclass, self.device)
        output, target = output.cuda(self.device), target.cuda(self.device)
        if self.old_model == None:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))

            return loss_cur
        else:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))

            distill_target = target.clone()
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            distill_target[..., :old_task_size] = old_target
            loss_old = F.binary_cross_entropy_with_logits(output, distill_target)

            return 0.5 * loss_cur + 0.5 * loss_old

    def efficient_old_class_weight(self, output, label):
        pred = torch.sigmoid(output)
        
        N, C = pred.size(0), pred.size(1)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = get_one_hot(label, self.numclass, self.device)
        g = torch.abs(pred.detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        if len(self.learned_classes) != 0:
            for i in self.learned_classes:
                ids = torch.where(ids != i, ids, ids.clone().fill_(-1))

            index1 = torch.eq(ids, -1).float()
            index2 = torch.ne(ids, -1).float()
            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.)
            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.)

            w = w1 + w2
        
        else:
            w = g.clone().fill_(1.)

        return w

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
     
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        self.exemplar_set.append(exemplar)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).cuda(self.device)
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            exemplar=self.exemplar_set[index]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def proto_grad_sharing(self):
        if self.signal:
            proto_grad = self.prototype_mask()
        else:
            proto_grad = None

        return proto_grad

    def prototype_mask(self):
        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        iters = 50
        criterion = nn.CrossEntropyLoss().to(self.device)
        proto = []
        proto_grad = []

        for i in self.current_class:
            images = self.train_dataset.get_image_class(i)
            class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
            dis = class_mean - feature_extractor_output
            dis = np.linalg.norm(dis, axis=1)
            pro_index = np.argmin(dis)
            proto.append(images[pro_index])

        for i in range(len(proto)):
            self.model.eval()
            data = proto[i]
            label = self.current_class[i]
            data = Image.fromarray(data)
            label_np = label
            
            data, label = tt(data), torch.Tensor([label]).long()
            data, label = data.cuda(self.device), label.cuda(self.device)
            data = data.unsqueeze(0).requires_grad_(True)
            target = get_one_hot(label, self.numclass, self.device)

            opt = optim.SGD([data, ], lr=self.learning_rate / 10, weight_decay=0.00001)
            proto_model = copy.deepcopy(self.model)
            proto_model = model_to_device(proto_model, False, self.device)

            for ep in range(iters):
                outputs = proto_model(data)
                loss_cls = F.binary_cross_entropy_with_logits(outputs, target)
                opt.zero_grad()
                loss_cls.backward()
                opt.step()

            self.encode_model = model_to_device(self.encode_model, False, self.device)
            data = data.detach().clone().to(self.device).requires_grad_(False)
            outputs = self.encode_model(data)
            loss_cls = criterion(outputs, label)
            dy_dx = torch.autograd.grad(loss_cls, self.encode_model.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            proto_grad.append(original_dy_dx)

        return proto_grad