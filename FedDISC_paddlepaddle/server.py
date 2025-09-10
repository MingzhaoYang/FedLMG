import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image

from sdmodel import ImageGenerator, ClientImageEncoder
from datasets.openimage import get_openimage_classes
from utils import partition, Truncated, evaluation


class Server:
    def __init__(self, server_labeled_loader, transform, bs, classes, beta=0, repeat=1, steps=50, imgpath=''):
        self.client_features = None
        self.server_labeled_loader = server_labeled_loader
        self.agg_features = None
        self.dataset = None
        self.args = {'trans': transform, 'bs': bs, 'beta': beta}
        self.dataloader = None

        self.model = ServerTune(classes=classes)
        self.tempmodel = TempModel(classes=classes)
        self.new_features = None
        self.repeat = repeat
        self.steps = steps
        self.num_classes = classes
        self.imgpath = imgpath
        self.start_class = 0

        # DomainNet 类别
        path = r"/home/share/DomainNet/clipart"
        f = os.listdir(path)
        f = [x.lower() for x in f]
        self.class_prompts = sorted(f)

        # NICO++
        nicopp_path = "/home/share/NICOpp/NICO_DG/autumn"
        f = os.listdir(nicopp_path)
        f = [x.lower() for x in f]
        self.nicopp_class_prompts = sorted(f)

        self.open_image_class_prompts, self.open_image_rough_classes = get_openimage_classes()

    def get_class_proto(self):
        proto = paddle.zeros([self.num_classes, 2048], dtype='float16')
        ori_features, labels = [], []

        for i, (image, label) in enumerate(tqdm(self.server_labeled_loader)):
            feature = self.model(image, get_fea=True)
            ori_features.append(feature)
            labels.append(label)

        ori_features = paddle.concat(ori_features, axis=0)
        labels = paddle.concat(labels, axis=0)
        allclass = list(set(labels.numpy().tolist()))

        self.ori_features = ori_features
        feadict = {}
        for i in allclass:
            idx = paddle.nonzero(labels == i, as_tuple=True)[0]
            proto[i - self.start_class] = paddle.mean(ori_features[idx], axis=0)
        return proto


class ServerTune(nn.Layer):
    def __init__(self, classes=345, noise_level=0):
        super(ServerTune, self).__init__()
        self.encoder = ClientImageEncoder()
        self.noise_level = noise_level
        self.final_proj = nn.Linear(2048, classes)

    def forward(self, x, get_fea=False, return_1024=False):
        with paddle.no_grad():
            fea = self.encoder(x, noise_level=self.noise_level)
        if get_fea:
            return paddle.reshape(fea, [fea.shape[0], -1])
        out = self.final_proj(paddle.reshape(fea, [fea.shape[0], -1]))
        return out


class TempModel(nn.Layer):
    def __init__(self, classes=345):
        super(TempModel, self).__init__()
        self.encoder = ClientImageEncoder()
        self.final_proj = nn.Linear(2048, classes)

    def forward(self, x, get_fea=False, input_image=True):
        if input_image:
            with paddle.no_grad():
                x = self.encoder(x)
        out = self.final_proj(x)
        return out
