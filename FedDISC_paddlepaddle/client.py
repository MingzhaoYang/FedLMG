import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sdmodel import ClientImageEncoder
from tqdm import tqdm
from sklearn.cluster import KMeans


class Client:  # as a user
    def __init__(self, dataloader, classes, noise_level=500, beta=0):
        self.dataloader = dataloader
        self.model = ClientTune(classes, noise_level)
        self.ori_features = None
        self.beta = beta
        self.K = 5
        self.classes = classes

    def get_ori_features(self):
        self.ori_features = self.__get_all_fea__()

    def __get_all_fea__(self):
        ori_features = []
        for i, (image, label) in enumerate(tqdm(self.dataloader)):
            image = paddle.to_tensor(image)
            feature = self.model(image, get_fea=True)
            ori_features.append(feature)
            del image
        ori_features = paddle.concat(ori_features, axis=0)
        ori_features = ori_features.astype("float16")
        return ori_features

    def post_precess(self, beta, proto):
        # 计算距离矩阵
        dis = -2 * paddle.matmul(self.ori_features, proto.T) + \
              paddle.sum(self.ori_features ** 2, axis=1).unsqueeze(1) + \
              paddle.ones([self.ori_features.shape[0], proto.shape[0]], dtype="float16") * paddle.sum(proto ** 2, axis=1).unsqueeze(0)

        pseudo_label = paddle.argmin(dis, axis=1)

        dtype = self.ori_features.dtype
        new_features = {}
        for c in range(self.classes):
            idx = (pseudo_label == c).nonzero(as_tuple=False).squeeze()
            if idx.shape[0] == 0:
                new_features[c] = None
                continue

            tempfeature = paddle.index_select(self.ori_features, idx, axis=0)

            if tempfeature.shape[0] > self.K:
                km = KMeans(n_clusters=self.K, max_iter=100).fit(tempfeature.numpy())
                centers = paddle.to_tensor(km.cluster_centers_, dtype=dtype)
                if beta > 0:
                    noise = beta * paddle.randn(tempfeature.shape, dtype=dtype)
                    new_features[c] = centers + noise
                else:
                    new_features[c] = centers
            else:
                noise = beta * paddle.randn(tempfeature.shape, dtype=dtype)
                new_features[c] = tempfeature + noise

        return new_features

    def get_features(self, proto):
        return self.post_precess(self.beta, proto)

    def train(self, lr, epochs):
        self.model.final_proj.train()
        task_criterion = nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.SGD(
            parameters=self.model.final_proj.parameters(),
            learning_rate=lr,
            momentum=0.9,
            weight_decay=1e-5
        )
        for _ in tqdm(range(epochs)):
            for i, (image, label) in enumerate(self.dataloader):
                image = paddle.to_tensor(image)
                label = paddle.to_tensor(label)
                output = self.model(image)
                loss = task_criterion(output, label)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()


class ClientTune(nn.Layer):
    def __init__(self, classes=345, noise_level=0):
        super(ClientTune, self).__init__()
        self.encoder = ClientImageEncoder()
        self.noise_level = noise_level
        self.final_proj = nn.Sequential(
            nn.Linear(2048, classes, dtype="float16")
        )

    def forward(self, x, get_fea=False):
        with paddle.no_grad():
            fea = self.encoder(x, self.noise_level)
        if get_fea:
            return fea.reshape([fea.shape[0], -1])
        out = self.final_proj(fea.reshape([fea.shape[0], -1]))
        return out
