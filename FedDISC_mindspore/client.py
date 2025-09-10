import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from mindspore import nn, Tensor, ops, context
from mindspore import dtype as mstype
from mindspore import value_and_grad

from sdmodel import ClientImageEncoder  # 保留用户已有的 encoder

context.set_context(mode=context.PYNATIVE_MODE)  # 方便调试

class ClientTune(nn.Cell):
    def __init__(self, classes=345, noise_level=0):
        super(ClientTune, self).__init__()
        self.encoder = ClientImageEncoder()
        self.noise_level = noise_level
        self.final_proj = nn.Dense(2048, classes).to_float(mstype.float16)

    def construct(self, x, get_fea=False):
        fea = self.encoder(x, self.noise_level)  # 默认 encoder 内部处理 float
        if get_fea:
            return fea.view(fea.shape[0], -1)
        out = self.final_proj(fea.view(fea.shape[0], -1))
        return out


class Client:
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
        for i, (image, label) in enumerate(tqdm(self.dataloader.create_dict_iterator())):
            feature = self.model(image, True)
            ori_features.append(feature)
        ori_features = ops.concat(ori_features, axis=0)
        ori_features = ops.cast(ori_features, mstype.float16)
        return ori_features

    def post_process(self, beta, proto):
        dis = -2 * (self.ori_features @ ops.transpose(proto, (1, 0))) + \
              ops.reduce_sum(self.ori_features**2, 1).view(-1, 1) + \
              ops.ones((self.ori_features.shape[0], proto.shape[0]), mstype.float16) * \
              ops.reduce_sum(proto**2, 1).view(1, -1)

        pseudo_label = ops.argmin(dis, axis=1)

        new_features = {}
        for c in range(self.classes):
            mask = ops.equal(pseudo_label, c)
            idx = ops.nonzero(mask).view(-1)
            tempfeature = ops.gather(self.ori_features, idx, 0) if idx.shape[0] > 0 else None

            if tempfeature is None or tempfeature.shape[0] == 0:
                new_features[c] = None
            elif tempfeature.shape[0] > self.K:
                km = KMeans(n_clusters=self.K, max_iter=100).fit(tempfeature.asnumpy())
                centers = Tensor(km.cluster_centers_, dtype=mstype.float16)
                if beta > 0:
                    noise = ops.standard_normal(tempfeature.shape).astype(mstype.float16)
                    new_features[c] = centers + beta * noise
                else:
                    new_features[c] = centers
            else:
                noise = ops.standard_normal(tempfeature.shape).astype(mstype.float16)
                new_features[c] = tempfeature + beta * noise

        return new_features

    def get_features(self, proto):
        return self.post_process(self.beta, proto)

    def train(self, lr, epochs):
        self.model.set_train()
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = nn.SGD(self.model.final_proj.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=1e-5)

        def forward_fn(inputs, labels):
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            return loss, outputs

        grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        for _ in tqdm(range(epochs)):
            for batch in self.dataloader.create_dict_iterator():
                images, labels = batch["image"], batch["label"]
                (loss, _), grads = grad_fn(images, labels)
                optimizer(grads)
