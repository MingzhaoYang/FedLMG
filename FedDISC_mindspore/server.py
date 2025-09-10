import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import Tensor, dtype as mstype
from mindspore.nn import TrainOneStepCell, WithLossCell
from sdmodel import ClientImageEncoder, ImageGenerator
from datasets.openimage import get_openimage_classes
from utils import evaluation

# ---------------------- 数据集类 ----------------------
class ServerData(ds.GeneratorDataset):
    def __init__(self, data_paths, transform=None):
        self.data = data_paths
        self.transform = transform
        super(ServerData, self).__init__(source=self, column_names=["image", "label"])

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, np.int32(label)

    def __len__(self):
        return len(self.data)


class FeatureData(ds.GeneratorDataset):
    def __init__(self, datas, num_classes):
        self.num_classes = num_classes
        self.data = self._prepare_data(datas)
        super(FeatureData, self).__init__(source=self, column_names=["feature", "label"])

    def _prepare_data(self, datas):
        all_data = []
        for c in range(self.num_classes):
            for client_data in datas:
                if client_data[c] is None:
                    continue
                for fea in client_data[c]:
                    all_data.append((fea.asnumpy(), c))  # 转为numpy，MindSpore Dataset可读
        return all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# ---------------------- 模型类 ----------------------
class ServerTune(nn.Cell):
    def __init__(self, classes=345, noise_level=0):
        super(ServerTune, self).__init__()
        self.encoder = ClientImageEncoder()
        self.noise_level = noise_level
        self.final_proj = nn.Dense(2048, classes, weight_init='normal')

    def construct(self, x, get_fea=False):
        fea = self.encoder(x, noise_level=self.noise_level)
        if get_fea:
            return fea.reshape(fea.shape[0], -1)
        out = self.final_proj(fea.reshape(fea.shape[0], -1))
        return out


class TempModel(nn.Cell):
    def __init__(self, classes=345):
        super(TempModel, self).__init__()
        self.encoder = ClientImageEncoder()
        self.final_proj = nn.Dense(2048, classes, weight_init='normal')

    def construct(self, x, get_fea=False, input_image=True):
        if input_image:
            x = self.encoder(x)
        out = self.final_proj(x)
        return out

# ---------------------- Server ----------------------
class Server:
    def __init__(self, server_labeled_loader, transform, bs, classes, beta=0, repeat=1, steps=50, imgpath=''):
        self.client_features = None
        self.server_labeled_loader = server_labeled_loader
        self.global_features = None
        self.dataset = None
        self.transform = transform
        self.bs = bs
        self.beta = beta
        self.repeat = repeat
        self.steps = steps
        self.imgpath = imgpath
        self.num_classes = classes
        self.start_class = 0

        # 模型
        self.model = ServerTune(classes=classes)
        self.tempmodel = TempModel(classes=classes)
        
        # OpenImage 类名
        self.open_image_class_prompts, self.open_image_rough_classes = get_openimage_classes()

    # ---------------------- 聚合客户端特征 ----------------------
    def __aggregation_fea__(self):
        classes = list(set([key for fea in self.client_features for key in fea.keys()]))
        global_features = [{i: None for i in range(self.num_classes)} for _ in range(len(self.client_features))]

        for cidx, fea_dict in enumerate(self.client_features):
            for k, v in fea_dict.items():
                if v is not None:
                    global_features[cidx][k] = ms.Tensor(np.mean(v.asnumpy(), axis=0), mstype.float16)
        return global_features

    # ---------------------- 获取类别原型 ----------------------
    def get_class_proto(self):
        proto = ms.Tensor(np.zeros([self.num_classes, 2048]), mstype.float16)
        all_features, all_labels = [], []
        for batch in tqdm(self.server_labeled_loader.create_dict_iterator()):
            images, labels = batch['image'], batch['label']
            fea = self.model(images, get_fea=True)
            all_features.append(fea)
            all_labels.append(labels)
        all_features = ms.ops.Concat(0)(all_features)
        all_labels = ms.ops.Concat(0)(all_labels)
        for i in np.unique(all_labels.asnumpy()):
            proto[i - self.start_class] = ms.ops.ReduceMean()(all_features[all_labels == i], 0)
        return proto

    # ---------------------- 更新特征 ----------------------
    def update_features(self, features, do_generate=True, directtrain=False):
        self.client_features = features
        self.global_features = self.__aggregation_fea__()
        if directtrain:
            return

        if do_generate:
            self.dataset = self.Generator(self.global_features, self.client_features, self.transform, path=self.imgpath, repeat=self.repeat)
        self.dataloader = ds.GeneratorDataset(self.dataset, column_names=["image", "label"]).batch(self.bs, drop_remainder=True)

    # ---------------------- 数据生成 ----------------------
    def Generator(self, global_features, client_features, transform, path='domainnet_0', repeat=1):
        img_gen = ImageGenerator()
        os.makedirs(f'/home/share/gen_data/{path}', exist_ok=True)
        datapath = []

        classes = list(client_features[0].keys())
        for c in tqdm(classes):
            idx = 0
            for client_idx, fea_dict in enumerate(client_features):
                if fea_dict[c] is None:
                    continue
                for imgfea in fea_dict[c]:
                    imgfea = imgfea.expand_dims(0)
                    for r in range(repeat):
                        global_fea = self.global_features[idx % len(self.global_features)][c]
                        noised_imgfea = img_gen.noise_image_embeddings(imgfea, noise_level=200)
                        input_fea = ms.ops.Concat(0)((ms.ops.ZerosLike()(noised_imgfea), noised_imgfea))
                        output = img_gen(prompt=self.open_image_rough_classes[c],
                                         image_embeddings=input_fea,
                                         global_embeddings=global_fea,
                                         generator=None,
                                         num_inference_steps=self.steps)
                        if not output["nsfw_content_detected"][0]:
                            image = output["images"][0]
                            save_dir = f'/home/share/gen_data/{path}/{self.open_image_rough_classes[c]}'
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f'{c}_{r}_{idx}.jpg')
                            ms.numpy.array(image)  # 转为数组后用PIL保存
                            Image.fromarray(np.array(image)).save(save_path)
                            datapath.append((save_path, c))
                        idx += 1

        return ServerData(datapath, transform)

    # ---------------------- 训练 ----------------------
    def train(self, lr, epochs, test_data):
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = nn.SGD(self.model.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=1e-5)
        train_net = TrainOneStepCell(WithLossCell(self.model, criterion), optimizer)
        for e in tqdm(range(epochs)):
            for batch in self.dataloader.create_dict_iterator():
                images, labels = batch['image'], batch['label']
                train_net(images, labels)
            top1, top5 = evaluation(self.model, test_data)
            print(f'Epoch {e}: top1={top1}, top5={top5}')

    def directtrain(self, lr, epochs, test_data):
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        dataset = FeatureData([self.client_features], self.num_classes)
        dataloader = ds.GeneratorDataset(dataset, column_names=["feature", "label"]).batch(self.bs, drop_remainder=False)
        optimizer = nn.SGD(self.tempmodel.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=1e-5)
        train_net = TrainOneStepCell(WithLossCell(self.tempmodel, criterion), optimizer)
        for e in tqdm(range(epochs)):
            for batch in dataloader.create_dict_iterator():
                features, labels = batch['feature'], batch['label']
                train_net(features, labels)
            top1, top5 = evaluation(self.tempmodel, test_data)
            print(f'Direct train Epoch {e}: top1={top1}, top5={top5}')
