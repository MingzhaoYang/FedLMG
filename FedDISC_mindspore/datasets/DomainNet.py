import os
from os import path
from PIL import Image
import numpy as np
import random
from mindspore import set_seed, Tensor
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

# 固定随机种子
random.seed(1)
np.random.seed(1)
set_seed(1)

imgsize = 224


def read_domainnet_data_test(dataset_path, domain_name, split="train", shotnum=999999999):
    data_paths = []
    data_labels = []
    shot = [0 for _ in range(345)]
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            if int(label) > 29:
                continue
            if shot[int(label)] < shotnum:
                data_path = path.join(dataset_path, data_path)
                data_paths.append(data_path)
                data_labels.append(int(label))
                shot[int(label)] += 1
    return data_paths, data_labels


def read_domainnet_data_train(dataset_path, domain_name, split="train", shotnum=999999999):
    data_paths_server, data_labels_server = [], []
    data_paths_client, data_labels_client = [], []
    shot = [0 for _ in range(345)]
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            if int(label) > 29:
                continue
            if shot[int(label)] < shotnum:
                data_path = path.join(dataset_path, data_path)
                data_paths_server.append(data_path)
                data_labels_server.append(int(label))
                shot[int(label)] += 1
            else:
                data_path = path.join(dataset_path, data_path)
                data_paths_client.append(data_path)
                data_labels_client.append(int(label))
    return data_paths_server, data_labels_server, data_paths_client, data_labels_client


# 生成器函数，替代 PyTorch Dataset
def domainnet_generator(data_paths, data_labels, preprocess):
    for path, label in zip(data_paths, data_labels):
        img = Image.open(path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if preprocess is not None:
            img = preprocess(img)
        yield np.array(img), np.int32(label)


def get_domainnet_dloader(base_path, domain_name, batch_size, preprocess=None, num_workers=16, shotnum=5):
    dataset_path = path.join(base_path)
    train_data_paths_server, train_data_labels_server, train_data_paths_client, train_data_labels_client = \
        read_domainnet_data_train(dataset_path, domain_name, split="train", shotnum=shotnum)
    test_data_paths, test_data_labels = read_domainnet_data_test(dataset_path, domain_name, split="test", shotnum=99999999)

    # 构造 MindSpore Dataset
    train_dataset_server = ds.GeneratorDataset(
        source=lambda: domainnet_generator(train_data_paths_server, train_data_labels_server, preprocess),
        column_names=["image", "label"],
        shuffle=True,
        num_parallel_workers=num_workers
    )

    train_dataset_client = ds.GeneratorDataset(
        source=lambda: domainnet_generator(train_data_paths_client, train_data_labels_client, preprocess),
        column_names=["image", "label"],
        shuffle=True,
        num_parallel_workers=num_workers
    )

    test_dataset = ds.GeneratorDataset(
        source=lambda: domainnet_generator(test_data_paths, test_data_labels, preprocess),
        column_names=["image", "label"],
        shuffle=False,
        num_parallel_workers=num_workers
    )

    # 设置 batch
    train_dataset_server = train_dataset_server.batch(batch_size)
    train_dataset_client = train_dataset_client.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset_server, train_dataset_client, test_dataset
