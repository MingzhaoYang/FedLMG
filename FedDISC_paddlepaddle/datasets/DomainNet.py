from os import path
import os
from PIL import Image
import paddle
import paddle.vision.transforms as transforms
from paddle.io import DataLoader, Dataset
import random
import numpy as np
from tqdm import tqdm

# 设置随机种子
random.seed(1)
np.random.seed(1)
paddle.seed(1)

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
    data_paths_server = []
    data_labels_server = []
    data_paths_client = []
    data_labels_client = []
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


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data = data_paths
        self.target = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.target[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data)


def get_domainnet_dloader(base_path, domain_name, batch_size, preprocess, num_workers=16, shotnum=5):
    dataset_path = path.join(base_path)
    train_data_paths_server, train_data_labels_server, train_data_paths_client, train_data_labels_client = read_domainnet_data_train(
        dataset_path, domain_name, split="train", shotnum=shotnum)
    test_data_paths, test_data_labels = read_domainnet_data_test(dataset_path, domain_name, split="test", shotnum=99999999)

    train_dataset_server = DomainNet(train_data_paths_server, train_data_labels_server, preprocess, domain_name)
    train_dataset_client = DomainNet(train_data_paths_client, train_data_labels_client, preprocess, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)

    # Paddle DataLoader
    train_loader_server = DataLoader(train_dataset_server, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    train_loader_client = DataLoader(train_dataset_client, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader_server, train_loader_client, test_loader
