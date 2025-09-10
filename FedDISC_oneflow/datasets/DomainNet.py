from os import path
import os
from PIL import Image

import oneflow as torch  # 关键修改：替换导入
import oneflow.nn as nn
import oneflow.optim as optim
import torchvision.transforms as transforms
from oneflow.utils.data import DataLoader, Dataset
import random
import numpy as np
from tqdm import tqdm

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

imgsize = 224



def read_domainnet_data_test(dataset_path, domain_name, split="train",shotnum=999999999):
    data_paths = []
    data_labels = []
    shot = [0 for _ in range(345)]
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            if int(label)>29: continue
            #if int(label)<245 or int(label)>249: continue
            if shot[int(label)]<shotnum:
                data_path = path.join(dataset_path, data_path)
                data_paths.append(data_path)
                data_labels.append(int(label))
                shot[int(label)]+=1
    return data_paths, data_labels

def read_domainnet_data_train(dataset_path, domain_name, split="train",shotnum=999999999):
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
            if int(label)>29: continue
           # if int(label)<245 or int(label)>249: continue
            if shot[int(label)]<shotnum:
                data_path = path.join(dataset_path, data_path)
                data_paths_server.append(data_path)
                data_labels_server.append(int(label))
                shot[int(label)]+=1
            else:
                data_path = path.join(dataset_path, data_path)
                data_paths_client.append(data_path)
                data_labels_client.append(int(label))
                
    return data_paths_server,data_labels_server, data_paths_client, data_labels_client


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
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data)


def get_domainnet_dloader(base_path,domain_name, batch_size, preprocess,num_workers=16,shotnum=5):
    dataset_path = path.join(base_path)
    train_data_paths_server, train_data_labels_server,train_data_paths_client,train_data_labels_client = read_domainnet_data_train(dataset_path, domain_name, split="train",shotnum=shotnum)
    test_data_paths, test_data_labels = read_domainnet_data_test(dataset_path, domain_name, split="test",shotnum=99999999)
    
    
    train_dataset_server = DomainNet(train_data_paths_server, train_data_labels_server, preprocess, domain_name)
    train_dataset_client = DomainNet(train_data_paths_client, train_data_labels_client, preprocess, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)
    return train_dataset_server,train_dataset_client,test_dataset


