import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
import paddle.nn.functional as F
import numpy as np
import random
import argparse
from datasets.DomainNet import get_domainnet_dloader
import os
import logging
import copy
from collections import OrderedDict
from utils import partition, Truncated, evaluation
from client import Client
from server import Server, ServerData_read
from sdmodel import ClientImageEncoder
from tqdm import tqdm

# logging.basicConfig()

# os.environ['CUDA_VISIBLE_DEVICES'] ='2'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default="/home/share/DomainNet")
    parser.add_argument('--alpha', default=1, type=float, help='degree of non-iid, only used for tinyimagenet')
    parser.add_argument('--beta', default=0, type=float, help='degree of noise')
    parser.add_argument('--data', default='openimage', help='tinyimagenet or domainnet or openimage or nicopp or nicou')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--serverbs', default=256, type=int)
    parser.add_argument('--serverepoch', default=10, type=int)
    parser.add_argument('--clientepoch', default=10, type=int)
    parser.add_argument('--learningrate', default=0.01, type=float)
    parser.add_argument('--fewnum', default=5, type=int, help='how many imgs in each class of the client')
    parser.add_argument('--num_clients', default=5, type=int, help='number of clinets, only used for tinyimagenet')
    parser.add_argument('--split-type', default='shard', help='dirichlet or shard')
    parser.add_argument('--repeat', default=10, type=int, help='how many imgs to be generated on the server')
    parser.add_argument('--inference-steps', default=20, type=int)
    parser.add_argument('--path-genimg', default='oi_test', help='where to save the generated imgs')
    return parser

drop_last = False

########################################################################################################################
parser = get_parser()
args = parser.parse_args()
seed = args.seed

# 设置随机种子
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

#======================= prepare dataset AND clients AND server==========================================
if args.data == 'domainnet':
    num_classes = 345
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation='bicubic'
        ),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch']
    clients, test_data = [], []
    server_labeled = {ddd: None for ddd in domains}
    for domain in domains:
        train_dataset_server, train_dataset_client, testdataset = get_domainnet_dloader(
            args.base_path, domain, args.batch_size, transform, shotnum=args.fewnum
        )
        print(len(train_dataset_server), len(train_dataset_client), len(testdataset))
        server_labeled[domain] = train_dataset_server

        test_data.append(DataLoader(testdataset, batch_size=256, num_workers=8, shuffle=False))
        trainloader = DataLoader(train_dataset_client, batch_size=args.batch_size, num_workers=8, shuffle=True)
        clients.append(Client(trainloader, num_classes, beta=args.beta))

    train_dataset_server, train_dataset_client, testdataset = get_domainnet_dloader(
        args.base_path, 'real', args.batch_size, transform, shotnum=args.fewnum
    )
    server_labeled['real'] = train_dataset_server
    print(len(train_dataset_server), len(train_dataset_client), len(testdataset))
    server_labeled_loader = DataLoader(server_labeled['real'], batch_size=args.batch_size, num_workers=8, shuffle=True)

server = Server(server_labeled_loader, transform, args.serverbs, num_classes,
                beta=0, repeat=args.repeat, steps=args.inference_steps, imgpath=args.path_genimg)

print('getting protos')
proto = server.get_class_proto()
print('getting features')
feas = []

for i, client in enumerate(tqdm(clients)):
    client.get_ori_features()
    fea = client.get_features(proto)
    feas.append(fea)

print('generating images')
server.update_features(feas, do_generate=True, directtrain=False)

print('server training')
server.train(lr=args.learningrate, epochs=args.serverepoch, test_data=test_data)

print('server testing')
print(domains)
top1, topk = evaluation(server.model, test_data)
print(f'final server model: top1 {top1}, top5 {topk}')

server.directtrain(lr=args.learningrate, epochs=args.serverepoch, test_data=test_data)
print('direct training')
print(domains)
top1, topk = evaluation(server.tempmodel, test_data)
print(f'final direct training model: top1 {top1}, top5 {topk}')
