import numpy as np
import random
import argparse
from tqdm import tqdm
import mindspore
from mindspore import set_seed, context
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

from datasets.DomainNet import get_domainnet_dloader
from client import Client   # 已迁移好的 Client
from server import Server, ServerData_read  # 需要迁移
from utils import partition, Truncated, evaluation  # 需要迁移
from sdmodel import ClientImageEncoder

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default="/home/share/DomainNet")
    parser.add_argument('--alpha', default=1,type=float)
    parser.add_argument('--beta', default=0,type=float)
    parser.add_argument('--data', default='domainnet')
    parser.add_argument('--seed', default=0,type=int)
    parser.add_argument('--batch_size', default=256,type=int)
    parser.add_argument('--serverbs', default=256,type=int)
    parser.add_argument('--serverepoch', default=10,type=int)
    parser.add_argument('--clientepoch', default=10,type=int)
    parser.add_argument('--learningrate', default=0.01,type=float)
    parser.add_argument('--fewnum', default=5,type=int)
    parser.add_argument('--num_clients', default=5,type=int)
    parser.add_argument('--split-type', default='shard')
    parser.add_argument('--repeat', default=10,type=int)
    parser.add_argument('--inference-steps', default=20,type=int)
    parser.add_argument('--path-genimg', default='oi_test')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # 固定随机种子
    set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.data == 'domainnet':
        num_classes = 345
        transform = [
            vision.Resize((224, 224), interpolation=vision.Inter.BICUBIC),
            vision.ToTensor(),
            vision.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ]

        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch']
        clients, test_data = [], []
        server_labeled = {ddd: None for ddd in domains}

        for domain in domains:
            train_dataset_server, train_dataset_client, testdataset = \
                get_domainnet_dloader(args.base_path, domain, args.batch_size, transform, shotnum=args.fewnum)

            print(train_dataset_server.get_dataset_size(),
                  train_dataset_client.get_dataset_size(),
                  testdataset.get_dataset_size())

            server_labeled[domain] = train_dataset_server
            test_data.append(testdataset.batch(256))

            clients.append(Client(train_dataset_client.batch(args.batch_size),
                                  num_classes, beta=args.beta))

        # real domain 用作 server
        train_dataset_server, train_dataset_client, testdataset = \
            get_domainnet_dloader(args.base_path, 'real', args.batch_size, transform, shotnum=args.fewnum)
        server_labeled['real'] = train_dataset_server
        print(train_dataset_server.get_dataset_size(),
              train_dataset_client.get_dataset_size(),
              testdataset.get_dataset_size())

        server_labeled_loader = train_dataset_server.batch(args.batch_size)

    # 初始化 server
    server = Server(server_labeled_loader, transform, args.serverbs,
                    num_classes, beta=0, repeat=args.repeat,
                    steps=args.inference_steps, imgpath=args.path_genimg)

    print('getting protos')
    proto = server.get_class_proto()

    print('getting features')
    feas = []
    for client in tqdm(clients):
        client.get_ori_features()
        fea = client.get_features(proto)
        feas.append(fea)

    print('generating images')
    server.update_features(feas, do_generate=True, directtrain=False)

    print('server training')
    server.train(lr=args.learningrate, epochs=args.serverepoch, test_data=test_data)

    print('server testing')
    top1, topk = evaluation(server.model, test_data)
    print(f'final server model: top1 {top1}, top5 {topk}')

    server.directtrain(lr=args.learningrate, epochs=args.serverepoch, test_data=test_data)
    print('direct training')
    top1, topk = evaluation(server.tempmodel, test_data)
    print(f'final direct training model: top1 {top1}, top5 {topk}')
