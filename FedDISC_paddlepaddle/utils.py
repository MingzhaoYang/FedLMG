import numpy as np
import paddle
import paddle.nn as nn
import collections
from tqdm import tqdm
from paddle.io import Dataset
from PIL import Image


def partition(alpha, dataset, num_clients, ptype='dirichlet'):
    if ptype == 'shard':
        dpairs, orilabels = [], []
        for did in range(len(dataset)):
            dpairs.append([did, dataset[did][-1]])
            orilabels.append(dataset[did][-1])
        orilabels = np.array(orilabels)
        num_classes = max(orilabels) + 1
        alpha = min(max(0, alpha), 1.0)
        num_shards = max(int((1 - alpha) * num_classes * 2), 1)
        client_datasize = int(len(dataset) / num_clients)
        all_idxs = [i for i in range(len(dataset))]
        z = zip([p[1] for p in dpairs], all_idxs)
        z = sorted(z)
        labels, all_idxs = zip(*z)
        shardsize = int(client_datasize / num_shards)
        idxs_shard = list(range(int(num_clients * num_shards)))
        local_datas = [[] for _ in range(num_clients)]
        for i in range(num_clients):
            rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
            idxs_shard = list(set(idxs_shard) - rand_set)
            for rand in rand_set:
                local_datas[i].extend(all_idxs[rand * shardsize:(rand + 1) * shardsize])
        traindata_cls_counts = record_net_data_stats(orilabels, local_datas)

    elif ptype == 'dirichlet':
        MIN_ALPHA = 0.01
        alpha = (-4 * np.log(alpha + 1e-8)) ** 4
        alpha = max(alpha, MIN_ALPHA)
        labels = [dataset[did][-1] for did in range(len(dataset))]
        num_classes = max(labels) + 1
        lb_counter = collections.Counter(labels)
        p = np.array([1.0 * v / len(dataset) for v in lb_counter.values()])
        lb_dict = {}
        labels = np.array(labels)
        for lb in range(len(lb_counter.keys())):
            lb_dict[lb] = np.where(labels == lb)[0]
        proportions = [np.random.dirichlet(alpha * p) for _ in range(num_clients)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(alpha * p) for _ in range(num_clients)]
        while True:
            mean_prop = np.mean(proportions, axis=0)
            error_norm = ((mean_prop - p) ** 2).sum()
            if error_norm <= 1e-2 / num_classes:
                break
            exclude_norms = []
            for cid in range(num_clients):
                mean_excid = (mean_prop * num_clients - proportions[cid]) / (num_clients - 1)
                error_excid = ((mean_excid - p) ** 2).sum()
                exclude_norms.append(error_excid)
            excid = np.argmin(exclude_norms)
            sup_prop = [np.random.dirichlet(alpha * p) for _ in range(num_clients)]
            alter_norms = []
            for cid in range(num_clients):
                if np.any(np.isnan(sup_prop[cid])):
                    continue
                mean_alter_cid = mean_prop - proportions[excid] / num_clients + sup_prop[cid] / num_clients
                error_alter = ((mean_alter_cid - p) ** 2).sum()
                alter_norms.append(error_alter)
            if len(alter_norms) > 0:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]

        local_datas = [[] for _ in range(num_clients)]
        dirichlet_dist = []
        for lb in lb_counter.keys():
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb] for pi in proportions])
            lb_proportion = lb_proportion / lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
        dirichlet_dist = np.array(dirichlet_dist).T
        for i in range(num_clients):
            np.random.shuffle(local_datas[i])
        traindata_cls_counts = record_net_data_stats(labels, local_datas)
    return local_datas, traindata_cls_counts


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in enumerate(net_dataidx_map):
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


class Truncated(Dataset):
    def __init__(self, ori_dataset, dataidxs=None, transform=None):
        super().__init__()
        self.ori_dataset = ori_dataset
        self.dataidxs = dataidxs
        self.transform = transform
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        data = self.ori_dataset.data
        target = self.ori_dataset.target
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def evaluation(model, testdata):
    model.eval()
    num_classes = 60
    top1s, topks = [], []

    if isinstance(testdata, list):
        for test_data in tqdm(testdata):
            with paddle.no_grad():
                classes_num = [0 for _ in range(num_classes)]
                classes_correct = [0 for _ in range(num_classes)]
                total, top1, topk = 0, 0, 0
                for test_imgs, test_labels in test_data:
                    test_labels = paddle.to_tensor(test_labels, dtype='int64')
                    out = model(test_imgs)
                    _, maxk = paddle.topk(out, 5, axis=-1)
                    total += test_labels.shape[0]
                    test_labels = paddle.reshape(test_labels, [-1, 1])
                    for i in range(len(test_labels)):
                        if test_labels[i] == maxk[i, 0:1]:
                            classes_correct[int(test_labels[i])] += 1
                        classes_num[int(test_labels[i])] += 1
                    top1 += (test_labels == maxk[:, 0:1]).astype('int').sum().item()
                    topk += (test_labels == maxk).astype('int').sum().item()
                classes_acc = [0 for _ in range(num_classes)]
                for i in range(num_classes):
                    if classes_num[i] != 0:
                        classes_acc[i] = classes_correct[i] / classes_num[i]
                    else:
                        classes_acc[i] = -1
                print(classes_acc)
            top1s.append(100 * top1 / total)
            topks.append(100 * topk / total)
        return top1s, topks
    else:
        with paddle.no_grad():
            total, top1, topk = 0, 0, 0
            for test_imgs, test_labels in testdata:
                test_labels = paddle.to_tensor(test_labels, dtype='int64')
                out = model(test_imgs)
                _, maxk = paddle.topk(out, 5, axis=-1)
                total += test_labels.shape[0]
                test_labels = paddle.reshape(test_labels, [-1, 1])
                top1 += (test_labels == maxk[:, 0:1]).astype('int').sum().item()
                topk += (test_labels == maxk).astype('int').sum().item()
        return 100 * top1 / total, 100 * topk / total
