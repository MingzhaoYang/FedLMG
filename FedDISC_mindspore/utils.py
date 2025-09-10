import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.vision import py_transforms as CV

# ---------------------- 数据划分 ----------------------
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
        all_idxs = np.arange(len(dataset))
        z = sorted(zip([p[1] for p in dpairs], all_idxs))
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
        alpha = max((-4 * np.log(alpha + 1e-8))**4, MIN_ALPHA)
        labels = np.array([dataset[did][-1] for did in range(len(dataset))])
        num_classes = max(labels) + 1
        lb_counter = {k: v for k, v in zip(*np.unique(labels, return_counts=True))}
        p = np.array([v / len(dataset) for v in lb_counter.values()])
        lb_dict = {lb: np.where(labels == lb)[0] for lb in lb_counter.keys()}

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
                exclude_norms.append(((mean_excid - p) ** 2).sum())
            excid = np.argmin(exclude_norms)
            sup_prop = [np.random.dirichlet(alpha * p) for _ in range(num_clients)]
            alter_norms = []
            for cid in range(num_clients):
                if np.any(np.isnan(sup_prop[cid])):
                    continue
                mean_alter_cid = mean_prop - proportions[excid] / num_clients + sup_prop[cid] / num_clients
                alter_norms.append(((mean_alter_cid - p) ** 2).sum())
            if len(alter_norms) > 0:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]

        local_datas = [[] for _ in range(num_clients)]
        for lb, lb_idxs in lb_dict.items():
            lb_proportion = np.array([pi[lb] for pi in proportions])
            lb_proportion = lb_proportion / lb_proportion.sum()
            lb_split = np.cumsum(lb_proportion * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_split)
            local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]

        for i in range(num_clients):
            np.random.shuffle(local_datas[i])
        traindata_cls_counts = record_net_data_stats(labels, local_datas)

    return local_datas, traindata_cls_counts

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in enumerate(net_dataidx_map):
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        net_cls_counts[net_i] = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return net_cls_counts

# ---------------------- 数据集 ----------------------
class Truncated(GeneratorDataset):
    def __init__(self, ori_dataset, dataidxs=None, transform=None):
        self.ori_dataset = ori_dataset
        self.dataidxs = dataidxs
        self.transform = transform
        self.data, self.target = self._build_truncated_dataset()
        super().__init__(source=self, column_names=["image", "label"])

    def _build_truncated_dataset(self):
        data = self.ori_dataset.data
        target = self.ori_dataset.target
        if self.dataidxs is not None:
            data = np.array(data)[self.dataidxs]
            target = np.array(target)[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = int(self.target[index])
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

# ---------------------- 评估 ----------------------
def evaluation(model, testdata):
    model.set_train(False)
    top1s, topks = [], []
    num_classes = 60

    if isinstance(testdata, list):
        for test_dataset in testdata:
            total = 0
            top1 = 0
            topk = 0
            classes_num = [0] * num_classes
            classes_correct = [0] * num_classes
            for imgs, labels in test_dataset.create_tuple_iterator():
                out = model(Tensor(imgs))
                topk_vals = ms.ops.TopK()(out, 5)[1].asnumpy()
                labels = labels.asnumpy().reshape(-1)
                total += len(labels)
                for i in range(len(labels)):
                    if labels[i] == topk_vals[i, 0]:
                        classes_correct[labels[i]] += 1
                    classes_num[labels[i]] += 1
                top1 += sum([labels[i] == topk_vals[i, 0] for i in range(len(labels))])
                topk += sum([labels[i] in topk_vals[i] for i in range(len(labels))])
            classes_acc = [classes_correct[i] / classes_num[i] if classes_num[i] > 0 else -1 for i in range(num_classes)]
            print(classes_acc)
            top1s.append(100 * top1 / total)
            topks.append(100 * topk / total)
        return top1s, topks

    else:
        total = 0
        top1 = 0
        topk = 0
        for imgs, labels in testdata.create_tuple_iterator():
            out = model(Tensor(imgs))
            topk_vals = ms.ops.TopK()(out, 5)[1].asnumpy()
            labels = labels.asnumpy().reshape(-1)
            total += len(labels)
            top1 += sum([labels[i] == topk_vals[i, 0] for i in range(len(labels))])
            topk += sum([labels[i] in topk_vals[i] for i in range(len(labels))])
        return 100 * top1 / total, 100 * topk / total
