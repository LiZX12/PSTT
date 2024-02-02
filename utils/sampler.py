import os
import pickle

from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
from itertools import chain
import scipy.io
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, Optional, Tuple
import logging

logger = logging.getLogger("pit.train")

class FsImageFolder(ImageFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            ratio=1,
            ratio_source=".",
            label_include=True
    ):
        super(FsImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

        al_len = len(self.samples)
        if ratio == 1:
            select_index = [i for i in range(al_len)]
        else:
            t = os.path.join(ratio_source, "ratio-%s-list.text" % (str(ratio),))
            if os.path.exists(t):
                with open(t, "r") as f:
                    lines = f.readlines()
                    select_index = [int(i) for i in lines]
            else:
                select_index = list(np.random.choice([i for i in range(al_len)], al_len // ratio, replace=False))
                with open(t, "w") as f:
                    f.writelines([str(l) + "\r" for l in select_index])

        select_index = sorted(select_index)
        if not label_include:
            s_ = []
            for i in range(al_len):
                if i not in select_index:
                    s_.append(i)
            select_index = s_

        self.samples = [self.samples[i] for i in select_index]
        self.targets = [self.targets[i] for i in select_index]
        logger.info("------------\n")
        logger.info("是否过滤%s \n" % ("Yes" if label_include else "No"))
        logger.info("样本倍率为%d \n" % (ratio))
        logger.info("照片总数为%d \n" % (len(self.imgs)))
        logger.info("行人总数为%d \n" % (len(self.classes)))
        logger.info("倍率后的照片总数为%d \n" % (len(self.samples)))
        logger.info("倍率后的行人总数为%d \n" % (len(set(self.targets))))
        logger.info("------------\n")
        self.imgs = self.samples


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        # al_len = len(self.data_source)
        # label_index = np.random.choice([i for i in range(al_len)],al_len//3)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class FS_RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size

        with open(r"H:\program\Spatial-Temporal-Re-identification-master\myDictionary.pkl", "rb") as tf:
            rank_viewids_list = pickle.load(tf)

        self.label_path = rank_viewids_list["label_path"]
        self.unlabel_path = rank_viewids_list["unlabel_path"]
        self.score = rank_viewids_list["score"]
        self.unlabel_path_sort = rank_viewids_list["unlabel_path_sort"]
        # self.label_pids = rank_viewids_list["label_pids"]
        # self.unlabel_pids = rank_viewids_list["unlabel_pids"]

        self.data_info = {}
        for index, (path, pid) in enumerate(self.data_source):
            self.data_info["/".join(path.split(os.sep)[-2:])] = {"index": index, "data": self.data_source[index]}

        self.index_list = []
        self.index_na_list = []
        # 选取
        # 每一个 unlabel 选两个label
        for i in range(len(self.unlabel_path_sort)):
            po = self.unlabel_path[i]
            na = self.unlabel_path_sort[i]
            po = "/".join(po.split(os.sep)[-2:])
            na = ["/".join(a.split(os.sep)[-2:]) for a in na]
            self.index_list.append(self.data_info[po]["index"])
            self.index_na_list.append([self.data_info[a]["index"] for a in na])

        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = len(self.index_list) * 3 - len(self.index_list) * 3 % self.batch_size

    def __iter__(self):
        print("sample sample sample sample sample")
        index_list = copy.deepcopy(self.index_list)
        final_idxs = []
        iiii = np.array(range(len(index_list)))
        for i in range(len(index_list)):
            index = np.random.choice(iiii, size=1, replace=True)[0]
            final_idxs.append(index_list[index])
            final_idxs.append(self.index_na_list[index][np.random.randint(0, 20)])
            final_idxs.append(self.index_na_list[index][np.random.randint(20, 40)])
        return iter(final_idxs)

    def __len__(self):
        return self.length


class PRE_FS_RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, name="", unlabel_size=20, hard_sample_size = 5,hard_sample_ratio=2,last_append_sam=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.unlabel_size = unlabel_size
        self.hard_sample_size = hard_sample_size
        self.hard_sample_ratio = hard_sample_ratio
        self.last_append_sam = last_append_sam

        # info_mat = scipy.io.loadmat(
        #     # os.path.join('./model', name, 'zzzz.mat'))
        #     r"H:\program\Spatial-Temporal-Re-identification-master\model\fs_ResNet50_pcb_market_e\zzzz.mat")
        #

        info_mat = scipy.io.loadmat(
            os.path.join('model', name, 'zzzz.mat'))
        # r"H:\program\Spatial-Temporal-Re-identification-master\model\fs\zzzz.mat")
        score = info_mat["score"]

        self.label_paths = np.array([os.path.sep.join(a.strip().split(os.sep)[-2:]) for a in info_mat['gallery_paths']])
        self.unlabel_paths = np.array([os.path.sep.join(a.strip().split(os.sep)[-2:]) for a in info_mat['query_paths']])

        self.index_sort = np.argsort(-score)  # 8374 4562
        self.score = score  # 8374 4562

        self.score_path_info = {}

        self.data_info = {}
        for index, (path, pid) in enumerate(self.data_source):
            self.data_info[os.path.sep.join(path.split(os.path.sep)[-2:])] = {"index": index,
                                                                              "data": self.data_source[index]}

        self.index_list = []
        self.index_na_list = []

        self.index_na_list_set = [self.data_info[a]["index"] for a in self.label_paths]
        # 选取
        # 每一个 unlabel 选两个label
        for i in range(len(self.unlabel_paths)):
            po = self.unlabel_paths[i]
            na = self.label_paths[self.index_sort[i]]

            self.index_list.append(self.data_info[po]["index"])  # 保存所有无标签
            self.index_na_list.append([self.data_info[a]["index"] for a in na])  # 保存所有标签
            # for a in na:
            #     self.index_na_list_set.add(self.data_info[a]["index"])

        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = len(self.index_list) * self.unlabel_size - len(
            self.index_list) * self.unlabel_size % self.batch_size

        self.class_to_index = {}
        self.index_to_class = {}
        for key in self.index_dic.keys(): # 所有的
            for k in self.index_dic[key]:
                if k in self.index_na_list_set:
                    if key not in self.class_to_index:
                        self.class_to_index[key] = []
                    self.class_to_index[key].append(k)
                    self.index_to_class[k] = key

        logger.info("select from %s" % (str([self.select_function(self.hard_sample_size, self.hard_sample_ratio, i) for i in range(1,6)]),))

    def select_function(self,a,b,index):
        if index == 0:
            return 0
        else:
            # return int(a + a/2*(index-1)**1.4 * b)
            return int(a + a*(index-1)**1.2 * b)
        # return a+(1+index)*index//2*b

    def __iter__(self):
        print("sample sample sample sample sample")
        index_list = copy.deepcopy(self.index_list)
        final_idxs = []
        iiii = [i for i in range(len(index_list))]
        slit_list = [None,]
        # if self.last_append_sam:
        if False:
            for i in range(0, self.unlabel_size - 2):
                slit_list.append([self.select_function(self.hard_sample_size, self.hard_sample_ratio, i),
                                  self.select_function(self.hard_sample_size, self.hard_sample_ratio, i + 1)])
            for i in range(len(index_list)):
                index = np.random.choice(iiii, size=1, replace=True)[0]
                iiii.remove(index)
                final_idxs.append(index_list[index])
                cu_index = None
                for j in range(1, self.unlabel_size-1):
                    cu_index = self.index_na_list[index][np.random.randint(slit_list[j][0], slit_list[j][1])]
                    final_idxs.append(cu_index)
                final_idxs.append(self.find_another_index(cu_index))
                if len(final_idxs) >= self.length:
                    break
        else:
           for i in range(0,self.unlabel_size-1):
               slit_list.append([self.select_function(self.hard_sample_size,self.hard_sample_ratio,i),self.select_function(self.hard_sample_size,self.hard_sample_ratio,i+1)])
           for i in range(len(index_list)):
               index = np.random.choice(iiii, size=1, replace=True)[0]
               iiii.remove(index)
               final_idxs.append(index_list[index])
               for j in range(1, self.unlabel_size):
                   cu_index = self.index_na_list[index][np.random.randint(slit_list[j][0], slit_list[j][1])]
                   final_idxs.append(cu_index)
               if len(final_idxs) >= self.length:
                   break

        # slit_list = [None, [0, 5], [5, 20], [20, 45], [1000, 2000], [200, 400]]
        # slit_list = [None, [0, 5], [5, 25], [25, 50], [1000, 2000], [200, 400]]


        return iter(final_idxs)
    def find_another_index(self,cu_index):
        n_index = self.class_to_index[self.index_to_class[cu_index]]
        n_index = copy.copy(n_index)
        if len(n_index) == 1:
            return cu_index
        else:
            n_index.remove(cu_index)
            return random.choice(n_index)
    def __len__(self):
        return self.length

    def get_score(self, path1_list, path2_list):  # 无标
        unlabel_paths = list(copy.deepcopy(self.unlabel_paths))
        label_paths = list(copy.deepcopy(self.label_paths))
        i_index = np.array([unlabel_paths.index(os.path.sep.join(p.split(os.path.sep)[-2:])) for p in path1_list])
        j_index = np.array([label_paths.index(os.path.sep.join(p.split(os.path.sep)[-2:])) for p in path2_list])

        return np.array([self.score[i_index[i]][j_index[i]] for i in range(len(i_index))])

    def get_km_label(self, label_path_list):
        label_paths = list(copy.deepcopy(self.label_paths))
        i_index = np.array([label_paths.index(os.path.sep.join(p.split(os.path.sep)[-2:])) for p in label_path_list])
        return self.km_labels[i_index]

class RATIO_PRE_FS_RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, name="", unlabel_size=20):
        self.data_source = data_source
        self.batch_size = batch_size
        self.unlabel_size = unlabel_size

        # info_mat = scipy.io.loadmat(
        #     # os.path.join('./model', name, 'zzzz.mat'))
        #     r"H:\program\Spatial-Temporal-Re-identification-master\model\fs_ResNet50_pcb_market_e\zzzz.mat")
        #

        info_mat = scipy.io.loadmat(
            os.path.join('model', name, 'zzzz.mat'))
        # r"H:\program\Spatial-Temporal-Re-identification-master\model\fs\zzzz.mat")
        score = info_mat["score"]

        self.label_paths = np.array([os.path.sep.join(a.split(os.sep)[-2:]) for a in info_mat['gallery_paths']])
        self.unlabel_paths = np.array([os.path.sep.join(a.split(os.sep)[-2:]) for a in info_mat['query_paths']])

        self.index_sort = np.argsort(-score)  # 8374 4562
        self.score = score  # 8374 4562

        self.score_path_info = {}

        self.data_info = {}
        for index, (path, pid) in enumerate(self.data_source):
            self.data_info[os.path.sep.join(path.split(os.path.sep)[-2:])] = {"index": index,
                                                                              "data": self.data_source[index]}

        self.index_list = []
        self.index_na_list = []
        # 选取
        # 每一个 unlabel 选两个label
        for i in range(len(self.unlabel_paths)):
            po = self.unlabel_paths[i]
            na = self.label_paths[self.index_sort[i]]

            self.index_list.append(self.data_info[po]["index"])  # 保存所有无标签
            self.index_na_list.append([self.data_info[a]["index"] for a in na])  # 保存所有标签

        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = len(self.index_list) * self.unlabel_size - len(
            self.index_list) * self.unlabel_size % self.batch_size
        #
        # km_info_mat = scipy.io.loadmat(
        #     # os.path.join('./model', name, 'zzzz.mat'))
        #     r"H:\program\Spatial-Temporal-Re-identification-master\model\fs_ResNet50_pcb_market_e\zzzz_kms.mat")
        # km_score = km_info_mat["score"]
        # import kmedoids
        # self.km_labels = kmedoids.fasterpam(-km_score, 751,random_state=1551).labels

    def __iter__(self):
        print("sample sample sample sample sample")
        index_list = copy.deepcopy(self.index_list)
        final_idxs = []
        iiii = [i for i in range(len(index_list))]
        if self.unlabel_size == 2:
            slit_list = [None, [0, 1], [1, 50], [1000, 2000], [200, 400]]
        elif self.unlabel_size == 3:
            slit_list = [None, [0, 1], [1, 50], [20, 50], [1000, 2000], [200, 400]]
        elif self.unlabel_size == 4:
            # slit_list = [None, [0, 1], [1, 15], [15, 50], [1000, 2000], [200, 400]]
            slit_list = [None, [0, 5], [5, 20], [20, 50], [1000, 2000], [200, 400]]
        elif self.unlabel_size == 5:
            slit_list = [None, [0, 5], [5, 15], [15, 30], [30, 50], [1000, 2000], [200, 400]]
        for i in range(len(index_list)):
            index = np.random.choice(iiii, size=1, replace=True)[0]
            iiii.remove(index)
            final_idxs.append(index_list[index])
            # iiii = [i for i in range(l
            # en(100))]
            # final_idxs.append(self.index_na_list[index][0])
            for j in range(1, self.unlabel_size):
                final_idxs.append(self.index_na_list[index][np.random.randint(slit_list[j][0], slit_list[j][1])])
            if len(final_idxs) >= self.length:
                break
        return iter(final_idxs)
        # print("sample sample sample sample sample")
        # index_list = copy.deepcopy(self.index_list)
        # final_idxs = []
        # iiii = [i for i in range(len(index_list))]
        # slit_list = [None,[1, 3], [3, 40],[20,50],[200,400]]
        # for i in range(0,len(index_list),self.batch_size//self.unlabel_size):
        #     # 选取n个无标签
        #     _list_index = []
        #     for k in range(self.batch_size//self.unlabel_size):
        #         index = np.random.choice(iiii, size=1, replace=True)[0]
        #         iiii.remove(index)
        #         _list_index.append(index)
        #
        #     for k in range(self.batch_size//self.unlabel_size):
        #         index = _list_index[k]
        #         final_idxs.append(index_list[index])
        #         # iiii = [i for i in range(len(100))]
        #         final_idxs.append(self.index_na_list[index][0])
        #
        #         # 剩下的 从概率里面找
        #         for j in range(1, self.unlabel_size):
        #             pppppp = []
        #             for g in range(self.batch_size // self.unlabel_size):
        #                 pppppp += self.index_na_list[_list_index[g]][slit_list[j][0]:slit_list[j][1]]
        #
        #             uuuuu = []
        #             for p in pppppp:
        #                 if p in self.index_na_list[index][slit_list[j][0]: slit_list[j][1]]:
        #                     uuuuu.append(p)
        #
        #             final_idxs.append(np.random.choice(uuuuu, size=1, replace=True)[0])
        #     if len(final_idxs) + self.batch_size >= self.length:
        #         break
        # return iter(final_idxs)

    def __len__(self):
        return self.length

    def get_score(self, path1_list, path2_list):  # 无标
        unlabel_paths = list(copy.deepcopy(self.unlabel_paths))
        label_paths = list(copy.deepcopy(self.label_paths))
        i_index = np.array([unlabel_paths.index(os.path.sep.join(p.split(os.path.sep)[-2:])) for p in path1_list])
        j_index = np.array([label_paths.index(os.path.sep.join(p.split(os.path.sep)[-2:])) for p in path2_list])

        return np.array([self.score[i_index[i]][j_index[i]] for i in range(len(i_index))])

    def get_km_label(self, label_path_list):
        label_paths = list(copy.deepcopy(self.label_paths))
        i_index = np.array([label_paths.index(os.path.sep.join(p.split(os.path.sep)[-2:])) for p in label_path_list])
        return self.km_labels[i_index]


def compute_pids_and_pids_dict(data_source):
    index_dic = defaultdict(list)
    for index, (_, pid, _, _) in enumerate(data_source):
        index_dic[pid].append(index)
    pids = list(index_dic.keys())
    return pids, index_dic


class ReIDBatchSampler(Sampler):

    def __init__(self, data_source, p: int, k: int, cfg=None):

        self._p = p
        self._k = k
        self.cfg = cfg

        pids, index_dic = compute_pids_and_pids_dict(data_source)

        self._unique_labels = np.array(pids)
        self._label_to_items = index_dic.copy()

        self._num_iterations = len(self._unique_labels) // self._p

        assert cfg is not None

        self._num_iterations = self._num_iterations * cfg.MODEL.ACCUMULATION_STEPS

    def __iter__(self):
        for i in range(self.cfg.MODEL.ACCUMULATION_STEPS):
            def sample(set, n):
                if len(set) < n:
                    return np.random.choice(set, n, replace=True)
                return np.random.choice(set, n, replace=False)

            np.random.shuffle(self._unique_labels)

            for k, v in self._label_to_items.items():
                random.shuffle(self._label_to_items[k])

            curr_p = 0
            for idx in range(self._num_iterations // self.cfg.MODEL.ACCUMULATION_STEPS):
                p_labels = self._unique_labels[curr_p: curr_p + self._p]
                curr_p += self._p
                batch = [sample(self._label_to_items[l], self._k) for l in p_labels]
                batch = list(chain(*batch))
                yield batch

    def __len__(self):
        return self._num_iterations


class PK_ReIDBatchSampler(Sampler):

    def __init__(self, data_source, p: int, k: int, cfg=None):

        self._p = p
        self._k = k
        self.cfg = cfg

        # pids, index_dic = compute_pids_and_pids_dict(data_source)

        ######
        index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (img_path, pid, camid) in enumerate(data_source):
            index_dic[pid].append(index)
        pids = list(index_dic.keys())
        #######

        self._unique_labels = np.array(pids)
        self._label_to_items = index_dic.copy()
        # 总label
        self._num_iterations = len(self._unique_labels) // self._p

    def __iter__(self):

        def sample(set, n):
            if len(set) < n:
                return np.random.choice(set, n, replace=True)
            return np.random.choice(set, n, replace=False)

        # 打乱所有顺序
        np.random.shuffle(self._unique_labels)

        for k, v in self._label_to_items.items():
            random.shuffle(self._label_to_items[k])

        curr_p = 0
        for idx in range(self._num_iterations):
            p_labels = self._unique_labels[curr_p: curr_p + self._p]  # 选择P个label
            curr_p += self._p
            batch = [sample(self._label_to_items[l], self._k) for l in p_labels]  # 再从label中选取帧
            batch = list(chain(*batch))
            yield batch

    def __len__(self):
        return self._num_iterations
