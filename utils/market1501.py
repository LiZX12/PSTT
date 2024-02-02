from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import os.path as osp
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from PIL import Image
import random
from timm.data.random_erasing import RandomErasing
import math

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    # **kwargs:一些不确定的参数
    def __init__(self, root, verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        self.num_train_vids = 0

    # 判断文件夹的路径是否存在问题
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    # 获取图片的路径，标注信息（person id，camera id）和图片数量
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # 获取.jpg类型的文件
        pattern = re.compile(r'([-\d]+)_c(\d)')  #

        pid_container = set()  # 存取训练集中的id，set()有去重功能
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}  # id重排，成为映射关系{...:...,1500:750}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background # 判断pid是否在该范围内
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0，归化到[0,5]
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

# 重构Dataset
class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index] # index：0-图片数
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB') # 读取图片并转换为RGB

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


# 随机裁剪图像大小和纵横比
class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area  # crop后图像的面积范围
            aspect_ratio = random.uniform(2, 3)  # 长宽比范围

            # crop图像的长宽
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                # 随机生成crop时的中心点横纵坐标
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))  # 先crop
                assert (img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)  # 再resize到指定的尺寸

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


# 随即擦除：随机选取一个图片的矩形区域，将这个矩形区域的像素值用随机值或者平均像素值代替，产生局部遮挡的效果
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    # probability：执行随机擦除的概率，默认是0.5；
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean  # 擦除的块中填充的数值，默认是ImageNet的像素归一化均值[0.4914, 0.4822, 0.4465]
        self.sl = sl  # 最小的擦除面积，这里是相对原图的面积比例，默认是0.02
        self.sh = sh  # 最大的擦除面积，默认是0.4
        self.r1 = r1  # 最小的长宽比，实际长宽比是[0.3,1/0.3]之间的随机值

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area  # 面积范围[sl,sh]*area=[0.02*area,0.4*area]
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)  # 长宽比范围[r1,1/r1]=[0.3,3.333...]

            h = int(round(math.sqrt(target_area * aspect_ratio)))  # 擦除区域高度
            w = int(round(math.sqrt(target_area / aspect_ratio)))  # 擦除区域宽度

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)  # 随机起始点纵坐标
                y1 = random.randint(0, img.size()[2] - w)  # 随机起始点横坐标
                if img.size()[0] == 3:  # RGB图
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]  # 将擦除的区域全部赋予预设的填充数值
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:  # 灰度图
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

if __name__ == '__main__':

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    a = Market1501(r"H:\data")
    # process = Preprocessor(a.train)
    process = Preprocessor(a.query,transform=train_transformer)
    for (img, fname, pid, camid, index) in enumerate(process):
        print(img)
