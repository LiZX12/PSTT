# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import os
import scipy.io
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

from logger_test import setup_logger
from model import ft_net, ft_net_dense, ft_net_dense_PCB, ft_net_pcb, ft_net_dense_pcb
from models.backbones.vit_pytorch import vit
import math

from utils.sampler import FsImageFolder


def gaussian_func2(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)


def gauss_smooth2(arr, o):
    hist_num = len(arr)
    vect = np.zeros((hist_num, 1))
    for i in range(hist_num):
        vect[i, 0] = i
    # gaussian_vect= gaussian_func2(vect,0,1)
    # o=50
    approximate_delta = 3 * o  # when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect = gaussian_func2(vect, 0, o)
    matrix = np.zeros((hist_num, hist_num))
    for i in range(hist_num):
        k = 0
        for j in range(i, hist_num):
            if k > approximate_delta:
                continue
            matrix[i][j] = gaussian_vect[j - i]
            k = k + 1
    matrix = matrix + matrix.transpose()
    for i in range(hist_num):
        matrix[i][i] = matrix[i][i] / 2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]
    xxx = np.dot(matrix, arr)
    return xxx

def gaussian_func(x, u, o=0.1):
    if (o == 0):
        print("In gaussian, o shouldn't equel to zero")
        return 0
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(math.pow(x - u, 2)) / (2 * math.pow(o, 2))
    return temp1 * math.exp(temp2)

def predict_id(query_feature, gallery_feature, gallery_targets):
    query_feature = query_feature.numpy()
    gallery_feature = gallery_feature.numpy()
    # gallery_targets = gallery_targets.numpy()

    query_feature = query_feature.transpose() / np.power(np.sum(np.power(query_feature, 2), axis=1), 0.5)
    query_feature = query_feature.transpose()
    logger.info('query_feature:' + str(query_feature.shape))
    gallery_feature = gallery_feature.transpose() / np.power(np.sum(np.power(gallery_feature, 2), axis=1), 0.5)
    gallery_feature = gallery_feature.transpose()
    logger.info('gallery_feature:' + str(gallery_feature.shape))
    prid_labels = []
    for i in range(query_feature.shape[0]):
        prid_labels.append(gallery_targets[np.argmax(np.dot(gallery_feature, query_feature[i]))])
    return torch.tensor(prid_labels)

def spatial_temporal_distribution(camera_id, labels, frames):


    spatial_temporal_sum = np.zeros((class_num*2, c_size))
    spatial_temporal_count = np.zeros((class_num*2, c_size))
    eps = 0.0000001
    # interval = 100.0

    for i in range(len(camera_id)):
        label_k = labels[i]  #### not in order, done
        cam_k = camera_id[i] - 1  ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        spatial_temporal_sum[label_k][cam_k] = spatial_temporal_sum[label_k][cam_k] + frame_k
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum / (
            spatial_temporal_count + eps)  # spatial_temporal_avg: 751 ids, 8cameras, center point

    distribution = np.zeros((c_size, c_size, max_hist))
    for i in range(class_num):
        for j in range(c_size - 1):
            for k in range(j + 1, c_size):
                if spatial_temporal_count[i][j] == 0 or spatial_temporal_count[i][k] == 0:
                    continue
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij > st_ik:
                    diff = st_ij - st_ik
                    hist_ = int(diff / interval)
                    distribution[j][k][hist_] = distribution[j][k][hist_] + 1  # [big][small]
                else:
                    diff = st_ik - st_ij
                    hist_ = int(diff / interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_] + 1

    sum_ = np.sum(distribution, axis=2)
    for i in range(c_size):
        for j in range(c_size):
            distribution[i][j][:] = distribution[i][j][:] / (sum_[i][j] + eps)

    return distribution  # [to][from], to xxx camera, from xxx camera


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def gauss_smooth(arr):
    # print(gaussian_func(0,0))
    for u, element in enumerate(arr):
        # print(u," ",element)
        if element != 0:
            for index in range(0, 3000):
                arr[index] = arr[index] + element * gaussian_func(index, u)

    sum = 0
    for v in arr:
        sum = sum + v
    if sum == 0:
        return arr
    for i in range(0, 3000):
        arr[i] = arr[i] / sum
    return arr

#######################################################################
# Evaluate
def evaluate(qf, ql, qc, qfr, gf, gl, gc, gfr, distribution=None):
    query = qf
    score = np.dot(gf, query)
    # score = -np.sqrt(np.sum((gf - query)**2,1))
    if distribution is not None:
        # spatial temporal scores: qfr,gfr, qc, gc
        # TODO
        # interval = 100
        score_st = np.zeros(len(gc))
        for i in range(len(gc)):
            if qfr > gfr[i]:
                diff = qfr - gfr[i]
                hist_ = int(diff / interval)
                pr = distribution[qc - 1][gc[i] - 1][hist_]
            else:
                diff = gfr[i] - qfr
                hist_ = int(diff / interval)
                pr = distribution[gc[i] - 1][qc - 1][hist_]
            score_st[i] = pr

        # ========================
        score = 1 / (1 + np.exp(-alpha * score)) * 1 / (1 + 2 * np.exp(-alpha * score_st))
    else:
        score = score
    index = np.argsort(-score)  # from large to small
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def evaluate_rank1(qf, ql, qc, qfr, gf, gl, gc, gfr):
    query = qf
    score = np.dot(gf, query)
    index = np.argsort(-score)  # from large to small
    return gl[index][0] == ql


def evaluate_cuda(qf, ql, qc, qfr, gf, gl, gc, gfr, distribution=None):
    query = qf
    # score = np.dot(gf, query)
    score = torch.matmul(gf, query)
    # score = -np.sqrt(np.sum((gf - query)**2,1))
    if distribution is not None and not isinstance(distribution, list):
        # spatial temporal scores: qfr,gfr, qc, gc
        # TODO
        # interval = 100
        score_st = np.zeros(len(gc))
        for i in range(len(gc)):
            if qfr > gfr[i]:
                diff = qfr - gfr[i]
                hist_ = int(diff / interval)
                pr = distribution[qc - 1][gc[i] - 1][hist_]
            else:
                diff = gfr[i] - qfr
                hist_ = int(diff / interval)
                pr = distribution[gc[i] - 1][qc - 1][hist_]
            score_st[i] = pr
        # ==========================
        score_st = torch.tensor(score_st).cuda()
        # score = 1 / (1 + torch.exp(-alpha * score)) * 1 / (opt.p_w + opt.p_1 * torch.exp(-alpha/opt.p_2 * score_st+opt.p_3))
        score = 1 / (1 + torch.exp(-alpha * score)) * 1 / (1 + 2 * torch.exp(-alpha * score_st + 0.2))
    else:
        score = score
    score = score.cpu().numpy()
    index = np.argsort(-score)  # from large to small
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def evaluate_cuda_for_dismat(qf, ql, qc, qfr, gf, gl, gc, gfr, distribution=None):
    query = qf
    score = torch.matmul(gf, query)
    if distribution is not None and not isinstance(distribution,list):
        score_st = np.zeros(len(gc))
        for i in range(len(gc)):
            if qfr > gfr[i]:
                diff = qfr - gfr[i]
                hist_ = int(diff / interval)
                pr = distribution[qc - 1][gc[i] - 1][hist_]
            else:
                diff = gfr[i] - qfr
                hist_ = int(diff / interval)
                pr = distribution[gc[i] - 1][qc - 1][hist_]
            score_st[i] = pr
        # ==========================
        score_st = torch.tensor(score_st).cuda()
        score = 1 / (1 + torch.exp(-alpha * score)) * 1 / (1 + 2 * torch.exp(-alpha * score_st))
    else:
        score = score
    score = score.cpu().numpy()
    # index = np.argsort(-score)  # from large to small
    # query_index = np.argwhere(gl == ql)
    # camera_index = np.argwhere(gc == qc)
    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index1 = np.argwhere(gl == -1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    # CMC_tmp = compute_mAP(index, good_index, junk_index)
    return score
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--test_dir', default='./dataset/market_rename', type=str, help='./test_data')
    parser.add_argument('--test_dir_1', default='./dataset/market_rename', type=str, help='./test_data')
    parser.add_argument('--name', default='ft', type=str, help='save model path')
    parser.add_argument('--batchsize', default=48, type=int, help='batchsize')
    parser.add_argument('--model_name', default='densnet', type=str, )
    parser.add_argument('--LR_TYPE', default=0, type=int, )
    parser.add_argument('--MATCH_RANK1', default=0, type=int, )
    parser.add_argument('--load_model', default='net_last.pth', type=str, )
    parser.add_argument('--distribution_path', default='', type=str, )
    parser.add_argument('--alpha', default=5, type=float, help='alpha')
    parser.add_argument('--smooth', default=50, type=float, help='smooth')
    parser.add_argument('--dis_type', default=0, type=int)
    parser.add_argument('--ratio', default=3, type=int, )
    parser.add_argument('--hp_size', default=3, type=int, )

    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(',')
    name = opt.name
    test_dir = opt.test_dir
    test_dir_1 = opt.test_dir_1
    logger = setup_logger("pit", "logs/" + name, True)
    logger.info("start")
    logger.info(str(opt))
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = test_dir


    if "duke" in data_dir:
        class_num = 702
        max_hist = 3000
        c_size = 8
        interval = 100.0
    elif "msmt" in data_dir:
        max_hist = 6000
        class_num = 1041
        c_size = 15
        interval = 100.0
    elif "market" in data_dir:
        class_num = 751
        max_hist = 5000
        c_size = 8
        interval = 100.0
    elif "cuhk03" in data_dir:
        class_num = 767
        max_hist = 5000
        c_size = 8
        interval = 100.0
    elif "occ_reid" in data_dir:
        class_num = 767
        max_hist = 5000
        c_size = 8
        # interval = 100.0
        interval = 100.0

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
    #                   ['gallery', 'query', "train_all", "train_all_"+str(opt.ratio), "train_all_unlabel_"+ str(opt.ratio)]}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
    #                                               shuffle=False, num_workers=2) for x in
    #                ['gallery', 'query',"train_all", "train_all_"+str(opt.ratio), "train_all_unlabel_"+ str(opt.ratio)]}

    image_datasets = {}
    image_datasets["gallery"] = FsImageFolder(os.path.join(data_dir, "gallery"), data_transforms)
    image_datasets["query"] = FsImageFolder(os.path.join(data_dir, "query"), data_transforms)
    image_datasets["train_all"] = FsImageFolder(os.path.join(data_dir, "train_all"), data_transforms)
    image_datasets["train_all_"+str(opt.ratio)] = FsImageFolder(os.path.join(data_dir, "train_all"), data_transforms, ratio=opt.ratio, ratio_source=data_dir)
    image_datasets["train_all_unlabel_"+str(opt.ratio)] = FsImageFolder(os.path.join(data_dir, "train_all"), data_transforms, ratio=opt.ratio, ratio_source=data_dir,label_include=False)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=4) for x in
                   ['gallery', 'query', "train_all", "train_all_" + str(opt.ratio),
                    "train_all_unlabel_" + str(opt.ratio)]}

    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()

    if opt.dis_type in [5,]:
        data_dir_1 = test_dir_1
        image_datasets_1 = {}
        image_datasets_1["gallery"] = FsImageFolder(os.path.join(data_dir_1, "gallery"), data_transforms)
        image_datasets_1["query"] = FsImageFolder(os.path.join(data_dir_1, "query"), data_transforms)
        image_datasets_1["train_all"] = FsImageFolder(os.path.join(data_dir_1, "train_all"), data_transforms)
        image_datasets_1["train_all_" + str(opt.ratio)] = FsImageFolder(os.path.join(data_dir_1, "train_all"),
                                                                      data_transforms, ratio=opt.ratio,
                                                                      ratio_source=data_dir_1)
        image_datasets_1["train_all_unlabel_" + str(opt.ratio)] = FsImageFolder(os.path.join(data_dir_1, "train_all"),
                                                                              data_transforms, ratio=opt.ratio,
                                                                              ratio_source=data_dir_1,
                                                                              label_include=False)

        dataloaders_1 = {x: torch.utils.data.DataLoader(image_datasets_1[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=4) for x in
                       ['gallery', 'query', "train_all", "train_all_" + str(opt.ratio),
                        "train_all_unlabel_" + str(opt.ratio)]}

        class_names_1 = image_datasets_1['query'].classes

    def load_network(network):

        return network


    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip


    def extract_feature(model, dataloaders):
        features = torch.FloatTensor()
        count = 0
        for data in dataloaders:
            img, label,_ = data
            n, c, h, w = img.size()
            count += n
            if count % 1000 == 0:
                print(count)
            if "pcb" in opt.model_name:
                ff = None
                # ff = torch.FloatTensor(n, 1664).zero_()
                for i in range(2):
                    if (i == 1):
                        img = fliplr(img)
                    input_img = Variable(img.cuda())
                    outputs = model(input_img)
                    outputs = torch.cat([j for i in outputs for j in i],1)
                    f = outputs.data.cpu()
                    if ff is None:
                        ff = f
                    else:
                        ff = ff + f
            else:
                if opt.model_name == "densnet":
                    ff = torch.FloatTensor(n, 1664).zero_()
                elif opt.model_name == "resnet":
                    ff = torch.FloatTensor(n, 2048).zero_()
                elif opt.model_name == "vit":
                    ff = torch.FloatTensor(n, 512).zero_()
                for i in range(2):
                    if (i == 1):
                        img = fliplr(img)
                    input_img = Variable(img.cuda())
                    outputs = model(input_img)
                    f = outputs.data.cpu()
                    ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)
        return features


    def get_id(img_path):
        camera_id = []
        labels = []
        frames = []
        paths = []
        for path, v in img_path:
            filename = path.split(os.sep)[-1]
            filename = filename[:-4]
            f_l = filename.split("_")
            label = f_l[0]
            camera = f_l[1][1:]
            frame = f_l[2][1:]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera))
            frames.append(int(frame))
            paths.append(path)
        return camera_id, labels, frames, paths


    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label, gallery_frames, gallery_paths = get_id(gallery_path)
    query_cam, query_label, query_frames, query_paths = get_id(query_path)

    ######################################################################
    # Load Collected data Trained model

    if opt.model_name == "densnet":
        model = ft_net_dense(class_num)
    elif opt.model_name == "densnet_pcb":
        model = ft_net_dense_pcb(class_num)
    elif opt.model_name == "resnet_pcb":
        model = ft_net_pcb(class_num,opt.hp_size)
    elif opt.model_name == "resnet":
        model = ft_net(class_num)
    elif opt.model_name == "vit":
        model = vit(img_size=(256, 128), stride_size=12, embed_dim=512, num_heads=8, num_classes=class_num)
    #
    # model.load_state_dict(torch.load(os.path.join('model', name, opt.load_model)))

    param_dict = torch.load(os.path.join('model', name, opt.load_model))
    for i in param_dict:
        if i in model.state_dict().keys() and "sn_classifier" not in  i  and not i.startswith("classifier"):
            # print(i)
            model.state_dict()[i].copy_(param_dict[i])

    model = model.eval()
    if use_gpu:
        model = model.cuda()

    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])



    # Save to Matlab for check
    # result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
    #           'gallery_frames': gallery_frames, 'query_f': query_feature.numpy(), 'query_label': query_label,
    #           'query_cam': query_cam, 'query_frames': query_frames}
    # scipy.io.savemat('model/' + name + '/' + 'pytorch_result.mat', result)

    # result = scipy.io.loadmat('model/' + name + '/' + 'pytorch_result.mat')
    # query_feature = result['query_f']
    # gallery_feature = result['gallery_f']

    gallery_cam = np.array(gallery_cam)
    query_cam = np.array(query_cam)
    query_label = np.array(query_label)
    gallery_label = np.array(gallery_label)
    query_frames = np.array(query_frames)
    gallery_frames = np.array(gallery_frames)
    query_feature = query_feature.numpy()
    gallery_feature = gallery_feature.numpy()

    query_feature = query_feature.transpose() / np.power(np.sum(np.power(query_feature, 2), axis=1), 0.5)
    query_feature = query_feature.transpose()
    logger.info('query_feature:' + str(query_feature.shape))
    gallery_feature = gallery_feature.transpose() / np.power(np.sum(np.power(gallery_feature, 2), axis=1), 0.5)
    gallery_feature = gallery_feature.transpose()
    logger.info('gallery_feature:' + str(gallery_feature.shape))

    result = {'gallery_f': gallery_feature, 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'gallery_frames': gallery_frames, 'query_f': query_feature, 'query_label': query_label,
              'query_cam': query_cam, 'query_frames': query_frames, 'gallery_paths': gallery_paths,
              "query_paths": query_paths}
    scipy.io.savemat('model/' + name + '/ALL_FEATURE.mat', result)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    if opt.dis_type == 0:
        query_feature = torch.tensor(query_feature).cuda()
        gallery_feature = torch.tensor(gallery_feature).cuda()
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = evaluate_cuda(query_feature[i], query_label[i], query_cam[i], query_frames[i], gallery_feature,
                                       gallery_label, gallery_cam, gallery_frames)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label)  # average CMC
        logger.info('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    elif opt.dis_type in (1,2,3,4,5):
        if opt.dis_type == 1:
            train_path = image_datasets['train_all'].imgs
            train_cam, train_label, train_frames, paths = get_id(train_path)
            distribution = spatial_temporal_distribution(train_cam, image_datasets['train_all'].targets, train_frames)
        elif opt.dis_type == 3:
            train_path = image_datasets['train_all_'+str(opt.ratio)].imgs
            train_cam, train_label, train_frames, paths = get_id(train_path)
            distribution = spatial_temporal_distribution(train_cam, image_datasets['train_all_'+str(opt.ratio)].targets, train_frames)
        elif opt.dis_type == 4:
            # 生成特征提取流信息
            gallery_feature_3 = extract_feature(model, dataloaders['train_all_' + str(opt.ratio)])
            query_feature_3 = extract_feature(model, dataloaders['train_all_unlabel_' + str(opt.ratio)])

            gallery_path_3 = image_datasets['train_all_' + str(opt.ratio)].imgs
            query_path_3 = image_datasets['train_all_unlabel_' + str(opt.ratio)].imgs

            gallery_cam_3, _, gallery_frames_3, gallery_paths_3 = get_id(gallery_path_3)
            query_cam_3, _, query_frames_3, query_paths_3 = get_id(query_path_3)

            train_all_targets_imgs = [i[0] for i in dataloaders['train_all'].dataset.imgs]
            # train_all_targets_label = [i[1] for i in dataloaders['train_all'].dataset.imgs]
            gallery_targets = [dataloaders['train_all'].dataset.imgs[train_all_targets_imgs.index(
                img[0].replace('train_all_' + str(opt.ratio), "train_all"))][1] for img in
                               dataloaders['train_all_' + str(opt.ratio)].dataset.imgs]

            query_targets = [dataloaders['train_all'].dataset.imgs[train_all_targets_imgs.index(
                img[0].replace('train_all_unlabel_' + str(opt.ratio), "train_all"))][1] for img in
                             dataloaders['train_all_unlabel_' + str(opt.ratio)].dataset.imgs]

            query_pre_id = predict_id(query_feature_3, gallery_feature_3, gallery_targets)
            sss = torch.tensor(query_targets) == torch.tensor(query_pre_id)
            logger.info("命中率 " + str(torch.sum(sss).item() / sss.shape[0]))

            train_cam_3 = query_cam_3
            train_label_order_3 =  query_pre_id.numpy().tolist()
            train_frames_3 =  query_frames_3
            features_3 = torch.cat([query_feature_3], dim=0)

            distribution = spatial_temporal_distribution(train_cam_3, train_label_order_3, train_frames_3)
        elif opt.dis_type == 5:
            # 生成特征提取流信息 dataset1 预测 dataset2的伪标签
            gallery_feature_3 = extract_feature(model, dataloaders_1['train_all_' + str(opt.ratio)])
            query_feature_3 = extract_feature(model, dataloaders['train_all_unlabel_' + str(opt.ratio)])

            gallery_path_3 = image_datasets['train_all_' + str(opt.ratio)].imgs
            query_path_3 = image_datasets['train_all_unlabel_' + str(opt.ratio)].imgs

            gallery_cam_3, _, gallery_frames_3, gallery_paths_3 = get_id(gallery_path_3)
            query_cam_3, _, query_frames_3, query_paths_3 = get_id(query_path_3)

            train_all_targets_imgs_1 = [i[0] for i in dataloaders_1['train_all'].dataset.imgs]
            train_all_targets_imgs = [i[0] for i in dataloaders['train_all'].dataset.imgs]
            # train_all_targets_label = [i[1] for i in dataloaders['train_all'].dataset.imgs]
            gallery_targets = [dataloaders_1['train_all'].dataset.imgs[train_all_targets_imgs_1.index(
                img[0].replace('train_all_' + str(opt.ratio), "train_all"))][1] for img in
                               dataloaders_1['train_all_' + str(opt.ratio)].dataset.imgs]

            query_targets = [dataloaders['train_all'].dataset.imgs[train_all_targets_imgs.index(
                img[0].replace('train_all_unlabel_' + str(opt.ratio), "train_all"))][1] for img in
                             dataloaders['train_all_unlabel_' + str(opt.ratio)].dataset.imgs]

            query_pre_id = predict_id(query_feature_3, gallery_feature_3, gallery_targets)
            sss = torch.tensor(query_targets) == torch.tensor(query_pre_id)
            logger.info("命中率 " + str(torch.sum(sss).item() / sss.shape[0]))

            train_cam_3 = query_cam_3
            train_label_order_3 =  query_pre_id.numpy().tolist()
            train_frames_3 =  query_frames_3
            features_3 = torch.cat([query_feature_3], dim=0)

            distribution = spatial_temporal_distribution(train_cam_3, train_label_order_3, train_frames_3)
        elif opt.dis_type == 2:
            # 生成特征提取流信息
            gallery_feature_3 = extract_feature(model, dataloaders['train_all_'+str(opt.ratio)])
            query_feature_3 = extract_feature(model, dataloaders['train_all_unlabel_'+str(opt.ratio)])

            gallery_path_3 = image_datasets['train_all_'+str(opt.ratio)].imgs
            query_path_3 = image_datasets['train_all_unlabel_'+str(opt.ratio)].imgs

            gallery_cam_3, _, gallery_frames_3, gallery_paths_3 = get_id(gallery_path_3)
            query_cam_3, _, query_frames_3, query_paths_3 = get_id(query_path_3)

            train_all_targets_imgs = [i[0] for i in dataloaders['train_all'].dataset.imgs]
            # train_all_targets_label = [i[1] for i in dataloaders['train_all'].dataset.imgs]
            gallery_targets = [dataloaders['train_all'].dataset.imgs[train_all_targets_imgs.index(
                img[0].replace('train_all_' + str(opt.ratio), "train_all"))][1] for img in
                               dataloaders['train_all_' + str(opt.ratio)].dataset.imgs]

            query_targets = [dataloaders['train_all'].dataset.imgs[train_all_targets_imgs.index(
                img[0].replace('train_all_unlabel_' + str(opt.ratio), "train_all"))][1] for img in
                               dataloaders['train_all_unlabel_' + str(opt.ratio)].dataset.imgs]

            query_pre_id = predict_id(query_feature_3, gallery_feature_3, gallery_targets)
            sss = torch.tensor(query_targets) == torch.tensor(query_pre_id)
            logger.info("命中率 " + str(torch.sum(sss).item() / sss.shape[0]))

            train_cam_3 = gallery_cam_3 + query_cam_3
            train_label_order_3 = gallery_targets + query_pre_id.numpy().tolist()
            train_frames_3 = gallery_frames_3 + query_frames_3
            features_3 = torch.cat([gallery_feature_3, query_feature_3], dim=0)

            distribution = spatial_temporal_distribution(train_cam_3, train_label_order_3, train_frames_3)

        alpha = opt.alpha
        smooth = opt.smooth


        # gauss处理
        for i in range(0, c_size):
            for j in range(0, c_size):
                print("gauss " + str(i) + "->" + str(j))
                distribution[i][j][:] = gauss_smooth2(distribution[i][j][:], smooth)

        eps = 0.0000001
        sum_ = np.sum(distribution, axis=2)
        for i in range(c_size):
            for j in range(c_size):
                distribution[i][j][:] = distribution[i][j][:] / (sum_[i][j] + eps)

        # scipy.io.savemat('distribution.mat', {"distribution": distribution})
        #############################################################

        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        query_feature = torch.tensor(query_feature).cuda()
        gallery_feature = torch.tensor(gallery_feature).cuda()
        if True:
            # logger.info(query_label)
            for i in range(len(query_label)):
                ap_tmp, CMC_tmp = evaluate_cuda(query_feature[i], query_label[i], query_cam[i], query_frames[i], gallery_feature,
                                           gallery_label, gallery_cam, gallery_frames, distribution)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                ap += ap_tmp
                print(i, CMC_tmp[0])
                # logger.info(str(i))
                # if i%10==0:
                #     logger.info('i:',i)
            CMC = CMC.float()
            CMC = CMC / len(query_label)  # average CMC
            logger.info('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
        else:
            query_score_list = []
            for i in range(len(query_label)):
                query_score = evaluate_cuda_for_dismat(query_feature[i], query_label[i], query_cam[i], query_frames[i],
                                                gallery_feature,
                                                gallery_label, gallery_cam, gallery_frames, distribution)
                query_score_list.append(query_score)
            scipy.io.savemat("MST_distmat.mat",{"distmat":np.asarray(query_score_list), 'query': np.asarray(query_path), 'gallery': np.asarray(gallery_paths),"query_ids":query_label,"gallery_ids":np.asarray(gallery_label)},)
