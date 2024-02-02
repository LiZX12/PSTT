# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import math
import os

import numpy as np
import scipy.io
import torch
from torch.autograd import Variable
from torchvision import transforms

from logger import setup_logger
from model import ft_net, ft_net_dense, ft_net_pcb
from utils.sampler import FsImageFolder

if __name__ == '__main__':

    ######################################################################
    # Options
    # --------
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir', default='./dataset/market_rename', type=str, help='./test_data')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
    # parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--use_dense', type=bool, default=True, help='use densenet121')
    parser.add_argument('--PCB', action='store_true', help='use PCB')
    parser.add_argument('--model_name', default='densnet', type=str, )
    parser.add_argument('--LR_TYPE', default=0, type=int, )
    parser.add_argument('--alpha', default=5, type=float, help='alpha')
    parser.add_argument('--smooth', default=50, type=float, help='smooth')
    parser.add_argument('--LOAD_MODEL', default="", type=str, )
    parser.add_argument('--nost', default=0, type=int, help='smooth')
    parser.add_argument('--ratio', default=6, type=int)

    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(',')
    # which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir
    logger = setup_logger("pit", "logs/" + name, True)
    logger.info("start")
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    data_transforms = transforms.Compose([
        transforms.Resize((288, 144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = test_dir
    image_datasets = {}
    image_datasets["gallery"] = FsImageFolder(os.path.join(data_dir, "gallery"), data_transforms)
    image_datasets["query"] = FsImageFolder(os.path.join(data_dir, "query"), data_transforms)
    image_datasets["train_all"] = FsImageFolder(os.path.join(data_dir, "train_all"), data_transforms)
    image_datasets["train_all_"+str(opt.ratio)] = FsImageFolder(os.path.join(data_dir, "train_all"), data_transforms, ratio=opt.ratio, ratio_source=data_dir)
    image_datasets["train_all_unlabel_"+str(opt.ratio)] = FsImageFolder(os.path.join(data_dir, "train_all"), data_transforms, ratio=opt.ratio, ratio_source=data_dir,label_include=False)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=4) for x in
                   ['gallery', 'query',"train_all", "train_all_"+str(opt.ratio), "train_all_unlabel_"+str(opt.ratio)]}

    class_names = image_datasets['query'].classes
    use_gpu = torch.cuda.is_available()


    ######################################################################
    # Load model
    # ---------------------------
    def load_network(network):
        save_path = os.path.join(opt.LOAD_MODEL)
        network.load_state_dict(torch.load(save_path))
        return network


    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip


    def extract_feature(model, dataloaders):
        features = None
        for data in dataloaders:
            img, label, _ = data
            if "pcb" in opt.model_name:
                ff = None
                # ff = torch.FloatTensor(n, 1664).zero_()
                for i in range(2):
                    if (i == 1):
                        img = fliplr(img)
                    input_img = Variable(img.cuda())
                    outputs = model(input_img)
                    outputs = torch.cat([j for i in outputs for j in i], 1)
                    f = outputs.data.cpu()
                    if ff is None:
                        ff = f
                    else:
                        ff = ff + f
            else:
                outputs_0 = model(Variable(img.cuda())).reshape(img.shape[0], -1)
                outputs_1 = model(Variable(fliplr(img).cuda())).reshape(img.shape[0], -1)
                ff = outputs_0.data.cpu() + outputs_1.data.cpu()
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            if features == None:
                features = ff
            else:
                features = torch.cat((features, ff), 0)

        return features


    def get_id(img_path):
        camera_id = []
        labels = []
        frames = []
        paths = []
        for path, v in img_path:
            filename = path.split(os.sep)[-1]
            # camera = filename.split('c')[1]
            filename = filename[:-4]
            f_l = filename.split("_")

            # label = f_l[0]
            camera = f_l[1][1:]
            frame = f_l[2][1:]

            # frame = filename.split('_')[2][1:]

            labels.append(v)
            camera_id.append(int(camera[0]))
            frames.append(int(frame))
            paths.append(path)
        return camera_id, labels, frames, paths


    # gallery_path = image_datasets['train_all_'+str(opt.ratio)].imgs
    gallery_path = image_datasets['train_all_' + str(opt.ratio)].imgs
    query_path = image_datasets['train_all_unlabel_' + str(opt.ratio)].imgs
    # query_path = image_datasets['train_all'].imgs

    gallery_cam, gallery_label, gallery_frames, gallery_paths = get_id(gallery_path)
    query_cam, query_label, query_frames, query_paths = get_id(query_path)

    ######################################################################
    # Load Collected data Trained model
    if "duke" in data_dir:
        class_num = 702
    elif "market" in data_dir:
        class_num = 751
    elif "msmt" in data_dir:
        class_num = 1041
    elif "cuhk" in data_dir:
        class_num = 767

    print('-------test-----------')
    if opt.model_name == "densnet":
        model = ft_net_dense(class_num)
    elif opt.model_name == "resnet":
        model = ft_net(class_num)
    elif opt.model_name == "resnet_pcb":
        model = ft_net_pcb(class_num)
    # 加载模型
    model.load_state_dict(torch.load(os.path.join(opt.LOAD_MODEL)))

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # 生成特征提取流信息
    gallery_feature = extract_feature(model, dataloaders['train_all_' + str(opt.ratio)])
    # query_feature = extract_feature(model, dataloaders['train_all'])
    query_feature = extract_feature(model, dataloaders['train_all_unlabel_' + str(opt.ratio)])


    def predict_id(query_feature, gallery_feature, gallery_targets):
        query_feature = query_feature.numpy()
        gallery_feature = gallery_feature.numpy()

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
    # 重新获取所有的targets
    train_all_targets_imgs = [i[0] for i in dataloaders['train_all'].dataset.imgs]
    # train_all_targets_label = [i[1] for i in dataloaders['train_all'].dataset.imgs]
    gallery_targets = [dataloaders['train_all'].dataset.imgs[
                           train_all_targets_imgs.index(img[0].replace('train_all_' + str(opt.ratio), "train_all"))][1]
                       for img in dataloaders['train_all_' + str(opt.ratio)].dataset.imgs]
    # query_targets = [dataloaders['train_all'].dataset.imgs[train_all_targets_imgs.index(img[0].replace('train_all',"train_all"))][1] for img in dataloaders['train_all'].dataset.imgs]
    query_targets = [dataloaders['train_all'].dataset.imgs[train_all_targets_imgs.index(
        img[0].replace('train_all_unlabel_' + str(opt.ratio), "train_all"))][1] for img in
                     dataloaders['train_all_unlabel_' + str(opt.ratio)].dataset.imgs]

    query_pre_id = predict_id(query_feature, gallery_feature, gallery_targets)
    # sss = torch.tensor(dataloaders['train_all_unlabel_'+str(opt.ratio)].dataset.targets) == torch.tensor(unlabel_pre_id)
    sss = torch.tensor(query_targets) == torch.tensor(query_pre_id)
    logger.info("命中率 " + str(torch.sum(sss).item() / sss.shape[0]))

    train_cam = gallery_cam + query_cam
    train_label_order = gallery_label + query_pre_id.numpy().tolist()
    train_frames = gallery_frames + query_frames


    # features = torch.cat([gallery_feature, query_feature], dim=0)

    # train_cam = query_cam
    # train_label_order = query_pre_id.numpy().tolist()
    # train_frames = query_frames

    # # Save to Matlab for check
    # result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
    #           'gallery_frames': gallery_frames, 'query_f': query_feature.numpy(), 'query_label': query_label,
    #           'query_cam': query_cam, 'query_frames': query_frames, 'gallery_paths': gallery_paths,
    #           "query_paths": query_paths}
    # scipy.io.savemat('model/' + name + '/ALL_FEATURE.mat', result)

    def spatial_temporal_distribution(camera_id, labels, frames):
        if "duke" in data_dir:
            class_num = 702
            max_hist = 3000
            c_size = 8
            interval = 100.0
        elif "msmt" in data_dir:
            class_num = 1041
            max_hist = 50
            c_size = 15
            interval = 100
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
        spatial_temporal_sum = np.zeros((class_num, c_size))
        spatial_temporal_count = np.zeros((class_num, c_size))
        eps = 0.0000001

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

        # result = {'distribution': distribution}
        # scipy.io.savemat('model/' + name + '/' + 'distribution_of_0.mat', result)  # 出现的次数

        sum_ = np.sum(distribution, axis=2)
        for i in range(c_size):
            for j in range(c_size):
                distribution[i][j][:] = distribution[i][j][:] / (sum_[i][j] + eps)

        # result = {'distribution': distribution}
        # scipy.io.savemat('model/' + name + '/' + 'distribution_of_1.mat', result) # 出现的次数

        return distribution  # [to][from], to xxx camera, from xxx camera


    def gaussian_func(x, u, o=0.1):
        if (o == 0):
            print("In gaussian, o shouldn't equel to zero")
            return 0
        temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
        temp2 = -(math.pow(x - u, 2)) / (2 * math.pow(o, 2))
        return temp1 * math.exp(temp2)


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


    distribution = spatial_temporal_distribution(train_cam, train_label_order, train_frames)

    # for i in range(0,8):
    #     for j in range(0,8):
    #         print("gauss "+str(i)+"->"+str(j))
    #         gauss_smooth(distribution[i][j])

    # result = {'distribution': distribution}
    # scipy.io.savemat('model/' + name + '/' + 'distribution.mat', result)

    # 融合
    ###########################
    alpha = opt.alpha
    smooth = opt.smooth


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


    query_feature = query_feature.numpy()
    gallery_feature = gallery_feature.numpy()

    query_feature = query_feature.transpose() / np.power(np.sum(np.power(query_feature, 2), axis=1), 0.5)
    query_feature = query_feature.transpose()
    logger.info('query_feature:' + str(query_feature.shape))
    gallery_feature = gallery_feature.transpose() / np.power(np.sum(np.power(gallery_feature, 2), axis=1), 0.5)
    gallery_feature = gallery_feature.transpose()
    logger.info('gallery_feature:' + str(gallery_feature.shape))

    #############################################################
    # result2 = scipy.io.loadmat('model/' + name + '/' + 'pytorch_result2.mat')
    # distribution = result2['distribution']  # 8 8 5000 c c time
    # if opt.nost == 0:
    #############################################################
    for i in range(0, 8):
        for j in range(0, 8):
            print("gauss " + str(i) + "->" + str(j))
            distribution[i][j][:] = gauss_smooth2(distribution[i][j][:], smooth)

    eps = 0.0000001
    sum_ = np.sum(distribution, axis=2)
    for i in range(8):
        for j in range(8):
            distribution[i][j][:] = distribution[i][j][:] / (sum_[i][j] + eps)

    # scipy.io.savemat('distribution.mat', {"distribution": distribution})
    # result = {'distribution': distribution}
    # scipy.io.savemat('model/' + name + '/' + 'distribution_of_2.mat', result) # 出现的次数
    #############################################################


    def evaluate_SCORE(qf, ql, qc, qfr, gf, gl, gc, gfr, distribution):
        query = qf
        score = np.dot(gf, query)

        # spatial temporal scores: qfr,gfr, qc, gc
        # TODO
        interval = 100
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
        if opt.nost == 0:
            score = 1 / (1 + np.exp(-alpha * score)) * 1 / (1 + 2 * np.exp(-alpha * score_st))
        return score


    score_list = []
    for i in range(len(query_label)):
        score_list.append(
            evaluate_SCORE(query_feature[i], query_label[i], query_cam[i], query_frames[i], gallery_feature,
                           gallery_label,
                           gallery_cam, gallery_frames, distribution))

    if not os.path.exists(os.path.join("model", name)):
        os.makedirs(os.path.join("model", name))

    scipy.io.savemat(os.path.join("model", name, 'zzzz.mat'),
                     {"score": np.array(score_list), "gallery_paths": gallery_paths, "query_paths": query_paths})
