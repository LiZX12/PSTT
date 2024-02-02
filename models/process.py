import copy
import json
import logging
import os
import time
import random

import torch
import torch.nn as nn
from pytorch_metric_learning.distances import LpDistance
from timm.utils import AverageMeter
from torch.autograd import Variable
from torchvision.transforms import transforms

import torch.distributed as dist
from torch.cuda import amp
import torch.nn.functional as F
import numpy as np


def do_train(cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,
                loss_fn,
                num_query, local_rank):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"

    logger = logging.getLogger("pit.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)

    cls_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.MODEL.DIVERSITY:
        div_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()
    isVideo = True if cfg.DATASETS.NAMES in ['mars', 'duke-video-reid', 'ilids', 'prid', "market1501"] else False
    freeze_layers = ['base', 'pyramid_layer']
    freeze_epochs = cfg.SOLVER.WARMUP_EPOCHS
    freeze_or_not = cfg.MODEL.FREEZE
    # loss_ver = torch.nn.BCEWithLogitsLoss(size_average=True)
    # loss_ver = F.binary_cross_entropy()
    tt_r = TripletLoss(0.2)

    # aur_trans = gray_scale(p=0.7)

    epochs = cfg.SOLVER.MAX_EPOCHS
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        cls_loss_meter.reset()
        tri_loss_meter.reset()

        acc_meter.reset()
        evaluator.reset()
        if cfg.MODEL.SCHEDULER == "cosine":
            scheduler.step(epoch)
        model.train()
        # if freeze_or_not and epoch <= cfg.SOLVER.FREZZ_EPOCHS:  # freeze layers for 2000 iterations
        #     for name, module in model.named_children():
        #         if name in ['base']:
        #             module.eval()
        for n_iter, (img, vid, target_cam, target_view, target_frame_index) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img_0 = img[0].to(device)
            img_1 = img[1].to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                if cfg.MODEL.LR_TYPE == "FT":
                    # target = target.repeat(2)
                    # target_cam = target_cam.repeat(2)
                    # img = torch.cat([img_0, img_1], dim=0)
                    img = img_0
                    score, feat = model(img)
                    ID_LOSS, TRI_LOSS = loss_fn(score, feat, target, target_cam)
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                elif cfg.MODEL.LR_TYPE == "SN":
                    # target = target.repeat(2)
                    # target_cam = target_cam.repeat(2)
                    img =img_0

                    B, _, _, _ = img.shape
                    rand_id = [i for i in range(B)]

                    c_b = B // 4
                    # rand_id = rand_id[0:B // 2] + rand_id[B // 2:][:c_b] + rand_id[B // 2:][c_b:][1:] + [rand_id[B // 2:][c_b:][0]]
                    rand_id = rand_id[0::2] + rand_id[1::2][0:c_b] + rand_id[1::2][c_b+1:] + [rand_id[1::2][c_b]]

                    img = img[rand_id]
                    target = target[rand_id]
                    target_cam = target_cam[rand_id]

                    b = B // 4
                    score, feat, score_ver = model(img[:B//2], img[B//2:])

                    # 孪生网络 类分类
                    pp = torch.abs(target[:b * 2] - target[b * 2:]) + 1
                    pp[pp > 1] = 0
                    ID_LOSS, TRI_LOSS = loss_fn(score, feat, target, None)
                    VER_LOSS = loss_fn([[score_ver, ], ], [], pp, None)[0]

                    loss = 0.1 * TRI_LOSS + 0.01 * VER_LOSS
                    ID_LOSS = VER_LOSS
                    TRI_LOSS = TRI_LOSS

                elif cfg.MODEL.LR_TYPE == "FS":
                    # 计算所有照片的分数 unlabel 与 label的分数
                    B, _, _, _ = img.shape
                    score, fs, feats = model(img[0::3], img[1::3], img[2::3])
                    img = aur_trans(img)
                    _, fs_arg, feats_arg = model(img[0::3], img[1::3], img[2::3])
                    # fs = model(img)

                    triplet_score = torch.exp(fs[0] - fs[1]) / (1 + torch.exp(fs[0] - fs[1]))
                    triplet_score_arg = torch.exp(fs_arg[0] - fs_arg[1]) / (1 + torch.exp(fs_arg[0] - fs_arg[1]))

                    fs[0] = disf.fusion_mul_2_mul(fs[0], target_cam[0::3], target_cam[1::3], target_frame_index[0::3],
                                                  target_frame_index[1::3])
                    fs[1] = disf.fusion_mul_2_mul(fs[1], target_cam[0::3], target_cam[2::3], target_frame_index[0::3],
                                                  target_frame_index[2::3])

                    fs_arg[0] = disf.fusion_mul_2_mul(fs_arg[0], target_cam[0::3], target_cam[1::3],
                                                      target_frame_index[0::3], target_frame_index[1::3])
                    fs_arg[1] = disf.fusion_mul_2_mul(fs_arg[1], target_cam[0::3], target_cam[2::3],
                                                      target_frame_index[0::3], target_frame_index[2::3])

                    triplet_score_fu = torch.exp(fs[0] - fs[1]) / (1 + torch.exp(fs[0] - fs[1]))
                    triplet_score_arg_fu = torch.exp(fs_arg[0] - fs_arg[1]) / (1 + torch.exp(fs_arg[0] - fs_arg[1]))

                    ID_LOSS = F.cross_entropy(triplet_score, triplet_score_fu) + F.cross_entropy(triplet_score_arg,
                                                                                                 triplet_score_arg_fu)

                    feat = torch.cat([feats, feats_arg], dim=0)
                    # ID_LOSS, TRI_LOSS = loss_fn([],[[torch.cat([feats, feats_arg],dim=0),]],target.repeat(2),None)
                    hard_pairs = miner(feat, target.repeat(2))
                    hard_pairs = [hard_pairs[0], torch.cat([hard_pairs[0][B:], hard_pairs[0][:B]], dim=0),
                                  hard_pairs[2]]
                    # fe = sft(feat[0][0])
                    TRI_LOSS = loss_func__(feat, target.repeat(2), hard_pairs)
                    # TRI_LOSS = loss_func__(feat, target.repeat(2))
                    loss = 0.1 * TRI_LOSS + 0.1 * ID_LOSS
                    score = feats

            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if cfg.MODEL.SCHEDULER != "cosine":
                scheduler.step()

            if isinstance(score, list):
                acc = (score[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            cls_loss_meter.update(ID_LOSS.item(), img.shape[0])
            tri_loss_meter.update(TRI_LOSS.item(), img.shape[0])

            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if cfg.MODEL.SCHEDULER == "WarmupMultiStepLR" or cfg.MODEL.SCHEDULER == "cosine":
                    lrr = scheduler._get_lr(epoch)[0]
                else:
                    lrr = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] cls_loss: {:.3f}, tri_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (n_iter + 1), len(train_loader),
                                cls_loss_meter.avg, tri_loss_meter.avg, acc_meter.avg, lrr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            if isVideo:
                num_samples = cfg.DATALOADER.P * cfg.DATALOADER.K * cfg.DATALOADER.NUM_TRAIN_IMAGES
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                            .format(epoch, time_per_batch, num_samples / time_per_batch))
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                            .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, str(num + 1), cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 and epoch not in cfg.RIGHT_STREAM.EVA_EPOCH_EXCLUDE_LIST:
            model.eval()
            do_lr_inference(cfg, model, val_loader, num_query)
            torch.cuda.empty_cache()

    # return cmc, mAP


def get_rand_id(B, repeat_num=0):
    rand_id = [i for i in range(B)]
    while sum(rand_id[:B // 2] == rand_id[B // 2:]) < repeat_num:
        rand_id = [i for i in range(B)]

    return rand_id


def do_lr_inference(cfg,
                    model,
                    val_loader,
                    num_query):
    device = "cuda"
    logger = logging.getLogger("pit.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))
            # img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_lr_fusion_inference(cfg,
                           model,
                           val_loader,
                           num_query, out_mask=None):
    device = "cuda"
    logger = logging.getLogger("pit.test")
    logger.info("Enter inferencing")

    evaluator = Fusion_R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, target_frame_index) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            # target_frame_index = target_frame_index.to(device)

            feat = model(Variable(img)) + model(Variable(fliplr(img)))
            fnorm = torch.norm(feat, p=2, dim=1, keepdim=True)
            feat = feat.div(fnorm.expand_as(feat))

            # feat = model(img)

            evaluator.update((feat, pid, camid, target_frame_index))
            # img_path_list.extend(imgpath)

    import pandas as pd
    import numpy as np
    # img_path_list = np.asarray(img_path_list)
    # data = pd.DataFrame({str(i): img_path_list[:, i] for i in range(img_path_list.shape[1])})
    # data.to_csv('img_path.csv', index=True, sep=',')

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.tensor(torch.arange(img.size(3) - 1, -1, -1)).to("cuda")  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip
