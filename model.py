import math

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

######################################################################
import ResNet
from models.models.densenet import densenet169


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class NormBlock(nn.Module):
    def __init__(self, input_dim, dropout=True, relu=True, num_bottleneck=512):
        super(NormBlock, self).__init__()
        add_block = []
        # add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        # if relu:
        #     add_block += [nn.LeakyReLU(0.1)]
        # if dropout:
        #     add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.add_block = add_block

    def forward(self, x):
        x = self.add_block(x)
        return x


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x, return_norm=False):
        if return_norm:
            x_norm = self.add_block(x)
            x = self.classifier(x_norm)
            return x, x_norm
        else:
            x = self.add_block(x)
            x = self.classifier(x)
            return x


class RankClassBlock(nn.Module):
    def __init__(self, input_dim, dropout=True, relu=True, num_bottleneck=512):
        super(RankClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block

    def forward(self, x):
        return self.add_block(x)


class PCBClassBlock(nn.Module):
    def __init__(self, input_dim, num_bottleneck=512):
        super(PCBClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Conv2d(input_dim, num_bottleneck, 1, bias=False)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        add_block += [nn.ReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block

    def forward(self, x):
        return self.add_block(x)


class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num,forward_auto_class=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.forward_auto_class = forward_auto_class

        def create_classfier(c_num_, num_classes_):
            lr = nn.Linear(c_num_, num_classes_, bias=False)
            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        def create_two_classfier(c_num_):
            lr = nn.Sequential(
                nn.Linear(c_num_, c_num_ // 2, bias=False),
                nn.PReLU(),
                nn.Linear(c_num_ // 2, 1, bias=False)
            )

            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        self.classifier = create_classfier(2048, class_num)
        self.sn_classifier = create_two_classfier(2048)
        # self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        # if self.forward_auto_class:
        #     x = self.classifier(x)
        # x = self.classifier(x)
        return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class ft_net_pcb(nn.Module):
    def __init__(self, class_num, hp_size=3,forward_auto_class=False):
        super(ft_net_pcb, self).__init__()
        model_ft = ResNet.resnet50(pretrained=True, last_conv_stride=1)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.forward_auto_class = forward_auto_class

        self.classifier = nn.ModuleList()
        self.pcb_classifier = nn.ModuleList()
        self.sn_classifier = nn.ModuleList()
        # self.part = 6
        self.combine = [[1, 2, 4][i] for i in range(hp_size)]
        for i in self.combine:
            self.classifier.append(nn.ModuleList())
            self.pcb_classifier.append(nn.ModuleList())
            self.sn_classifier.append(nn.ModuleList())
            for j in range(i):
                self.classifier[len(self.classifier) - 1].append(RankClassBlock(256, True, False, class_num))
                self.pcb_classifier[len(self.pcb_classifier) - 1].append(PCBClassBlock(2048, 256))
                self.sn_classifier[len(self.pcb_classifier) - 1].append(PCBClassBlock(256, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        def create_classfier(c_num_, num_classes_):
            lr = nn.Linear(c_num_, num_classes_, bias=False)
            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        def create_two_classfier(c_num_):
            lr = nn.Sequential(
                nn.Linear(c_num_, c_num_ // 2, bias=False),
                nn.PReLU(),
                nn.Linear(c_num_ // 2, 1, bias=False)
            )

            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        self.classifier_ = create_classfier(256*sum([1,2,4][:hp_size]), class_num)
        self.classifier_1 = create_classfier(256*7, class_num)
        self.classifier_2 = create_classfier(256*7, class_num)
        self.classifier_3 = create_classfier(256*7, class_num)
        self.sn_classifier_ = create_two_classfier(256*7)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        output = []
        for i in range(len(self.combine)):
            output.append([])
            for k in range(self.combine[i]):
                if i == 0:
                    feat = x
                elif i in (1,2):
                    # 进行h划分 b c h w
                    l = x.shape[-2] // self.combine[i]
                    feat = x[:, :, l * k:l * (k + 1)]
                # elif i == 2:
                #     进行w划分
                    # l = x.shape[-1] // self.combine[i]
                    # feat = x[:, :, :, l * k:l * (k + 1)]
                elif i == 3:
                    h_ = k // self.combine[i-1]
                    w_ = k % self.combine[i-1]
                    l_h = x.shape[-2] // self.combine[i-2]
                    l_w = x.shape[-1] // self.combine[i-1]
                    feat = x[:, :,  l_h * h_:l_h * (h_ + 1), l_w * w_:l_w * (w_ + 1)]
                output[-1].append(self.pcb_classifier[i][k](self.maxpool(feat)).reshape(feat.size(0), -1))
        if self.forward_auto_class:
            feats_ = torch.cat([j for i in output for j in i], 1)
            output = self.classifier_(feats_)
        return output

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


# Define the DenseNet121-based Model
class ft_net_dense_old(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet169(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1664, class_num)
        self.sn_classifier = ClassBlock(1664, 1)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        # x = self.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet169(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()

        def create_classfier(c_num_, num_classes_):
            lr = nn.Linear(c_num_, num_classes_, bias=False)
            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        def create_two_classfier(c_num_):
            lr = nn.Sequential(
                nn.Linear(c_num_, c_num_ // 2, bias=False),
                nn.PReLU(),
                nn.Linear(c_num_ // 2, 1, bias=False)
            )

            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = create_classfier(1664, class_num)
        self.sn_classifier = create_two_classfier(1664)

    def forward(self, x):

        return torch.squeeze(self.avg(self.model.features(x)))
        # x = self.model.features(x)
        # x = torch.squeeze(x)
        # if return_feat:
        #     return self.classifier(x, return_norm), x
        # else:
        #     return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class ft_net_dense_pcb(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet169(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()

        def create_classfier(c_num_, num_classes_):
            lr = nn.Linear(c_num_, num_classes_, bias=False)
            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        def create_two_classfier(c_num_):
            lr = nn.Sequential(
                nn.Linear(c_num_, c_num_ // 2, bias=False),
                nn.PReLU(),
                nn.Linear(c_num_ // 2, 1, bias=False)
            )

            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        # For DenseNet, the feature dim is 1024

        self.pcb_feat = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.sn_classifier = nn.ModuleList()
        self.combine = [1, 4, 2]
        for i in self.combine:
            self.classifier.append(nn.ModuleList())
            self.pcb_feat.append(nn.ModuleList())
            self.sn_classifier.append(nn.ModuleList())
            for j in range(i):
                # self.pcb_feat[len(self.pcb_feat) - 1].append(create_classfier(1664, 768))
                # self.classifier[len(self.classifier) - 1].append(create_classfier(768, class_num))
                # self.sn_classifier[len(self.sn_classifier) - 1].append(create_two_classfier(768))

                self.pcb_feat[len(self.pcb_feat) - 1].append(PCBClassBlock(1664, 1024))
                self.classifier[len(self.classifier) - 1].append(RankClassBlock(1024, True, False, class_num))
                self.sn_classifier[len(self.sn_classifier) - 1].append(RankClassBlock(1024, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.model.features[-4].pool = nn.Sequential()

    def forward(self, x):

        x = self.model.features[:-1](x)
        # c = self.model.features[-1](s)
        output = []
        for i in range(len(self.combine)):
            output.append([])
            for k in range(self.combine[i]):
                if i == 0:
                    feat = x
                    output[-1].append(torch.squeeze(self.maxpool(feat)))
                elif i == 1:
                    # 进行水平划分
                    l = x.shape[-2] // self.combine[i]
                    feat = x[:, :, l * k:l * (k + 1)]
                    output[-1].append(torch.squeeze(self.pcb_feat[i][k](self.maxpool(feat))))
                elif i == 2:
                    # 进行水平划分
                    l = x.shape[-1] // self.combine[i]
                    feat = x[:, :, :, l * k:l * (k + 1)]
                    # output[-1].append(self.pcb_feat[i][k](torch.squeeze(self.maxpool(feat))))
                    output[-1].append(torch.squeeze(self.pcb_feat[i][k](self.maxpool(feat))))
        return output

        # return torch.squeeze(self.avg(self.model.features(x)))
        # x = self.model.features(x)
        # x = torch.squeeze(x)
        # if return_feat:
        #     return self.classifier(x, return_norm), x
        # else:
        #     return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class ft_net_dense_121(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet169(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()

        def create_classfier(c_num_, num_classes_):
            lr = nn.Linear(c_num_, num_classes_, bias=False)
            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        def create_two_classfier(c_num_):
            lr = nn.Sequential(
                nn.Linear(c_num_, c_num_ // 2, bias=False),
                nn.PReLU(),
                nn.Linear(c_num_ // 2, 1, bias=False)
            )

            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = create_classfier(1664, class_num)
        self.sn_classifier = create_two_classfier(1664)

    def forward(self, x):
        return torch.squeeze(self.avg(self.model.features(x)))
        # x = self.model.features(x)
        # x = torch.squeeze(x)
        # if return_feat:
        #     return self.classifier(x, return_norm), x
        # else:
        #     return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class ft_net_dense_PCB(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        model_ft = models.densenet169(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.part = 6

        def create_classfier(c_num_, num_classes_):
            lr = nn.Linear(c_num_, num_classes_, bias=False)
            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        def create_two_classfier(c_num_):
            lr = nn.Sequential(
                nn.Linear(c_num_, c_num_ // 2, bias=False),
                nn.PReLU(),
                nn.Linear(c_num_ // 2, 1, bias=False)
            )

            lr.apply(self.weights_init_classifier)
            bnr = nn.BatchNorm1d(c_num_)
            bnr.bias.requires_grad_(False)
            bnr.apply(self.weights_init_kaiming)
            return nn.Sequential(bnr, lr)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = nn.ModuleList()
        self.sn_classifier = nn.ModuleList()
        self.combine = [1, self.part, ]
        for i in self.combine:
            self.classifier.append(nn.ModuleList())
            self.sn_classifier.append(nn.ModuleList())
            for j in range(i):
                # self.classifier[len(self.classifier)-1].append( create_classfier(1664, class_num))
                # self.sn_classifier[len(self.classifier)-1].append(create_two_classfier(1664))

                self.classifier[len(self.classifier) - 1].append(ClassBlock(1664, class_num, True, False, 256))
                self.sn_classifier[len(self.classifier) - 1].append(ClassBlock(1664, 1, True, False, 256))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        s = self.model.features[:-1](x)
        # c = self.model.features[-1](s)
        output = []
        for i in range(len(self.combine)):
            output.append([])
            for k in range(self.combine[i]):
                if i == 1:
                    feat = s
                else:
                    # 进行水平划分
                    l = s.shape[-2] // self.combine[i]
                    feat = s[:, :, l * k:l * (k + 1)]
                output[-1].append(torch.squeeze(self.avgpool(feat)))
        return output
        # 进行分割
        # d = torch.squeeze(self.avgpool(s))
        # return [[torch.squeeze(self.avg(c))], [d[:,:,i] for i in range(self.part)]]
        # return [[d[:,:,i] for i in range(self.part)]]
        # x = self.model.features(x)
        # x = torch.squeeze(x)
        # if return_feat:
        #     return self.classifier(x, return_norm), x
        # else:
        #     return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048 + 1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, part=6):
        super(PCB, self).__init__()

        self.part = part  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

        for i in range(self.part):
            name = 'sn_classifier' + str(i)
            setattr(self, name, ClassBlock(2048, 1, True, False, 256))

    def forward(self, x, return_feat=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        if return_feat:
            return y, part
        else:
            return y


class PCB_NORM(nn.Module):
    def __init__(self, class_num, part=6):
        super(PCB_NORM, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, False, False, 256))

        for i in range(self.part):
            name = 'sn_classifier' + str(i)
            setattr(self, name, ClassBlock(2048, 1, False, False, 256))

        self.normBlock = NormBlock(2048, num_bottleneck=2048)

    def forward(self, x, return_feat=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        # x = torch.squeeze(x)
        # x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = self.normBlock(torch.squeeze(x[:, :, i]))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        if return_feat:
            return y, part
        else:
            return y


class PCB_ONE(nn.Module):
    def __init__(self, class_num, part=6):
        super(PCB_ONE, self).__init__()

        self.part = part  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, False, False, 256))

        for i in range(self.part):
            name = 'sn_classifier' + str(i)
            setattr(self, name, ClassBlock(2048, 1, False, False, 256))

    def forward(self, x, return_feat=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        # x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        if return_feat:
            return y, part
        else:
            return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


class PCB_ONE_test(nn.Module):
    def __init__(self, model):
        super(PCB_ONE_test, self).__init__()
        self.part = 1
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


class PCB_NORM_test(nn.Module):
    def __init__(self, model):
        super(PCB_NORM_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        x = self.normBlock(y)
        return y


if __name__ == '__main__':
    # # debug model structure
    # #net = ft_net(751)
    net = ft_net_dense_pcb(751)
    # #print(net)
    input = Variable(torch.FloatTensor(2, 3, 256, 128))
    output = net(input)
    print('net output size:')
    # print(output.shape)
