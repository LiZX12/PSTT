import matplotlib
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torchvision
import torch
from matplotlib import pyplot as plt
import numpy as np
import os.path
from model import ft_net, ft_net_pcb
from torchvision import models, transforms
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F

resnet_0 = None
resnet_1 = None
resnet_2 = None

def myimshows(imgs, titles=False, fname="1.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1][0::2,0::2], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1][0::2,0::2])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()  # [0]
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    # np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr


path = r"heat_map.jpg"
# bin_data = torchvision.io.read_file(path)
# img = torchvision.io.decode_image(bin_data) / 255
# img = img.unsqueeze(0)
# input_tensor = torchvision.transforms.functional.resize(img, [224, 224])
#
# input_tensors = torch.cat([input_tensor, input_tensor.flip(dims=(3,))], axis=0)
def heat_map(img_name,status=0):
    preprocess = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # size_upsample = (256,128)

    img_pil = Image.open(os.path.join("heat_map", img_name))
    img = cv2.imread(os.path.join("heat_map", img_name))
    img_tensor = preprocess(img_pil)
    input_tensors = Variable(img_tensor.unsqueeze(0))

    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip
    input_tensors = torch.cat([input_tensors, fliplr(input_tensors)], axis=0)
    global resnet_0,resnet_1,resnet_2
    if status == 0 and resnet_0 is not None:
        model = resnet_0
    elif status == 1 and resnet_1 is not None:
        model = resnet_1
    elif status == 2 and resnet_2 is not None:
        model = resnet_2
    else:
        # model = resnet50(pretrained=True)
        if status == 0:
            model = ft_net(751,forward_auto_class=True)
            resnet_0= model
            model_path = model_path_0
        elif status == 1:
            model = ft_net_pcb(751,forward_auto_class=True)
            resnet_1 = model
            model_path = model_path_1
        elif status == 2:
            model = ft_net_pcb(751,forward_auto_class=True)
            resnet_2 = model
            model_path = model_path_2
        # param_dict = torch.load(model_path)
        # for i in param_dict:
        #     model.state_dict()[i].copy_(param_dict[i])
    target_layers = [model.model.layer4[-1]]
    model.eval()

    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
        # targets = [ClassifierOutputTarget(3), ClassifierOutputTarget(1)]
        # if "fs" not in model_path:
        #     targets = [ClassifierOutputTarget(1),ClassifierOutputTarget(2),ClassifierOutputTarget(3),ClassifierOutputTarget(4),]
        # else:
        #     targets = None
        targets = None
        # targets = []
        # aug_smooth=True, eigen_smooth=True
        grayscale_cams = cam(input_tensor=input_tensors, targets=targets)
        grayscale_cams = np.average(grayscale_cams,axis=0).reshape(1,256,128)
        for grayscale_cam, tensor in zip(grayscale_cams, input_tensors):
            rgb_img = tensor2img(tensor)
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # myimshows([rgb_img, grayscale_cam, visualization], ["image", "cam", "image + cam"])
            # myimshows([visualization], [""])
        return visualization


def save_shows(imgs, fname="1.jpg"):
    lens = len(imgs)
    for i in range(1, lens + 1):
        matplotlib.image.imsave(fname, imgs[i - 1])

if __name__ == '__main__':

    # path = r"0014_c5_f0001026_00.jpg"
    # path = r"3.jpg"
    model_path_0 = r"H:\program\Spatial-Temporal-Re-identification-master\model\ft_market_resnet_r5\net_last.pth"
    model_path_1 = r"H:\program\Spatial-Temporal-Re-identification-master\model\ft_market_resnet_pcb_r5\net_last.pth"
    model_path_2 = r"H:\program\Spatial-Temporal-Re-identification-master\model\fs_market_resnet_pcb_r5_n\net_last.pth"
    path = r"H:\program\Spatial-Temporal-Re-identification-master\heat_map"
    for file_name in os.listdir(path):
        if "h_" in file_name:
            continue
        ff = heat_map(os.path.join(path, file_name), 0)
        save_shows([ff, ], fname=os.path.join(path, "h_" + file_name[:-4]+"_0_.jpg"))

        ff = heat_map(os.path.join(path, file_name), 1)
        save_shows([ff, ], fname=os.path.join(path, "h_" + file_name[:-4]+"_1_.jpg"))

        ff = heat_map(os.path.join(path, file_name), 2)
        save_shows([ff, ], fname=os.path.join(path, "h_" + file_name[:-4]+"_2_.jpg"))
    # h_1_1 = heat_map(r"1.jpg",model_path_0,"resnet")
    # h_1_2 = heat_map(r"1.jpg",model_path_1,"resnet_pcb")
    # h_1_3 = heat_map(r"1.jpg",model_path_2,"resnet_pcb")
    # # myimshows([h_1_1,h_1_2,h_1_3], ["IT","IT+HP","TDL+HP"])
    # myimshows([h_1_1,h_1_2,h_1_3], ["","",""])
