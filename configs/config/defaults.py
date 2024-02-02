from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

_C.MODEL.SCHEDULER = "cosine"

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.PATCH_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 5
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# Sample Strategy
_C.MODEL.TRAIN_STRATEGY = '' # ['multiview', 'chunk']
_C.MODEL.SPATIAL = False
_C.MODEL.TEMPORAL = False
_C.MODEL.FREEZE = False
_C.MODEL.PYRAMID0_TYPE = ''
_C.MODEL.PYRAMID1_TYPE = ''
_C.MODEL.PYRAMID2_TYPE = ''
_C.MODEL.PYRAMID3_TYPE = ''
_C.MODEL.PYRAMID4_TYPE = ''
_C.MODEL.PYRAMID5_TYPE = ''
_C.MODEL.PYRAMID6_TYPE = ''
_C.MODEL.PYRAMID7_TYPE = ''
_C.MODEL.PYRAMID8_TYPE = ''
_C.MODEL.LAYER_COMBIN = 1
_C.MODEL.LAYER0_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER1_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER2_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER3_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER4_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER5_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER6_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER7_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER8_DIVISION_TYPE = 'NULL'
_C.MODEL.DIVERSITY = False
_C.MODEL.MASK_FLAG = False
_C.MODEL.ACCUMULATION_STEPS = 1
_C.MODEL.THREE_ARG = False
_C.MODEL.LR_TYPE = "FT"


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Random erasing
_C.INPUT.RE = True
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# random select p persons for each sample
_C.DATALOADER.P = 16
# random select k tracklets for each person
_C.DATALOADER.K = 8
# random select 8 images of each tracklet for test
_C.DATALOADER.NUM_TEST_IMAGES = 8
# random select 8 images of each tracklet for train
_C.DATALOADER.NUM_TRAIN_IMAGES = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.LR_DATA_RATIO = 1

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.FREZZ_EPOCHS = 0
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

_C.SOLVER.OUT_POINT_FLAG = False
_C.SOLVER.OUT_POINT_MODEL = ''
_C.SOLVER.OUT_POINT_EPOCH = 120
# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
#
_C.TEST.IMG_TEST_BATCH = 512
_C.TEST.TEST_BATCH = 32
_C.TEST.VIS = False
_C.TEST.EPOCH = 120
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""

_C.GA = CN()
_C.GA.GA_BUILD_TYPE = 0

_C.RIGHT_STREAM = CN()
_C.RIGHT_STREAM.HEAD_DISPATHER = r'{}'
_C.RIGHT_STREAM.HEAD_DISPATHER_WEIGHT = r'{}'
_C.RIGHT_STREAM.SPLIT_TYPE = 0
_C.RIGHT_STREAM.FORWARD_TYPE = 0
_C.RIGHT_STREAM.SPLIT_TYPE_INFO = None
_C.RIGHT_STREAM.STRIDE_SIZE = []
_C.RIGHT_STREAM.FUSION_TYPE = 0
_C.RIGHT_STREAM.SPLIT_CAT_DIM = 2
_C.RIGHT_STREAM.IMG_SIZE = [[256,128], [256,128]]
_C.RIGHT_STREAM.MERGE_LEN = 0
_C.RIGHT_STREAM.END_BLOCK_ATTEN_TYPE = 0
_C.RIGHT_STREAM.SPLIT_ATTION_TYPE_INDEX = 1
_C.RIGHT_STREAM.ALL_ROLL = 1
_C.RIGHT_STREAM.EMBED_TYPE = 0
_C.RIGHT_STREAM.HEAD_PATCH_SIZE = r'{}' # {"1":[8,11]}
_C.RIGHT_STREAM.HEAD_PATCH_SIZE_TYPE = 0
_C.RIGHT_STREAM.END_BLOCK_ATTEN_INDEX_LIST = []
_C.RIGHT_STREAM.TAA_FLAG = False
_C.RIGHT_STREAM.ADD_TR_LOSS = False
_C.RIGHT_STREAM.ATT_END_TYPE = 0
_C.RIGHT_STREAM.MID_BLOCK_ATTEN_SIZE = 0
_C.RIGHT_STREAM.MERLEN_LIST = []
_C.RIGHT_STREAM.PATCH_EMBED_TYPE = 0
_C.RIGHT_STREAM.KV_RELATIVE = False
_C.RIGHT_STREAM.QKV_SPLIT_TYPE = False
_C.RIGHT_STREAM.MID_SPLIT_ATTION_TYPE_INDEX = 1
_C.RIGHT_STREAM.AAA_LOSS_FLAG = False
_C.RIGHT_STREAM.LOSS_INFO =  r''
_C.RIGHT_STREAM.TRAN_TYPE = 0
_C.RIGHT_STREAM.FEAT_TYPE = 0
_C.RIGHT_STREAM.GA_EPOCH_NUM = 0
_C.RIGHT_STREAM.SAVE_EPOCH_LIST = []
_C.RIGHT_STREAM.EVA_EPOCH_EXCLUDE_LIST = []
_C.RIGHT_STREAM.KL_FLAG = False
_C.RIGHT_STREAM.KL_LOSS_WEIGHT = 1.0
_C.RIGHT_STREAM.SSNET_TYPE = 0
_C.RIGHT_STREAM.PSTA_TRIPLET = False
_C.RIGHT_STREAM.AAAA_FLAG = 0
_C.RIGHT_STREAM.BFT_FLAG_TYPE = 0
_C.RIGHT_STREAM.COM_POS = False