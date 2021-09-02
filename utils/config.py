from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPU = 4
_C.SYSTEM.NUM_WORKERS = 16

_C.WANDB = CN()
_C.WANDB.PROJECT_NAME = "thesis_pytorch"
_C.WANDB.ENTITY = "tijmenk_"
_C.WANDB.RUN = 16
_C.WANDB.LOG_DIR = ""
_C.WANDB.NUM_ROW = 0

_C.TRAIN = CN()
_C.TRAIN.NUM_TOTAL_STEP = 6000  # 40000 # 302000
_C.TRAIN.START_STEP = 0
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_STEPS_FOR_JOINT = 1800  # 2000
_C.TRAIN.LOG_INTERVAL = 1000
_C.TRAIN.SAVE_INTERVAL = 1000
_C.TRAIN.SAVE_DIR = "./weights"
_C.TRAIN.RESUME = False
_C.TRAIN.VISUALIZE_INTERVAL = 150
_C.TRAIN.TUNE = False

_C.MODEL = CN()
_C.MODEL.NAME = "VCNet"
_C.MODEL.IS_TRAIN = True

_C.MODEL.MPN = CN()
_C.MODEL.MPN.NAME = "MaskPredictionNetwork"
_C.MODEL.MPN.NUM_CHANNELS = 128  # 64
_C.MODEL.MPN.NECK_CHANNELS = 256  # 128
_C.MODEL.MPN.LR = 1e-5  # 1e-3
_C.MODEL.MPN.BETAS = (0.5, 0.9)  # (0.9, 0.999)
_C.MODEL.MPN.SCHEDULER = []
_C.MODEL.MPN.DECAY_RATE = 0.
_C.MODEL.MPN.LOSS_COEFF = 2.

_C.MODEL.RIN = CN()
_C.MODEL.RIN.NAME = "RobustInpaintingNetwork"
_C.MODEL.RIN.NUM_CHANNELS = 64  # 32
_C.MODEL.RIN.NECK_CHANNELS = 256  # 128
_C.MODEL.RIN.LR = 1e-4
_C.MODEL.RIN.BETAS = (0.5, 0.9)
_C.MODEL.RIN.SCHEDULER = []
_C.MODEL.RIN.DECAY_RATE = 0.
_C.MODEL.RIN.LOSS_COEFF = 1.
_C.MODEL.RIN.EMBRACE = True

_C.MODEL.D = CN()
_C.MODEL.D.NAME = "1-ChOutputDiscriminator"
_C.MODEL.D.NUM_CHANNELS = 64
_C.MODEL.D.LR = 1e-3
_C.MODEL.D.BETAS = (0.5, 0.9)
_C.MODEL.D.SCHEDULER = []
_C.MODEL.D.DECAY_RATE = 0.5
_C.MODEL.D.NUM_CRITICS = 5

_C.MODEL.JOINT = CN()
_C.MODEL.JOINT.NAME = "JointNetwork"
_C.MODEL.JOINT.LR = 2e-4
_C.MODEL.JOINT.BETAS = (0.5, 0.9)
_C.MODEL.JOINT.SCHEDULER = []
_C.MODEL.JOINT.DECAY_RATE = 0.5

_C.OPTIM = CN()
_C.OPTIM.GP = 10
_C.OPTIM.MASK = 1
_C.OPTIM.RECON = 1.4
_C.OPTIM.SEMANTIC = 1e-4
_C.OPTIM.TEXTURE = 1e-3
_C.OPTIM.ADVERSARIAL = 1e-3

_C.DATASET = CN()
_C.DATASET.NAME = "TVB"  # "FFHQ"
_C.DATASET.ROOT = "./datasets/data_tvb_512" # "./datasets/data_tvb_256[]"  # "./datasets/ffhq/images1024x1024" 
_C.DATASET.MASKS = "./datasets/masks_tvb_720" #"./datasets/masks_tvb_256_large"
_C.DATASET.SIZE = 512
_C.DATASET.SIZE_H = 256
_C.DATASET.SIZE_W = 455
_C.DATASET.MEAN = [0.5, 0.5, 0.5]
_C.DATASET.STD = [0.5, 0.5, 0.5]

_C.TEST = CN()
_C.TEST.OUTPUT_DIR = "./outputs"
_C.TEST.ABLATION = False
_C.TEST.WEIGHTS = ""
_C.TEST.BATCH_SIZE = 64
_C.TEST.ITER = 4
_C.TEST.MODE = 7
_C.TEST.IMG_ID = 52
_C.TEST.C_IMG_ID = 38

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# provide a way to import the defaults as a global singleton:
cfg = _C  # users can `from config import cfg`
