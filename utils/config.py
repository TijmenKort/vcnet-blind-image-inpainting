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
_C.TRAIN.NUM_TOTAL_STEP = 15000
_C.TRAIN.START_STEP = 0
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_STEPS_FOR_JOINT = 6600
_C.TRAIN.LOG_INTERVAL = 500
_C.TRAIN.SAVE_INTERVAL = 5000
_C.TRAIN.SAVE_DIR = "./weights/Compleet_8bs_"
_C.TRAIN.RESUME = False
_C.TRAIN.VISUALIZE_INTERVAL = 500
_C.TRAIN.TUNE = False

_C.MODEL = CN()
_C.MODEL.NAME = "VCNet_Compleet"
_C.MODEL.IS_TRAIN = True

_C.MODEL.MPN = CN()
_C.MODEL.MPN.NAME = "MaskPredictionNetwork"
_C.MODEL.MPN.NUM_CHANNELS = 64
_C.MODEL.MPN.NECK_CHANNELS = 128
_C.MODEL.MPN.LR = 1e-5  # 1e-3
_C.MODEL.MPN.BETAS = (0.9, 0.99)  # (0.5, 0.9)
_C.MODEL.MPN.LOSS_COEFF = 1.  # 2.

_C.MODEL.RIN = CN()
_C.MODEL.RIN.NAME = "RobustInpaintingNetwork"
_C.MODEL.RIN.NUM_CHANNELS = 32
_C.MODEL.RIN.NECK_CHANNELS = 128
_C.MODEL.RIN.LR = 1e-6  # 1e-4
_C.MODEL.RIN.BETAS = (0.9, 0.99)  # (0.5, 0.9)
_C.MODEL.RIN.LOSS_COEFF = 2.  # 1.
_C.MODEL.RIN.EMBRACE = True

_C.MODEL.D = CN()
_C.MODEL.D.NAME = "1-ChOutputDiscriminator"
_C.MODEL.D.NUM_CHANNELS = 64
_C.MODEL.D.LR = 1e-5 # 1e-3
_C.MODEL.D.BETAS = (0.9, 0.99)  # (0.5, 0.9)
_C.MODEL.D.NUM_CRITICS = 5

_C.MODEL.JOINT = CN()
_C.MODEL.JOINT.NAME = "JointNetwork"
_C.MODEL.JOINT.LR = 2e-5  # 2e-4
_C.MODEL.JOINT.BETAS = (0.9, 0.99)  # (0.5, 0.9)

_C.OPTIM = CN()
_C.OPTIM.GP = 10
_C.OPTIM.MASK = 1
_C.OPTIM.RECON = 1.4
_C.OPTIM.SEMANTIC = 1e-2 # 1e-4
_C.OPTIM.TEXTURE = 1e-1 # 1e-3
_C.OPTIM.ADVERSARIAL = 1e-3

_C.DATASET = CN()
_C.DATASET.NAME = "TVB"  # "FFHQ"
_C.DATASET.ROOT = "./datasets/data_256" #"./datasets/data_non_cartoon/data_256_train" # "./datasets/data_tvb_256[]"  # "./datasets/ffhq/images1024x1024" 
_C.DATASET.MASKS = "./datasets/masks_tv_256" #"./datasets/masks_tvb_256_large"
# _C.DATASET.SIZE = 512
_C.DATASET.SIZE_H = 256
_C.DATASET.SIZE_W = 256

_C.TEST = CN()
_C.TEST.OUTPUT_DIR = "./outputs"
_C.TEST.ABLATION = False
_C.TEST.WEIGHTS = "./weights/VCNet_TVB_100000step_4bs_0.0002lr_4gpu_16run/checkpoint-30000.pth"
_C.TEST.BATCH_SIZE = 64
_C.TEST.ITER = 4
_C.TEST.MODE = 1
_C.TEST.IMG_ID = 52
_C.TEST.C_IMG_ID = 38

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# provide a way to import the defaults as a global singleton:
cfg = _C  # users can `from config import cfg`
