import os
import glog as log
import argparse
from engine.trainer import Trainer
from engine.tester import Tester
from utils.config import get_cfg_defaults
from utils.config_test import get_cfg_test_defaults

parser = argparse.ArgumentParser()

parser.add_argument("--base_cfg", default="./wandb/run-20201023_213704-3o2q3c4r/config.yaml", metavar="FILE", help="path to config file")
parser.add_argument("--weights", "-w", default=None, help="weights for VCNet")
parser.add_argument("--dataset", "-d", default="TVB", help="dataset names: FFHQ, TVB")
parser.add_argument("--dataset_dir", default="./datasets/data_tvb_480", help="dataset directory: './datasets/data_tvb_480'")
parser.add_argument("--tune", action="store_true", help="true for starting tune for ablation studies")
parser.add_argument("--test", "-t", default="", help="true for testing phase")
parser.add_argument("--ablation", "-a", action="store_true", help="true for ablation studies")
parser.add_argument("--test_mode", default=1, help="test mode: 1: contaminant image,"
                                                   "2: random brush strokes with noise,"
                                                   "3: random brush strokes with colors,"
                                                   "4: real occlusions,"
                                                   "5: graffiti,"
                                                   "6: facades,"
                                                   "7: words,"
                                                   "8: face swap")

args = parser.parse_args()

if __name__ == '__main__':
    cfg = get_cfg_defaults()

    if args.test == 'test':
        cfg = get_cfg_test_defaults()

    # cfg.merge_from_file(args.base_cfg)
    # cfg.MODEL.IS_TRAIN = not args.test
    # cfg.DATASET.NAME = args.dataset
    # cfg.DATASET.ROOT = args.dataset_dir
    # cfg.TEST.WEIGHTS = args.weights
    # cfg.TEST.ABLATION = args.ablation
    # cfg.TEST.MODE = args.test_mode
    # cfg.freeze()
    print(cfg)

    if cfg.MODEL.IS_TRAIN:
        trainer = Trainer(cfg)
        trainer.run()
    else:
        tester = Tester(cfg)
        if cfg.TEST.ABLATION:
            for i_id in list(range(250, 500)):
                for c_i_id in list(range(185, 375)):
                    for mode in list(range(1, 9)):
                        tester.do_ablation(mode=mode, img_id=i_id, c_img_id=c_i_id)
                        log.info("I: {}, C: {}, Mode:{}".format(i_id, c_i_id, mode))
        else:
            # qualitative
            for img_path in os.listdir('datasets/test_data'):

                img_path = f"datasets/test_data/{img_path}"

                tester.infer(img_path, output_dir="outputs")
            
            # quantitative
            # tester.eval()
