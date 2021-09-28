import os
import copy
import json
import torch
import kornia
import glog as log
import numpy as np

from tqdm import tqdm
from colorama import Fore

from PIL import Image, ImageDraw, ImageFont
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from modeling.architecture import MPN, RIN, Discriminator
from utils.data_utils import linear_scaling, linear_unscaling
from utils.mask_utils import mask_loader, mask_binary
from metrics.psnr import PSNR
from metrics.ssim import SSIM
from losses.bce import WeightedBCELoss


class Tester:
    def __init__(self, cfg):
        self.opt = cfg

        self.ablation_map = {
            1: "cont_image",
            2: "random_noise",
            3: "random_color",
            4: "real_occl",
            5: "graffiti",
            6: "facades",
            7: "words",
            8: "face_swap"
        }

        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.JOINT.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)

        self.transform = transforms.Compose([transforms.Resize((self.opt.DATASET.SIZE_H, self.opt.DATASET.SIZE_W)),
                                             # transforms.RandomHorizontalFlip(),
                                            #  transforms.CenterCrop(self.opt.DATASET.SIZE),
                                             transforms.ToTensor(),
                                             # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                             ])
        self.dataset = ImageFolder(root=self.opt.DATASET.ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.opt.TEST.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)
        # self.cont_dataset = ImageFolder(root=self.opt.DATASET.CONT_ROOT, transform=self.transform)
        # self.cont_image_loader = data.DataLoader(dataset=self.cont_dataset, batch_size=self.opt.TEST.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)
        
        # ADDED MASK LOADER
        self.mask_loader = mask_loader
        
        self.to_pil = transforms.ToPILImage()
        self.tensorize = transforms.ToTensor()

        self.mpn = MPN(base_n_channels=self.opt.MODEL.MPN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.rin = RIN(base_n_channels=self.opt.MODEL.RIN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.discriminator = Discriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)

        log.info("Checkpoints loading...")
        self.load_checkpoints(self.opt.TEST.WEIGHTS)
        # self.remove_module_str = remove_module_str

        self.mpn = self.mpn.cuda()
        self.rin = self.rin.cuda()
        self.discriminator = self.discriminator.cuda()
        # self.mask_smoother = torch.nn.DataParallel(self.mask_smoother).cuda()

        self.PSNR = kornia.losses.psnr.PSNRLoss(max_val=1.)
        self.SSIM = SSIM()  # kornia's SSIM is buggy.
        self.BCE = WeightedBCELoss()

    def remove_module_str(self, d):
        """
        Removing ".module" from weights when DataParallel is used
        when saving module.
        """

        d = {k.replace("module.", ""): v for k, v in d.items()}

        return d

    def load_checkpoints(self, fname=None):
        """
        Loading checkpoints. 
        TODO: Probably getting errors because of DataParallel
        """
        print("\n\nFNAME: ", fname)

        if fname is None:
            fname = "{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, self.opt.TRAIN.START_STEP)
        checkpoints_mpn = torch.load(fname)
        checkpoints_rin = torch.load("./weights/VCNet_weights/VCNet_Places_300000step_4bs_0.0002lr_1gpu_17run.pth")
        self.mpn.load_state_dict(self.remove_module_str(checkpoints_mpn["mpn"]))
        self.rin.load_state_dict(checkpoints_rin["rin"])
        self.discriminator.load_state_dict(checkpoints_rin["D"])

    def eval(self):
        psnr_lst, ssim_lst, bce_lst = list(), list(), list()
        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(self.image_loader):
                imgs = linear_scaling(imgs.float().cuda())
                batch_size, channels, h, w = imgs.size()

                # load masks from directory
                masks = self.mask_loader(h, w, self.opt.DATASET.MASKS).repeat([batch_size, 1, 1, 1]).float().cuda()

                # cont_imgs, _ = next(iter(self.cont_image_loader))
                cont_imgs = masks.clone().cpu()
                cont_imgs = linear_scaling(cont_imgs.float())
                if cont_imgs.size(0) != imgs.size(0):
                    cont_imgs = cont_imgs[:imgs.size(0)]

                # create a binary tensor of masks
                masks = torch.stack([mask_binary(mask, h, w) for mask in masks])

                masked_imgs = (cont_imgs * masks.cpu() + imgs.cpu() * (1. - masks.cpu())).cuda()

                pred_masks, neck = self.mpn(masked_imgs)
                masked_imgs_embraced = masked_imgs * (1. - pred_masks)
                output = self.rin(masked_imgs_embraced, pred_masks, neck)
                output = torch.clamp(output, max=1., min=0.)

                unknown_pixel_ratio = torch.sum(masks.view(batch_size, -1), dim=1).mean() / (h * w)
                bce = self.BCE(torch.sigmoid(pred_masks), masks, torch.tensor([1 - unknown_pixel_ratio, unknown_pixel_ratio])).item()
                bce_lst.append(bce)

                ssim = self.SSIM(255. * linear_unscaling(imgs), 255. * output).item()
                ssim_lst.append(ssim)

                psnr = self.PSNR(linear_unscaling(imgs), output).item()
                psnr_lst.append(psnr)

                log.info("{}/{}\tBCE: {}\tSSIM: {}\tPSNR: {}".format(batch_idx, len(self.image_loader),
                    round(bce, 3),
                    round(ssim, 3),
                    round(psnr, 3)))

        results = {"Dataset": self.opt.DATASET.NAME, "PSNR": np.mean(psnr_lst), "SSIM": np.mean(ssim_lst), "BCE": np.mean(bce_lst)}
        with open(os.path.join(self.opt.TEST.OUTPUT_DIR, "metrics.json"), "a+") as f:
            json.dump(results, f)

    def infer(self, img_path, output_dir=None):

        with torch.no_grad():
            im = Image.open(img_path).convert("RGB")
            w_og, h_og = im.size
            input_im = im.resize((self.opt.DATASET.SIZE_W, self.opt.DATASET.SIZE_H))
            masked_imgs = linear_scaling(self.tensorize(input_im).unsqueeze(0).cuda())

            pred_masks, neck = self.mpn(masked_imgs)
            # pred_masks = pred_masks if mode != 8 else torch.clamp(pred_masks * smooth_masks, min=0., max=1.)
            masked_imgs_embraced = masked_imgs * (1. - pred_masks)
            output = self.rin(masked_imgs_embraced, pred_masks, neck)
            output = torch.clamp(output, max=1., min=0.)

            if output_dir is not None:
                # output_dir = os.path.join(output_dir, self.ablation_map[mode])
                os.makedirs(output_dir, exist_ok=True)

                # resize outputs
                masked_imgs = self.to_pil(linear_unscaling(masked_imgs[0]).cpu()).resize((w_og, h_og))
                # masked_imgs.save(
                #     os.path.join(output_dir, "masked_{}.png".format(img_path.split("/")[-1].split(".")[0]))
                # )
                pred_masks = self.to_pil(pred_masks[0].cpu()).resize((w_og, h_og))
                # pred_masks.save(
                #     os.path.join(output_dir, "mask_{}.png".format(img_path.split("/")[-1].split(".")[0]))
                # )
                output = self.to_pil(output.squeeze().cpu()).resize((w_og, h_og))
                
                
                # prepare input and output image for export
                im = self.tensorize(im)
                masked_imgs = self.tensorize(masked_imgs)
                pred_masks = self.tensorize(pred_masks).expand(3, -1, -1)
                output = self.tensorize(output)

                # masked_imgs = linear_unscaling(masked_imgs.cpu())
                # output = self.to_pil(output.squeeze().cpu())
                # output = self.tensorize(output.resize((w_og, h_og)))
                stack = torch.cat([im, masked_imgs, pred_masks, output], dim=1)

                # stack and export result images
                stack = self.to_pil(stack)
                stack.save(
                    os.path.join(output_dir, "out_{}".format(img_path.split("/")[-1]))
                )

            else:
                self.to_pil(output.squeeze().cpu()).show()
                self.to_pil(pred_masks.squeeze().cpu()).show()
                self.to_pil(linear_unscaling(masked_imgs).squeeze().cpu()).show()

    # def do_ablation(self, mode=None, img_id=None, c_img_id=None, color=None, output_dir=None):
    #     mode = self.opt.TEST.MODE if mode is None else mode
    #     assert mode in range(1, 9)
    #     img_id = self.opt.TEST.IMG_ID if img_id is None else img_id
    #     assert img_id < len(self.image_loader.dataset)
    #     c_img_id = self.opt.TEST.C_IMG_ID if c_img_id is None else c_img_id
    #     assert c_img_id < len(self.cont_image_loader.dataset)
    #     color = self.opt.TEST.BRUSH_COLOR if color is None else color
    #     assert str(color).upper() in list(COLORS.keys())
    #     output_dir = os.path.join(self.opt.TEST.OUTPUT_DIR, self.ablation_map[mode]) if output_dir is None else output_dir
    #     # output_dir = os.path.join(self.opt.TEST.OUTPUT_DIR, str(mode), "{}_{}".format(img_id, c_img_id)) if output_dir is None else output_dir
    #     os.makedirs(output_dir, exist_ok=True)

    #     x, _ = self.image_loader.dataset.__getitem__(img_id)
    #     x = linear_scaling(x.unsqueeze(0).cuda())
    #     batch_size, channels, h, w = x.size()
    #     with torch.no_grad():
    #         masks = torch.cat([torch.from_numpy(self.mask_generator.generate(h, w)) for _ in range(batch_size)], dim=0).float().cuda()
    #         smooth_masks = self.mask_smoother(1 - masks) + masks
    #         smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

    #         if mode == 1:  # contaminant image
    #             c_x, _ = self.cont_image_loader.dataset.__getitem__(c_img_id)
    #             c_x = c_x.unsqueeze(0).cuda()
    #         elif mode == 2:  # random brush strokes with noise
    #             c_x = torch.rand_like(x)
    #         elif mode == 3:  # random brush strokes with different colors
    #             brush = torch.tensor(list(COLORS["{}".format(color).upper()])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    #             c_x = torch.ones_like(x) * brush
    #         elif mode == 4:  # real occlusions
    #             c_x = linear_unscaling(x)
    #         elif mode == 5:  # graffiti
    #             c_x, smooth_masks = self.put_graffiti()
    #         elif mode == 6:  # facades (i.e. resize whole c_img to 64x64, paste to a random location of img)
    #             c_x, smooth_masks = self.paste_facade(x, c_img_id)
    #             c_x = linear_unscaling(c_x)
    #         elif mode == 7:  # words (i.e. write text with particular font size and color)
    #             c_x, smooth_masks = self.put_text(x, color)
    #         else:  # face swap  (i.e. 64x64 center crop from c_img, paste to the center of img)
    #             c_x, smooth_masks = self.swap_faces(x, c_img_id)

    #         c_x = linear_scaling(c_x)
    #         masked_imgs = c_x * smooth_masks + x * (1. - smooth_masks)

    #         pred_masks, neck = self.mpn(masked_imgs)
    #         masked_imgs_embraced = masked_imgs * (1. - pred_masks) + torch.ones_like(masked_imgs) * pred_masks
    #         output = self.rin(masked_imgs_embraced, pred_masks, neck)

    #         vis_output = torch.cat([linear_unscaling(x).squeeze(0).cpu(),
    #                                 linear_unscaling(c_x).squeeze(0).cpu(),
    #                                 smooth_masks.squeeze(0).repeat(3, 1, 1).cpu(),
    #                                 linear_unscaling(masked_imgs).squeeze(0).cpu(),
    #                                 linear_unscaling(masked_imgs_embraced).squeeze(0).cpu(),
    #                                 pred_masks.squeeze(0).repeat(3, 1, 1).cpu(),
    #                                 torch.clamp(output.squeeze(0), max=1., min=0.).cpu()], dim=-1)
    #         self.to_pil(vis_output).save(os.path.join(output_dir, "output_{}_{}.png".format(img_id, c_img_id)))

    #         # self.to_pil(self.unnormalize(x).squeeze(0).cpu()).save(os.path.join(output_dir, "img.png"))
    #         # self.to_pil(smooth_masks.squeeze(0).cpu()).save(os.path.join(output_dir, "mask.png"))
    #         # self.to_pil(self.unnormalize(masked_imgs).squeeze(0).cpu()).save(os.path.join(output_dir, "input.png"))
    #         # self.to_pil(output.squeeze(0).cpu()).save(os.path.join(output_dir, "output.png"))
    #         # self.to_pil(pred_masks.squeeze(0).cpu()).save(os.path.join(output_dir, "output_mask.png"))
