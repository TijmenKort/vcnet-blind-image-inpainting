import os
import copy
import json
import torch
import glog as log
import numpy as np

from tqdm import tqdm
from colorama import Fore

from PIL import Image, ImageDraw, ImageFont
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from datasets.graffiti_dataset.dataset import DatasetSample as sample_graffiti

from modeling.architecture import MPN, RIN, Discriminator
from utils.data_utils import linear_scaling, linear_unscaling
from utils.mask_utils import MaskGenerator, ConfidenceDrivenMaskLayer, COLORS
from metrics.psnr import PSNR
from metrics.ssim import SSIM


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

        self.transform = transforms.Compose([transforms.Resize(self.opt.DATASET.SIZE),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                             ])
        self.dataset = ImageFolder(root=self.opt.DATASET.ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)
        self.cont_dataset = ImageFolder(root=self.opt.DATASET.CONT_ROOT, transform=self.transform)
        self.cont_image_loader = data.DataLoader(dataset=self.cont_dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)
        self.mask_generator = MaskGenerator(self.opt.MASK)
        self.mask_smoother = ConfidenceDrivenMaskLayer(self.opt.MASK.GAUS_K_SIZE, self.opt.MASK.SIGMA)

        self.to_pil = transforms.ToPILImage()
        self.tensorize = transforms.ToTensor()

        self.mpn = MPN(base_n_channels=self.opt.MODEL.MPN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.rin = RIN(base_n_channels=self.opt.MODEL.RIN.NUM_CHANNELS, neck_n_channels=self.opt.MODEL.MPN.NECK_CHANNELS)
        self.discriminator = Discriminator(base_n_channels=self.opt.MODEL.D.NUM_CHANNELS)

        log.info("Checkpoints loading...")
        self.load_checkpoints(self.opt.TEST.WEIGHTS)

        self.mpn = self.mpn.cuda()
        self.rin = self.rin.cuda()
        self.discriminator = self.discriminator.cuda()
        self.mask_smoother = self.mask_smoother.cuda()

        self.PSNR = PSNR()
        self.SSIM = SSIM()
        self.BCE = torch.nn.BCELoss()

    def load_checkpoints(self, fname=None):
        if fname is None:
            fname = "{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, self.opt.TRAIN.START_STEP)
        checkpoints = torch.load(fname)
        self.mpn.load_state_dict(checkpoints["mpn"])
        self.rin.load_state_dict(checkpoints["rin"])
        self.discriminator.load_state_dict(checkpoints["D"])

    def eval(self):
        psnr_lst, ssim_lst, bce_lst = list(), list(), list()
        with torch.no_grad():
            for batch_idx, (imgs, _) in tqdm(enumerate(self.image_loader), ncols=100, desc="Training", bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
                imgs = linear_scaling(imgs.float().cuda())
                batch_size, channels, h, w = imgs.size()

                masks = torch.from_numpy(self.mask_generator.generate(h, w)).repeat([batch_size, 1, 1, 1]).float().cuda()
                smooth_masks = self.mask_smoother(1 - masks) + masks
                smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

                cont_imgs, _ = next(iter(self.cont_image_loader))
                cont_imgs = linear_scaling(cont_imgs.float().cuda())
                if cont_imgs.size(0) != imgs.size(0):
                    cont_imgs = cont_imgs[:imgs.size(0)]

                masked_imgs = cont_imgs * smooth_masks + imgs * (1. - smooth_masks)
                pred_masks, neck = self.mpn(masked_imgs)
                masked_imgs_embraced = masked_imgs * (1. - pred_masks) + torch.ones_like(masked_imgs) * pred_masks
                output = self.rin(masked_imgs_embraced, pred_masks, neck)
                output = torch.clamp(output, max=1., min=0.)

                bce = self.BCE(linear_unscaling(imgs), output).item()
                bce_lst.append(bce)

                ssim = self.SSIM(linear_unscaling(imgs), output)
                ssim_lst.append(ssim)

                for img, out in zip(linear_unscaling(imgs), output):
                    psnr = self.PSNR(255. * img.squeeze(0).cpu(), 255. * out.squeeze(0).cpu()).item()
                    psnr_lst.append(psnr)

        results = {"Dataset": self.opt.DATASET.NAME, "PSNR": np.mean(psnr_lst), "SSIM": np.mean(ssim_lst), "BCE": np.mean(bce_lst)}
        with open(os.path.join(self.opt.TEST.OUTPUT_DIR, "metrics.json"), "a+") as f:
            json.dump(results, f)

    def infer(self, mode=None, img_id=None, c_img_id=None, color=None, output_dir=None):
        mode = self.opt.TEST.MODE if mode is None else mode
        assert mode in range(1, 9)
        img_id = self.opt.TEST.IMG_ID if img_id is None else img_id
        assert img_id < len(self.image_loader.dataset)
        c_img_id = self.opt.TEST.C_IMG_ID if c_img_id is None else c_img_id
        assert c_img_id < len(self.cont_image_loader.dataset)
        color = self.opt.TEST.BRUSH_COLOR if color is None else color
        assert str(color).upper() in list(COLORS.keys())
        output_dir = os.path.join(self.opt.TEST.OUTPUT_DIR, self.ablation_map[mode]) if output_dir is None else output_dir
        # output_dir = os.path.join(self.opt.TEST.OUTPUT_DIR, str(mode), "{}_{}".format(img_id, c_img_id)) if output_dir is None else output_dir
        os.makedirs(output_dir, exist_ok=True)

        x, _ = self.image_loader.dataset.__getitem__(img_id)
        x = linear_scaling(x.unsqueeze(0).cuda())
        batch_size, channels, h, w = x.size()
        with torch.no_grad():
            masks = torch.cat([torch.from_numpy(self.mask_generator.generate(h, w)) for _ in range(batch_size)], dim=0).float().cuda()
            smooth_masks = self.mask_smoother(1 - masks) + masks
            smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)

            if mode == 1:  # contaminant image
                c_x, _ = self.cont_image_loader.dataset.__getitem__(c_img_id)
                c_x = c_x.unsqueeze(0).cuda()
            elif mode == 2:  # random brush strokes with noise
                c_x = torch.rand_like(x)
            elif mode == 3:  # random brush strokes with different colors
                brush = torch.tensor(list(COLORS["{}".format(color).upper()])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
                c_x = torch.ones_like(x) * brush
            elif mode == 4:  # real occlusions
                c_x = linear_unscaling(x)
            elif mode == 5:  # graffiti
                c_x, smooth_masks = self.put_graffiti()
            elif mode == 6:  # facades (i.e. resize whole c_img to 64x64, paste to a random location of img)
                c_x, smooth_masks = self.paste_facade(x, c_img_id)
                c_x = linear_unscaling(c_x)
            elif mode == 7:  # words (i.e. write text with particular font size and color)
                c_x, smooth_masks = self.put_text(x, color)
            else:  # face swap  (i.e. 64x64 center crop from c_img, paste to the center of img)
                c_x, smooth_masks = self.swap_faces(x, c_img_id)

            c_x = linear_scaling(c_x)
            masked_imgs = c_x * smooth_masks + x * (1. - smooth_masks)

            pred_masks, neck = self.mpn(masked_imgs)
            masked_imgs_embraced = masked_imgs * (1. - pred_masks) + torch.ones_like(masked_imgs) * pred_masks
            output = self.rin(masked_imgs_embraced, pred_masks, neck)

            vis_output = torch.cat([linear_unscaling(x).squeeze(0).cpu(),
                                    linear_unscaling(c_x).squeeze(0).cpu(),
                                    smooth_masks.squeeze(0).repeat(3, 1, 1).cpu(),
                                    linear_unscaling(masked_imgs).squeeze(0).cpu(),
                                    linear_unscaling(masked_imgs_embraced).squeeze(0).cpu(),
                                    pred_masks.squeeze(0).repeat(3, 1, 1).cpu(),
                                    torch.clamp(output.squeeze(0), max=1., min=0.).cpu()], dim=-1)
            self.to_pil(vis_output).save(os.path.join(output_dir, "output_{}_{}.png".format(img_id, c_img_id)))

            # self.to_pil(self.unnormalize(x).squeeze(0).cpu()).save(os.path.join(output_dir, "img.png"))
            # self.to_pil(smooth_masks.squeeze(0).cpu()).save(os.path.join(output_dir, "mask.png"))
            # self.to_pil(self.unnormalize(masked_imgs).squeeze(0).cpu()).save(os.path.join(output_dir, "input.png"))
            # self.to_pil(output.squeeze(0).cpu()).save(os.path.join(output_dir, "output.png"))
            # self.to_pil(pred_masks.squeeze(0).cpu()).save(os.path.join(output_dir, "output_mask.png"))

    def put_graffiti(self):
        resizer = transforms.Resize((self.opt.DATASET.SIZE, self.opt.DATASET.SIZE))
        sample = sample_graffiti(self.opt.TEST.GRAFFITI_PATH)
        masks = self.tensorize(resizer(Image.fromarray(sample.graffiti_mask))).unsqueeze(0).cuda()
        graffiti_img = self.tensorize(resizer(Image.fromarray(sample.image))).unsqueeze(0).cuda()
        c_x = graffiti_img * masks
        smooth_masks = self.mask_smoother(1 - masks) + masks
        smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
        return c_x, smooth_masks

    def paste_facade(self, x, c_img_id):
        resizer = transforms.Resize((self.opt.DATASET.SIZE // 8))
        facade, _ = self.cont_image_loader.dataset.__getitem__(c_img_id)
        facade = linear_scaling(self.tensorize(resizer(self.to_pil(facade))))
        coord_x, coord_y = np.random.randint(self.opt.DATASET.SIZE - self.opt.DATASET.SIZE // 8, size=(2,))
        x_scaled = copy.deepcopy(x)
        x_scaled[:, :, coord_x:coord_x + facade.size(1), coord_y:coord_y + facade.size(2)] = facade
        masks = torch.zeros((1, 1, self.opt.DATASET.SIZE, self.opt.DATASET.SIZE)).cuda()
        masks[:, :, coord_x:coord_x + facade.size(1), coord_y:coord_y + facade.size(2)] = torch.ones_like(facade[0])
        smooth_masks = self.mask_smoother(1 - masks) + masks
        smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
        return x_scaled, smooth_masks

    def put_text(self, x, color):
        text = self.opt.TEST.TEXT
        mask = self.to_pil(torch.zeros_like(x).squeeze(0).cpu())
        x_scaled = copy.deepcopy(x)
        x_scaled = self.to_pil(x_scaled.squeeze(0).cpu())
        d = ImageDraw.Draw(x_scaled)
        d_m = ImageDraw.Draw(mask)
        font = ImageFont.truetype(self.opt.TEST.FONT, self.opt.TEST.FONT_SIZE)
        font_w, font_h = d.textsize(text, font=font)
        c_w = (self.opt.DATASET.SIZE - font_w) // 2
        c_h = (self.opt.DATASET.SIZE - font_h) // 2
        d.text((c_w, c_h), text, font=font, fill=tuple([int(a * 255) for a in COLORS["{}".format(color).upper()]]))
        d_m.text((c_w, c_h), text, font=font, fill=(255, 255, 255))
        masks = self.tensorize(mask)[0].unsqueeze(0).unsqueeze(0).cuda()
        smooth_masks = self.mask_smoother(1 - masks) + masks
        smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
        return smooth_masks.repeat(1, 3, 1, 1), smooth_masks

    def swap_faces(self, x, c_img_id):
        center_cropper = transforms.CenterCrop((self.opt.DATASET.SIZE // 2, self.opt.DATASET.SIZE // 2))
        c_x, _ = self.cont_image_loader.dataset.__getitem__(c_img_id)
        crop = linear_scaling(self.tensorize(center_cropper(self.to_pil(linear_unscaling(c_x)))))
        coord_x = coord_y = (self.opt.DATASET.SIZE - self.opt.DATASET.SIZE // 2) // 2
        x_scaled = copy.deepcopy(linear_unscaling(x))
        x_scaled[:, :, coord_x:coord_x + self.opt.DATASET.SIZE // 2, coord_y:coord_y + self.opt.DATASET.SIZE // 2] = crop
        masks = torch.zeros((1, 1, self.opt.DATASET.SIZE, self.opt.DATASET.SIZE)).cuda()
        masks[:, :, coord_x:coord_x + self.opt.DATASET.SIZE // 2, coord_y:coord_y + self.opt.DATASET.SIZE // 2] = torch.ones_like(crop[0])
        smooth_masks = self.mask_smoother(1 - masks) + masks
        smooth_masks = torch.clamp(smooth_masks, min=0., max=1.)
        return x_scaled, smooth_masks
