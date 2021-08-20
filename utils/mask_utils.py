import os
import torch
import random
from PIL import Image
from torchvision import transforms


def mask_loader(): # h, w):
    """
    Load single mask randomly from the mask dataset

    TODO:
    add a implementation for non-hardcoded dir path
    """

    # asign transform variable
    trans = transforms.ToTensor()
    # randomly choose mask path
    mask_path = random.choice(os.listdir('./datasets/masks_tvb_256_large'))

    # load image and tranform to tensor
    with Image.open("{}/{}".format(
        './datasets/masks_tvb_256_large', mask_path
    )) as mask:
        mask = trans(mask)

    # reshape to correct size
    mask = torch.reshape(mask, (1, 3, 256, 256))

    return mask


def mask_binary(mask): #, h, w):
    """
    Set three channel masks to binary.
    """

    # sum the color channels
    mask = torch.sum(mask, dim=(-3), keepdim=True)

    # replace every non-black pixel with white
    mask = torch.where(mask > 0, torch.full((256, 256), 1.).cuda(), mask)

    return mask
