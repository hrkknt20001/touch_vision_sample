# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

#from engine import train_one_epoch, evaluate
#import utils
#import transforms as T

import pandas as pd

import matplotlib
import matplotlib.pylab as plt

import randomcolor
import glob

def main():
    rand_color = randomcolor.RandomColor()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = pd.read_pickle("./model_epoch_15.pkl")

    # move model to the right device
    model.to(device)

    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    image_list = glob.glob('./HouseFloorPlan_test/PNGImages/*.png')
    for filepath in image_list:
        img = Image.open(filepath).convert("RGB")
        image_tensor = torchvision.transforms.functional.to_tensor(img)
        image_tensor = image_tensor.to(device)
        output = model([image_tensor])[0]

        mask_image = np.zeros((img.height, img.width, 3), np.uint8)
        for score, mask in zip(output['scores'], output['masks']):
            if score < 0.8:
                continue

            mask_np = mask.to('cpu').detach().numpy().copy().transpose(1,2,0)
            x = (mask_np > 0.0075).sum(axis=2)==1
            cr = rand_color.generate()
            mask_image[x] = [
                int(cr[0][1:3],16), int(cr[0][3:5],16), int(cr[0][5:7],16), 
            ]

        pil_img = Image.fromarray(mask_image)
        pil_img.save(
            os.path.join(
                './HouseFloorPlan_test/PredictImages',
                os.path.basename(filepath)
            )
        )

        bg = Image.new("RGBA", (img.width, img.height), (255, 255, 255, 0))

        img_1 = img.convert('RGBA')

        img_2 = pil_img.convert('RGBA')
        img_2.putalpha(128)        

        bg.paste(img_1,img_1)
        bg.paste(img_2,img_2)
        bg.save(
            os.path.join(
                './HouseFloorPlan_test/composite',
                os.path.basename(filepath)
            )
        )

    print("That's it!")
    
if __name__ == "__main__":
    main()
