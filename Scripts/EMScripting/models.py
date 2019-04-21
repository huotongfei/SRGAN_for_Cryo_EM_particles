import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from scipy import ndimage
from PIL import Image
import pickle
from scipy import interpolate
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
import random

# ======================
# DataProcessing
# ======================
# TODO Data augmentor
def Mover(img_lr, dx , dy):

    # Tiling
    img = np.column_stack((img_lr, img_lr, img_lr))
    img = np.row_stack((img, img, img))

    # Translating Mirror image of img a 2D array
    m,n = img_lr.shape
    #x=n-dx
    #y=m-dy
    return img[m+dx:dx+(2*m),n+dy:dy+(2*n)]

# NOTE we changed the image dataset to our own
class ImageDataset():
    def __init__(self, lr_shape , hr_shape , lr_flist , hr_flist , lr_mean = 0.05, lr_std = 1.0, lr_min = -10.0, lr_max = 13.0, hr_mean = 0.56, hr_std = 1.0, hr_min = -1.0, hr_max = 7.5):

        # File list for hr and lr images
        self.hr_flist = hr_flist
        self.lr_flist = lr_flist

        # Mean and std for noisy lr 
        self.lr_mean = lr_mean
        self.lr_std = lr_std
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.hr_min = hr_min
        self.hr_max = hr_max

        # We provide noisy low resolution experimentally obtained image
        lr_height, lr_width = lr_shape
        # We provide high resolution motion corrected 2D image projected from relion 3D voxels
        hr_height, hr_width = hr_shape

        # We provide noisy low resolution experimentally obtained image
        self.lr_height, self.lr_width = lr_shape
        # We provide high resolution motion corrected 2D image projected from relion 3D voxels
        self.hr_height, self.hr_width = hr_shape


    def __getitem__(self, index):

        random.seed(index)
        # A. Data Augmentation parameter
        # 1. Translation
        # NOTE that lr is 64 64 hr is 256 256 i.e. 4 times. at most move 1/3 length
        direction_x = random.choice((-1, 1))
        direction_y = random.choice((-1, 1))
        lr_rand_translation_x = direction_x * random.randint(0,self.lr_width // 2.5) 
        lr_rand_translation_y = direction_y * random.randint(0,self.lr_height // 2.5)
        hr_rand_translation_x = int(lr_rand_translation_x * 4)
        hr_rand_translation_y = int(lr_rand_translation_y * 4)

        # 2. 90 degree rotation
        


        # B. LR image
        # 1. Resample to fixed size
        fn_lr = self.lr_flist[index]
        img_lr = pickle.load(open('%s' %(fn_lr),'rb'))
        img_lr = resize(img_lr, (self.lr_height, self.lr_width), mode='constant')
        #fig,ax = plt.subplots(1,1,figsize = (5,5))
        #ax.imshow(img_lr, cmap = "gray")
        #plt.show()


        # 2. Augment

        # Translate
        img_lr = Mover(img_lr, lr_rand_translation_x, lr_rand_translation_y)
        # Rotate


        #fig,ax = plt.subplots(1,1,figsize = (5,5))
        #ax.imshow(np.concatenate((img_lr, img_lr_m), axis = 1), cmap = "gray")
        #plt.show()


        # 3. Provide classical filters
        # Normalise
        lr_stack_1 = (img_lr - self.lr_mean )/ self.lr_std
        # Sobel
        #lr_stack_2 = ndimage.sobel(img_lr)
        # Min max
        #lr_stack_2 = (img_lr - self.lr_min )/ self.lr_max
        lr_stack_2 = gaussian_filter(img_lr, sigma=4)
        # Median
        lr_stack_3 = ndimage.median_filter(img_lr, size=(9,9))
        img_lr = np.stack([lr_stack_1, lr_stack_2, lr_stack_3])
        img_lr = torch.tensor(img_lr)

        # Show image
        #fig,ax = plt.subplots(1,1,figsize = (5,5))
        #ax.imshow(lr_stack_2, cmap = "gray")
        #plt.show()
        #sys.exit()





        # C. HR image
        fn_hr = self.hr_flist[index]
        img_hr = pickle.load(open('%s' %(fn_hr),'rb'))
        # 1. Resample to fixed size
        img_hr = resize(img_hr, (self.hr_height, self.hr_width), mode='constant')


        # 3. Classical filters
        img_hr = (img_hr - self.hr_min )/ self.hr_max


        # 4. Gaussian Mask on 
        blurred_img_hr = gaussian_filter(img_hr, sigma=30)
        # Mask all > 0.133
        img_hr = img_hr * (blurred_img_hr > 0.133)
        #random_background = np.random.random((self.hr_height, self.hr_width))*0.1
        #img_hr = img_hr + (random_background * (blurred_img_hr < 0.133) )
        #img_hr = img_hr + 0.01
        # 2. Augment
        hr_stack_1 = Mover(img_hr, hr_rand_translation_x, hr_rand_translation_y)



        #fig,ax = plt.subplots(1,1,figsize = (5,5))
        #s = ax.imshow(hr_stack_1, cmap = "gray")
        #fig.colorbar(s, ax=ax)
        #plt.show()
        #sys.exit()
 

        # Stack to 3D
        img_hr = np.stack([hr_stack_1, hr_stack_1, hr_stack_1])
        img_hr = torch.tensor(img_hr)
        
        return {"lr": img_lr, "hr": img_hr}

        


    def __len__(self):
        assert len(self.hr_flist) == len(self.lr_flist)
        return len(self.hr_flist)


# VGG19 as a pretrained extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            #nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Upscaling 64*64 to 256*256 NOTE in our case in_ and out_channels are 1 
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        # 20 15
        # 9 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())
        self.drop1 = nn.Dropout(p=0.01)

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),

                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Sigmoid()) #nn.ReLU())

    def forward(self, x):
        out1 = self.conv1(x)
        drop1 = self.drop1(out1)
        out = self.res_blocks(out1)
        #out = self.res_blocks(drop1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        # Outer residual skip connection
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
