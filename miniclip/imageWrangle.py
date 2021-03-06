from PIL import Image
import numpy as np
from streamlit.logger import update_formatter
import torch
from matplotlib import cm



def min_max_norm(array):
    lim = [array.min(), array.max()]
    array = array - lim[0] 
    array.mul_(1 / (1.e-10+ (lim[1] - lim[0])))
    # array = torch.clamp(array, min=0, max=1)
    return array

def torch_to_rgba(img):
    img = min_max_norm(img)
    rgba_im = img.permute(1, 2, 0).cpu()
    if rgba_im.shape[2] == 3:
        rgba_im = torch.cat((rgba_im, torch.ones(*rgba_im.shape[:2], 1)), dim=2)
    assert rgba_im.shape[2] == 4
    return rgba_im


def numpy_to_image(img, size):
    """
    takes a [0..1] normalized rgba input and returns resized image as [0...255] rgba image
    """
    resized = Image.fromarray((img*255.).astype(np.uint8)).resize((size, size))
    return resized

def upscale_pytorch(img:np.array, size):
    torch_img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
    print(torch_img)
    upsampler = torch.nn.Upsample(size=size)    
    return upsampler(torch_img)[0].permute(1,2,0).cpu().numpy()


def heatmap(image:torch.Tensor, heatmap: torch.Tensor, size=None, alpha=.6):
    if not size:
        size = image.shape[1]
    # print(heatmap)
    # print(min_max_norm(heatmap))

    img = torch_to_rgba(image).numpy() # [0...1] rgba numpy "image"
    hm = cm.hot(min_max_norm(heatmap).numpy()) # [0...1] rgba numpy "image"

    # print(hm.shape, hm)
 #

    img = np.array(numpy_to_image(img,size))
    hm = np.array(numpy_to_image(hm, size))
    # hm = upscale_pytorch(hm, size)
    # print (hm) 

    return Image.fromarray((alpha * hm + (1-alpha)*img).astype(np.uint8))
    # return Image.fromarray(hm)