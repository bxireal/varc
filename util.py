import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision.transforms import Compose, ToTensor
#
#
# def transform():
#     return Compose([
#         ToTensor(),
#     ])



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# def load_img(filepath):
#     output = Image.open(filepath).convert('RGB')
#     inputt = output.resize((16, 16), Image.BICUBIC)
#     input = inputt.resize((128, 128), Image.BICUBIC)
#
#     # im = img[0,:,:]
#     # img = img.resize((620, 460), Image.BICUBIC)
#     return output, output

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # img = Image.open(filepath)
    # img = img.convert('RGB')
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  * 255.0
    image_numpy[image_numpy < 0] = 0
    image_numpy[image_numpy > 255.] = 255.
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


