from os import listdir
from os.path import join
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

from util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.input_path = join(image_dir, "input")
        self.output_path = join(image_dir, "output")
        self.image_filenames = [x for x in listdir(self.output_path) if is_image_file(x)]

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.input_path, self.image_filenames[index]))
        input = self.transform(input)

        output= load_img(join(self.output_path, self.image_filenames[index]))
        output = self.transform(output)

        # inputt = output.resize((16, 16), Image.BICUBIC)
        # input = inputt.resize((128, 128), Image.BICUBIC)


        return input, output

    def __len__(self):
        return len(self.image_filenames)
