try:
    from petrel_client.client import Client
except:
    print('Petrel oss client is not installed')
import numpy as np
import io
from PIL import Image
import torch
import time
import pdb

def path_mapping(path):
    return path

class Image_OSS_IO():
    def __init__(self, conf_path = '~/petreloss.conf', client=None, path_mapping=path_mapping, disable_client=False):
        # we can explicitly disable_client, in this case, we need to make sure only file system path is used
        if client is None and not disable_client:
            self.client = Client(conf_path)
        else:
            self.client = client

        self.path_mapping = path_mapping
        return

    def read(self, path, return_numpy_array=True):
        # path is the path to an image file
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            img_bytes = self.client.get(path)
            image = Image.open(io.BytesIO(img_bytes))
        else:
            image = Image.open(path)
        if return_numpy_array:
            image = np.array(image)
        return image

    def write(self, image, path, format='png'):
        # image is an numpy arry of shape H,W,3, dtype unit 8
        # path is a normal path or ceph url, e.g., 's3://ZylyuBucket/folder/0999_label_0_array.png'
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            with io.BytesIO() as f:
                Image.fromarray(image).save(f, format=format)
                self.client.put(path, f.getvalue())
        else:
            Image.fromarray(image).save(path)

class Npz_OSS_IO():
    def __init__(self, conf_path = '~/petreloss.conf', client=None, path_mapping=path_mapping, disable_client=False):
        if client is None and not disable_client:
            self.client = Client(conf_path)
        else:
            self.client = client

        self.path_mapping = path_mapping
        return 

    def read(self, path, update_cache=False):
        # path is the path to an .npz file
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            # time1 = time.time()
            bbytes = io.BytesIO(self.client.get(path, update_cache=update_cache))
            # time2 = time.time()
            data = np.load(bbytes)
            # time3 = time.time()
            # print('read bytes time %.4f load numpy time %.4f total ceph read time %.4f' % (time2-time1, time3-time2, time3-time1))
        else:
            data = np.load(path)
        return data

    def write(self, path, *args, **kwds):
        # path is a normal path or ceph url, e.g., 's3://ZylyuBucket/folder/0999_label_0_array.png'
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            with io.BytesIO() as f:
                np.savez(f, *args, **kwds)
                self.client.put(path, f.getvalue())
        else:
            np.savez(path, *args, **kwds)

class Torch_OSS_IO():
    def __init__(self, conf_path = '~/petreloss.conf', client=None, path_mapping=path_mapping, disable_client=False):
        if client is None and not disable_client:
            self.client = Client(conf_path)
        else:
            self.client = client

        self.path_mapping = path_mapping
        return

    def read(self, path, map_location='cpu'):
        # path is the path to an .npz file
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            # checkpoint = torch.load(io.BytesIO(self.client.get(path)), map_location=map_location)
            with io.BytesIO(self.client.get(path)) as f:
                checkpoint = torch.load(f, map_location=map_location)
        else:
            checkpoint = torch.load(path, map_location=map_location)
        return checkpoint

    def write(self, path, checkpoint):
        # path is a normal path or ceph url, e.g., 's3://ZylyuBucket/folder/0999_label_0_array.png'
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            with io.BytesIO() as f:
                torch.save(checkpoint, f)
                self.client.put(path, f.getvalue())
        else:
            torch.save(checkpoint, path)

if __name__ == '__main__':
    image_oss = Image_OSS_IO()
    img_url = 's3://zylyu_datasets/celeba/img_align_celeba_64/test_18929.png'
    x = image_oss.read(img_url)

    save_url = 's3://ZylyuBucket/folder/0999_label_0_array.png'
    image_oss.write(x, save_url)

    y = image_oss.read(img_url)
    print(np.abs(x-y).max())

    pdb.set_trace()

    # npz_oss = Npz_OSS_IO()
    # read_path = 's3://zylyu_datasets/imagenet/val_64.npz'
    # data = npz_oss.read(read_path)
    # images = data['arr_0']
    # labels = data['arr_1']

    # save_path = 's3://zylyu_datasets/imagenet/val_64_copy.npz'
    # npz_oss.write(save_path, images, labels)

    # data = npz_oss.read(save_path)
    # images2 = data['arr_0']
    # labels2 = data['arr_1']

    # print(np.abs(images2-images).max())
    # print(np.abs(labels2-labels).max())

    # pdb.set_trace()


    torch_oss = Torch_OSS_IO()
    model_path = 's3://DDPM_Beat_GAN/scripts/models/64x64_diffusion.pt'
    model = torch_oss.read(model_path)

    model_save_path = 's3://DDPM_Beat_GAN/scripts/models/64x64_diffusion_copy.pt'
    torch_oss.write(model_save_path, model)
    mode2 = torch_oss.read(model_save_path)

    for key in model.keys():
        print(torch.abs(model[key]-mode2[key]).max())

    pdb.set_trace()