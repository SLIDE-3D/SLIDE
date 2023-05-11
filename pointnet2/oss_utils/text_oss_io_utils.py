try:
    from petrel_client.client import Client
except:
    print('Petrel oss client is not installed')
import numpy as np
import io
from PIL import Image
import torch
import os
import yaml
import pdb

def path_mapping(path):
    return path

class Text_OSS_IO():
    def __init__(self, conf_path = '~/petreloss.conf', client=None, path_mapping=path_mapping, disable_client=False):
        # we can explicitly disable_client, in this case, we need to make sure only file system path is used
        if client is None and not disable_client:
            self.client = Client(conf_path)
        else:
            self.client = client

        self.path_mapping = path_mapping
        return

    def read(self, path):
        # path is the path to a text file
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            content_bytes = self.client.get(path)
            if path.endswith('.yaml'):
                content = yaml.load(content_bytes, Loader=yaml.Loader)
            else:
                content = content_bytes.decode('utf-8')
        else:
            with open(path, 'r') as f:
                if path.endswith('.yaml'):
                    content = yaml.load(f, Loader=yaml.Loader)
                else:
                    content = f.read()
        
        return content

    # def write(self, image, path, format='png'):
    #     # image is an numpy arry of shape H,W,3, dtype unit 8
    #     # path is a normal path or ceph url, e.g., 's3://ZylyuBucket/folder/0999_label_0_array.png'
    #     path = self.path_mapping(path) 
    #     # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
    #     if path.startswith('s3://'):
    #         with io.BytesIO() as f:
    #             Image.fromarray(image).save(f, format=format)
    #             self.client.put(path, f.getvalue())
    #     else:
    #         Image.fromarray(image).save(path)

if __name__ == '__main__':
    conf_path = '~/petreloss.conf'
    client = Client(conf_path)

    # url = 's3://zylyu_datasets/shapenet_psr/02691156/train.lst'
    # url = 's3://zylyu_datasets/shapenet_psr/metadata.yaml'
    url = '../shapenet_psr_dataloader/metadata.yaml'
    # content = client.get(url)

    text_oss = Text_OSS_IO()
    content = text_oss.read(url)

    pdb.set_trace()