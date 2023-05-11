from shapenet_psr_dataset import Shapes3dDataset
import time
import torch

if __name__ == '__main__':
    # dataset_folder='./shapenet_psr'
    # metadata_file = 'metadata.yaml'
    dataset_folder='s3://zylyu_datasets/shapenet_psr/'
    # split='train'
    split=None
    categories=None 

    dataset = Shapes3dDataset(dataset_folder, split, categories, rank=0, world_size=1, load_psr=True)
    print(len(dataset))

    # pdb.set_trace()
    # start = time.time()
    # for i in range(len(dataset)):
    #     data = dataset.__getitem__(i)
    #     print('Progress [%d|%d] % .3f time: %.4f s' % (i, len(dataset), i/len(dataset), time.time()-start), data['points'].shape, data['normals'].shape, data['psr'].shape, data['label'])
    #     start = time.time()

    while True:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)
        start = time.time()
        for i, data in enumerate(dataloader):
            print('Progress [%d|%d] % .3f time: %.8f s' % (i, len(dataloader), i/len(dataloader), time.time()-start), 
                    data['points'].shape, data['normals'].shape, data['psr'].shape, data['label'].shape, flush=True)
            start = time.time()