import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from util import find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams
# from scheduler import QuantityScheduler
from shapenet_psr_dataloader.npz_dataset import ShapeNpzDataset

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from models.autoencoder import PointAutoencoder
from data_utils.points_sampling import sample_keypoints
from autoencoder_evaluation import evaluate_per_rank, quantitative_evaluate_per_rank

from shutil import copyfile
import copy

from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, autoencoder_read_config
import pickle
import pdb

def train(num_gpus, config_file, rank, group_name, dataset, root_directory, output_directory, 
          tensorboard_directory, ckpt_iter, n_epochs, epochs_per_ckpt, iters_per_logging,
          learning_rate, loss_type, conditioned_on_cloud,
          eval_start_epoch = 0, eval_per_ckpt = 1, task='generation', split_dataset_to_multi_gpus=False):
    """
    Train the PointNet2SemSegSSG model on the 3D dataset

    Parameters:
    num_gpus, rank, group_name:     parameters for distributed training
    config_file:                    path to the config file
    output_directory (str):         save model checkpoints to this path
    tensorboard_directory (str):    save tensorboard events to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    n_epochs (int):                 number of epochs to train
    epochs_per_ckpt (int):          number of epochs to save checkpoint
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """
    assert task in ['autoencode']
    # generate experiment (local) path
    local_path = pointnet_config['model_name']
        
    # Create tensorboard logger.
    if rank == 0:
        tb = SummaryWriter(os.path.join(root_directory, local_path, tensorboard_directory))

    # distributed running initialization
    if num_gpus > 1:
        dist_config.pop('CUDA_VISIBLE_DEVICES', None)
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared output_directory ready
    output_directory = os.path.join(root_directory, local_path, output_directory)
    if rank == 0:
        os.makedirs(output_directory, exist_ok=True)
        print("output directory is", output_directory, flush=True)

        config_file_copy_path = os.path.join(root_directory, local_path, os.path.split(config_file)[1])
        try:
            copyfile(config_file, config_file_copy_path)
        except:
            print('The two files are the same, no need to copy')
        print("Config file has been copied from %s to %s" % (config_file, config_file_copy_path), flush=True)

    # load training data
    if split_dataset_to_multi_gpus:
        # we need to make sure that batch_size and eval_batch_size can be divided by number of gpus
        # batch_size and eval_batch_size will be divided by world_size in get_dataloader
        trainloader = get_dataloader(trainset_config, phase='train', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=True, shuffle_before_rank_split=True)
        val_config = copy.deepcopy(trainset_config)
        val_config['repeat_dataset'] = 1
        val_config['batch_size'] = val_config['eval_batch_size']
        val_config['augmentation'] = False
        vis_loader = get_dataloader(val_config, phase='train', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=False, shuffle_before_rank_split=False,
                        random_subsample=True, num_samples=trainset_config['num_samples_tested'])
        test_trainloader = get_dataloader(val_config, phase='train', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=False, shuffle_before_rank_split=False,
                        random_subsample=False)
        test_valloader = get_dataloader(val_config, phase='val', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=False, shuffle_before_rank_split=False,
                        random_subsample=False)
    else:
        raise Exception('Must split_dataset_to_multi_gpus')
    
    print('Data loaded')
    
    # load network
    net = PointAutoencoder(encoder_config, decoder_config_list, 
                apply_kl_regularization=pointnet_config.get('apply_kl_regularization',False),
                kl_weight=pointnet_config.get('kl_weight', 0),
                feature_weight=pointnet_config.get('feature_weight', None)).cuda()
    net.train()
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint model
    time0 = time.time()
    _, num_ckpts = find_max_epoch(output_directory, 'pointnet_ckpt', return_num_ckpts=True)
    # num_ckpts is number of ckpts found in the output_directory
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory, 'pointnet_ckpt')
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, 'pointnet_ckpt_{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully', flush=True)
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.', flush=True)
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.', flush=True)

    print(net)

    # training
    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs) # number of total training steps 
    iters_per_ckpt = int(loader_len * epochs_per_ckpt) # save a ckpt every iters_per_ckpt steps
    n_iter = ckpt_iter + 1 # starting iter number
    eval_start_iter = eval_start_epoch *  loader_len - 1 
    # we start evaluating the trained model at least after eval_start_epoch steps
    

    log_start_time = time.time() # used to compute how much time is consumed between 2 printing log

    # n_iter from 0 to n_iters if we train the model from sratch
    while n_iter < n_iters + 1:
        # shuffle before split to multi ranks
        if split_dataset_to_multi_gpus and num_gpus > 1:
            trainloader = get_dataloader(trainset_config, phase='train', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=True, shuffle_before_rank_split=True)
            
        for data in trainloader: 
            epoch_number = int((n_iter+1)/loader_len)
            # load data
            X = data['points'].cuda() # of shape (B, npoints, 3), roughly in the range of -scale to scale
            normals = data['normals'].cuda() # of shape (B, npoints, 3), the normals are normalized
            normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
            label = data['label'].cuda()
            if trainset_config['keypoints_source'] == 'farthest_points_sampling':
                keypoints, _ = sample_keypoints(X, K=trainset_config['num_keypoints'], 
                                add_centroid=trainset_config.get('add_centroid_to_keypoints', True),
                                random_subsample=trainset_config.get('random_sample_keypoints', False))
                keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)
                if keypoint_noise_magnitude > 0:
                    keypoints = keypoints + keypoint_noise_magnitude * torch.randn_like(keypoints)
            X = torch.cat([X, normals], dim=2)

            l_xyz, loss_list = net(X, keypoints, ts=None, label=label, loss_type='cd_p')

            loss = 0
            for loss_dict in loss_list:
                loss = loss + loss_dict['training_loss'].mean()
            
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pdb.set_trace()

            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            # output to log
            if n_iter % iters_per_logging == 0:
                print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \ttime: {:.2f}s".format(
                    n_iter, reduced_loss, loss.item(), time.time()-log_start_time), flush=True)
                for key in loss_list[0].keys():
                    values = [loss_dict[key].mean().item() for loss_dict in loss_list]
                    print(key, values)
                log_start_time = time.time()
                if rank == 0:
                    tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                    tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
            
            # save checkpoint
            if n_iter > 0 and (n_iter+1) % iters_per_ckpt == 0:
                num_ckpts = num_ckpts + 1
                # save checkpoint
                if rank == 0:
                    checkpoint_name = 'pointnet_ckpt_{}.pkl'.format(n_iter)
                    torch.save({'iter': n_iter,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time()-time0)}, 
                                os.path.join(output_directory, checkpoint_name))
                    print('model at iteration %s at epoch %d is saved' % (n_iter, epoch_number), flush=True)

                # evaluate the model at the checkpoint
                if n_iter >= eval_start_iter and num_ckpts % eval_per_ckpt==0:
                    save_dir = os.path.join(root_directory, local_path, 'eval_result')
                    os.makedirs(save_dir, exist_ok=True)
                    evaluate_per_rank(net, vis_loader, dataset, save_dir, n_iter, epoch_number, 
                                        trainset_config, rank=rank, world_size=num_gpus)
                    quantitative_evaluate_per_rank(net, test_trainloader, dataset, os.path.join(save_dir, 'trainset_eval'), 
                        n_iter, epoch_number, trainset_config, rank=rank, world_size=num_gpus)
                    quantitative_evaluate_per_rank(net, test_valloader, dataset, os.path.join(save_dir, 'valset_eval'), 
                        n_iter, epoch_number, trainset_config, rank=rank, world_size=num_gpus)
                    if trainset_config.get('keypoint_noise_magnitude', 0) > 0:
                        trainset_config_copy = copy.deepcopy(trainset_config)
                        trainset_config_copy['keypoint_noise_magnitude'] = 0
                        quantitative_evaluate_per_rank(net, test_valloader, dataset, os.path.join(save_dir, 'valset_eval_keypoint_noise_0'), 
                        n_iter, epoch_number, trainset_config_copy, rank=rank, world_size=num_gpus)

            n_iter += 1


if __name__ == "__main__":
    # import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    # parser.add_argument('-d', '--device', type=int, default=0, help='cuda gpu index for training')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument('--dist_url', type=str, default='', help='distributed training url')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    config = read_json_file(args.config)
    config_file_path = os.path.split(args.config)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(config_file_path, config)
    
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))
    # global gen_config
    # gen_config = config["gen_config"]
    global train_config
    train_config = config["train_config"]        # training parameters
    global dist_config
    dist_config = config["dist_config"]         # to initialize distributed training
    if len(args.dist_url) > 0:
        dist_config['dist_url'] = args.dist_url
    global pointnet_config
    pointnet_config = config["pointnet_config"]     # to define pointnet
    # global diffusion_config
    # diffusion_config = config["diffusion_config"]    # basic hyperparameters
    
    global trainset_config
    if train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet_psr_dataset':
        trainset_config = config['shapenet_psr_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    # global diffusion_hyperparams 
    # diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    print('Visible GPUs are', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
    num_gpus = torch.cuda.device_count()
    print('%d GPUs are available' % num_gpus, flush=True)
    if num_gpus > 1:
        assert args.group_name != ''
    else:
        assert args.rank == 0
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.config, args.rank, args.group_name, **train_config)
