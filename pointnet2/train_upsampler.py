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
from util import training_loss #, calc_diffusion_hyperparams
# from scheduler import QuantityScheduler
from shapenet_psr_dataloader.npz_dataset import ShapeNpzDataset

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor


from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.point_upsample_module import point_upsample
from models.pointwise_net import get_pointwise_net
from models.autoencoder import PointAutoencoder
from data_utils.points_sampling import sample_keypoints
# from chamfer_loss import Chamfer_Loss
# from chamfer_loss_new import calc_cd

from dpsr_utils.dpsr import DPSR
from data_utils.mirror_partial import mirror_and_concat

from shutil import copyfile
import copy

# from completion_eval import evaluate, get_each_category_distance, gather_eval_result_of_different_iters, plot_train_and_val_eval_result
from dpsr_evaluation import evaluate_per_rank, network_output_to_dpsr_grid, visualize_per_rank
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
    assert task in ['upsample']
    # generate experiment (local) path
    # local_path = "T{}_betaT{}".format(diffusion_config["T"], diffusion_config["beta_T"])
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
    
    # # map diffusion hyperparameters to gpu
    # for key in diffusion_hyperparams:
    #     if key != "T":
    #         diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # load training data
    if split_dataset_to_multi_gpus:
        # we need to make sure that batch_size and eval_batch_size can be divided by number of gpus
        # batch_size and eval_batch_size will be divided by world_size in get_dataloader
        if use_autoencoder:
            noise_magnitude = trainset_config['augmentation']['noise_magnitude']
            trainset_config['augmentation']['noise_magnitude'] = 0
            # we should not add noise before autoencoder, instead, we should add noise after autoencoder
        else:
            noise_magnitude = None
        trainloader = get_dataloader(trainset_config, phase='train', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=True, shuffle_before_rank_split=True)
        val_loader = get_dataloader(trainset_config, phase='val', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=False, shuffle_before_rank_split=False)
        vis_loader = get_dataloader(trainset_config, phase='val', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=False, shuffle_before_rank_split=False,
                        random_subsample=True, num_samples=trainset_config['num_vis_samples'])
        if 'external_vis_dataset' in trainset_config.keys() and trainset_config['external_vis_dataset'] is not None:
            ext_vis_dataset = ShapeNpzDataset(trainset_config['external_vis_dataset'], scale=trainset_config['scale'], 
                                noise_magnitude=trainset_config["augmentation"]["noise_magnitude"], rank=rank, world_size=num_gpus)
            ext_vis_loader = torch.utils.data.DataLoader(ext_vis_dataset, batch_size=int(trainset_config['eval_batch_size'] / num_gpus), 
                                    shuffle=False, num_workers=trainset_config['num_workers'])
    else:
        raise Exception('Must split_dataset_to_multi_gpus')
    
    print('Data loaded')
    
    network_type = pointnet_config.get('network_type', 'pointnet++')
    assert network_type in ['pointnet++', 'pointwise_net', 'pvd']
    if network_type == 'pointnet++':
        net = PointNet2CloudCondition(pointnet_config).cuda()
    elif network_type == 'pointwise_net':
        net = get_pointwise_net(pointnet_config['network_args']).cuda()
    elif network_type == 'pvd':
        net = PVCNN2(**pointnet_config['network_args']).cuda()

    net.train()
    print(net)
    
    # build autoencoder and load the checkpoint
    if use_autoencoder:
        autoencoder = PointAutoencoder(encoder_config, decoder_config_list, 
                    apply_kl_regularization=autoencoder_config['pointnet_config'].get('apply_kl_regularization', False),
                    kl_weight=autoencoder_config['pointnet_config'].get('kl_weight', 0))
        autoencoder.load_state_dict( torch.load(config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'] )
        autoencoder.cuda()
        autoencoder.eval()
        print(autoencoder)
    else:
        autoencoder = None

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
    
    print('upsampler size:')
    print_size(net)
    if use_autoencoder:
        print('autoencoder size:')
        print_size(autoencoder)

    # build dpsr
    dpsr = DPSR(res=(dpsr_config['grid_res'], dpsr_config['grid_res'], dpsr_config['grid_res']), 
                sig=dpsr_config['psr_sigma']).cuda()

    # training
    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs) # number of total training steps 
    iters_per_ckpt = int(loader_len * epochs_per_ckpt) # save a ckpt every iters_per_ckpt steps
    n_iter = ckpt_iter + 1 # starting iter number
    eval_start_iter = eval_start_epoch *  loader_len - 1 
    # we start evaluating the trained model at least after eval_start_epoch steps
    

    loss_function = nn.MSELoss()

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
            if use_autoencoder:
                if trainset_config['keypoints_source'] == 'farthest_points_sampling':
                    keypoint, _ = sample_keypoints(X, K=trainset_config['num_keypoints'], add_centroid=True)
                else:
                    raise Exception('Only support farthest_points_sampling')
            normals = data['normals'].cuda() # of shape (B, npoints, 3), the normals are normalized
            normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
            label = data['label'].cuda()
            psr_gt = data['psr'].cuda()
            if trainset_config.get('include_normals', True):
                X = torch.cat([X, normals], dim=2) # X already contains noises if noise_manigtude > 0
            else:
                X = torch.cat([X, torch.zeros_like(X)], dim=2)
                # in this case, we assume the input point cloud do not have normals, the refinement network need to estimate normals
            condition = None

            if use_autoencoder:
                with torch.no_grad():
                    feature_at_keypoint = autoencoder.encode(X, keypoint, ts=None, label=label, sample_posterior=True)
                    X = autoencoder.decode(keypoint, feature_at_keypoint, ts=None, label=label)
                    if noise_magnitude > 0:
                        if dpsr_config.get('split_before_refine', False):
                            split_factor = dpsr_config['split_factor']
                            B,N,F = X.shape
                            noise = noise_magnitude * torch.randn(B,N,split_factor,F, dtype=X.dtype, device=X.device)
                            X = X.unsqueeze(2) + noise # B,N,split_factor,F
                            X = X.view(B, -1, F).contiguous()
                        else:
                            X = X + noise_magnitude * torch.randn_like(X)

            mirror_before_upsampling = dpsr_config.get('mirror_before_upsampling', False)
            only_original_points_split = dpsr_config.get('only_original_points_split', False)
            if mirror_before_upsampling:
                X = mirror_and_concat(X, axis=2, num_points=[], attach_label=True, permute=not only_original_points_split)[0]

            displacement = net(X, condition, ts=None, label=label) # (B, npoints, 6*point_upsample_factor)
            psr_grid, _, _ = network_output_to_dpsr_grid(X, displacement, dpsr, trainset_config['scale'], pointnet_config,
                        last_dim_as_indicator=mirror_before_upsampling, only_original_points_split=only_original_points_split)

            if dpsr_config['psr_tanh']:
                psr_grid = torch.tanh(psr_grid)
                psr_gt = torch.tanh(psr_gt)

            loss = loss_function(psr_grid, psr_gt)
            
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            # output to log
            if n_iter % iters_per_logging == 0:
                print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \ttime: {:.2f}s".format(
                    n_iter, reduced_loss, loss.item(), time.time()-log_start_time), flush=True)
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
                    ckpt_info = '_epoch_%s_iter_%d' % (str(epoch_number).zfill(4), n_iter)
                    print('\nBegin evaluting the saved checkpoint')
                    # num_samples_tested and eval_batch_size will be splited to multi ranks automatically
                    visualize_per_rank(net, dpsr, vis_loader, pointnet_config, dpsr_config, trainset_config, dataset, save_dir, n_iter, epoch_number,
                                    scale=trainset_config['scale'], rank=rank, world_size=num_gpus, 
                                    use_autoencoder=use_autoencoder, autoencoder=autoencoder, noise_magnitude=noise_magnitude)
                    evaluate_per_rank(net, dpsr, val_loader, pointnet_config, dpsr_config, trainset_config, dataset, save_dir, n_iter, epoch_number, 
                                    scale=trainset_config['scale'], rank=rank, world_size=num_gpus, 
                                    use_autoencoder=use_autoencoder, autoencoder=autoencoder, noise_magnitude=noise_magnitude)
                    if 'external_vis_dataset' in trainset_config.keys() and trainset_config['external_vis_dataset'] is not None:
                        ext_vis_save_dir = os.path.join(save_dir, 'external_dataset_vis_results')
                        visualize_per_rank(net, dpsr, ext_vis_loader, pointnet_config, dpsr_config, trainset_config, dataset, ext_vis_save_dir, 
                                        n_iter, epoch_number, scale=trainset_config['scale'], rank=rank, world_size=num_gpus, 
                                        use_autoencoder=use_autoencoder, autoencoder=autoencoder, noise_magnitude=noise_magnitude)
                    print('Have finished evaluting the saved checkpoint\n')
                    if num_gpus > 1:
                        torch.distributed.barrier()
                    # if rank == 0 and num_gpus > 1:
                    #     gather_generated_results(dataset, save_dir, num_gpus, num_points=trainset_config['npoints'], ckpt_info=ckpt_info)

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
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
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

    global dpsr_config
    dpsr_config = config['dpsr_config']

    # read autoencoder configs
    global use_autoencoder
    if 'autoencoder_config' in config.keys():
        use_autoencoder = True
        autoencoder_config_file = config['autoencoder_config']['config_file']
        global autoencoder_config
        autoencoder_config = read_json_file(autoencoder_config_file)
        autoencoder_config_file_path = os.path.split(autoencoder_config_file)[0]
        global encoder_config
        global decoder_config_list
        encoder_config, decoder_config_list = autoencoder_read_config(autoencoder_config_file_path, autoencoder_config)
        print('The autoencoder configuration is:')
        print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(autoencoder_config)), indent=4))
    else:
        use_autoencoder = False

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
