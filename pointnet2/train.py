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

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor, broadcast_params
from diffusion_utils.diffusion import Diffusion

from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.point_upsample_module import point_upsample
from models.pointwise_net import get_pointwise_net


from shutil import copyfile
import copy

from mesh_evaluation import evaluate_per_rank, gather_generated_results 
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
from data_utils.ema import EMAHelper
from data_utils.points_sampling import sample_keypoints
import pickle
import pdb

def train(num_gpus, config_file, rank, group_name, dataset, root_directory, output_directory, 
          tensorboard_directory, ckpt_iter, n_epochs, epochs_per_ckpt, iters_per_logging,
          learning_rate, loss_type, conditioned_on_cloud,
          eval_start_epoch = 0, eval_per_ckpt = 1, task='generation', split_dataset_to_multi_gpus=False,
          diffusion_model=None, ema_rate=None):
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
    assert task in ['generation', 'keypoint_generation', 'keypoint_conditional_generation']
    # if task is generation, we generate point clouds from scatch, the ddpp models the distribution of point clouds
    # if task is keypoint_generation, we generate keypoints from scatch, the ddpp models the distribution of keypoints
    # if task in keypoint_conditional_generation, we generate point clouds conditioned on keypoints

    # generate experiment (local) path
    local_path = "T{}_betaT{}".format(diffusion_config["T"], diffusion_config["beta_T"])
    local_path = local_path + '_' + pointnet_config['model_name']
        
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
    
    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # load training data
    if split_dataset_to_multi_gpus:
        # we need to make sure that batch_size and eval_batch_size can be divided by number of gpus
        # batch_size and eval_batch_size will be divided by world_size in get_dataloader
        trainloader = get_dataloader(trainset_config, phase='train', rank=rank, world_size=num_gpus, 
                        append_samples_to_last_rank=True, shuffle_before_rank_split=True)
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
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # set ema
    if ema_rate is not None and rank == 0:
        assert isinstance(ema_rate, list)
        ema_helper_list = [EMAHelper(mu=rate) for rate in ema_rate]
        for ema_helper in ema_helper_list:
            ema_helper.register(net)


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
            if ema_rate is not None and rank==0:
                for i in range(len(ema_helper_list)):
                    ema_helper_list[i].load_state_dict(checkpoint['ema_state_list'][i])
                    ema_helper_list[i].to(torch.device('cuda'))
                print('Ema helper has been loaded', flush=True)

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
            points = data['points'].cuda() # of shape (npoints, 3), roughly in the range of -scale to scale
            normals = data['normals'].cuda() # of shape (npoints, 3), the normals are normalized
            label = data['label'].cuda()
            condition = None
            if task in ['keypoint_generation', 'keypoint_conditional_generation']:
                if trainset_config['keypoints_source'] == 'farthest_points_sampling':
                    keypoint, _ = sample_keypoints(points, K=trainset_config['num_keypoints'], 
                                    add_centroid=trainset_config.get('add_centroid_to_keypoints', True),
                                    random_subsample=trainset_config.get('random_sample_keypoints', False))
                else:
                    raise Exception('Only support farthest_points_sampling')
            if task in ['generation', 'keypoint_conditional_generation'] and trainset_config.get('include_normals', True):
                points = torch.cat([points, normals], dim=2)
            
            if task == 'keypoint_generation':
                X = keypoint
            elif task == 'generation':
                X = points
            elif task == 'keypoint_conditional_generation':
                X = points
                condition = keypoint
            
            
            # back-propagation
            optimizer.zero_grad()
            if diffusion_model is None:
                loss = training_loss(net, loss_function, X, diffusion_hyperparams,
                                label=label, condition=condition)
            else:
                loss_dict = diffusion_model.train_loss(net, X, label, normal_loss_type=standard_diffusion_config['normal_loss_type'],
                                                        loss_type=standard_diffusion_config['loss_type'])
                loss = loss_dict['training_loss']
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            if ema_rate is not None and rank==0:
                for ema_helper in ema_helper_list:
                    ema_helper.update(net)

            # output to log
            if n_iter % iters_per_logging == 0:
                if diffusion_model is None:
                    print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \ttime: {:.2f}s".format(
                        n_iter, reduced_loss, loss.item(), time.time()-log_start_time), flush=True)
                else:
                    print("iteration: {} \treduced loss: {:.4f} \tloss: {:.4f} \tcd_p: {:.4f} \tcd_feature_p: {:.4f} \tcd_t: {:.4f} \tcd_feature_t: {:.4f} \tf1: {:.4f} \ttime: {:.2f}s".format(
                        n_iter, reduced_loss, loss.item(), loss_dict['cd_p'].item(), loss_dict['cd_feature_p'].item(), 
                        loss_dict['cd_t'].item(), loss_dict['cd_feature_t'].item(), loss_dict['f1'].item(),
                        time.time()-log_start_time), flush=True)
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
                    checkpoint_states = {'iter': n_iter,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time()-time0)}
                    if not ema_rate is None:
                        checkpoint_states['ema_state_list'] = [ema_helper.state_dict() for ema_helper in ema_helper_list]
                    torch.save(checkpoint_states, os.path.join(output_directory, checkpoint_name))
                    print('model at iteration %s at epoch %d is saved' % (n_iter, epoch_number), flush=True)

                # evaluate the model at the checkpoint
                if n_iter >= eval_start_iter and num_ckpts % eval_per_ckpt==0:
                    save_dir = os.path.join(root_directory, local_path, 'eval_result')
                    ckpt_info = '_epoch_%s_iter_%d' % (str(epoch_number).zfill(4), n_iter)
                    print('\nBegin evaluting the saved checkpoint')
                    # num_samples_tested and eval_batch_size will be splited to multi ranks automatically
                    evaluate_and_gather(net, trainset_config, diffusion_hyperparams, save_dir, task, pointnet_config, 
                        ckpt_info, diffusion_model, rank, num_gpus)
                    if trainset_config.get('test_external_keypoint', False):
                        assert task == 'keypoint_conditional_generation'
                        evaluate_and_gather(net, trainset_config, diffusion_hyperparams, 
                            os.path.join(save_dir, 'external_keypoint_result'), task, pointnet_config, 
                            ckpt_info, diffusion_model, rank, num_gpus, test_external_keypoint=True, 
                            external_keypoint_file=trainset_config['external_keypoint_file'])
                    
                    # evaluate the ema models at the checkpoint
                    if not ema_rate is None:
                        net_ema = copy.deepcopy(net)
                        for i in range(len(ema_rate)):
                            ema_save_dir = os.path.join(save_dir, 'model_ema_%.5f' % ema_rate[i])
                            # net_ema.load_state_dict(ema_helper.state_dict())
                            if rank == 0:
                                ema_helper.ema(net_ema)
                            if num_gpus > 1:
                                torch.distributed.barrier()
                                broadcast_params(net_ema)
                            evaluate_and_gather(net_ema, trainset_config, diffusion_hyperparams, ema_save_dir, task, pointnet_config, 
                                                    ckpt_info, diffusion_model, rank, num_gpus)
                            if trainset_config.get('test_external_keypoint', False):
                                assert task == 'keypoint_conditional_generation'
                                evaluate_and_gather(net_ema, trainset_config, diffusion_hyperparams, 
                                    os.path.join(ema_save_dir, 'external_keypoint_result'), task, pointnet_config, 
                                    ckpt_info, diffusion_model, rank, num_gpus, test_external_keypoint=True, 
                                    external_keypoint_file=trainset_config['external_keypoint_file'])
                        del net_ema
                    print('Have finished evaluting the saved checkpoint\n')
            n_iter += 1


def evaluate_and_gather(net, trainset_config, diffusion_hyperparams, save_dir, task, pointnet_config, 
                        ckpt_info, diffusion_model, rank, num_gpus, test_external_keypoint=False, external_keypoint_file=None):
    os.makedirs(save_dir, exist_ok=True)
    evaluate_per_rank(net, trainset_config, diffusion_hyperparams, save_dir, task,
                                        point_feature_dim=pointnet_config['in_fea_dim'], 
                                        rank=rank, world_size=num_gpus, ckpt_info=ckpt_info,
                                        diffusion_model=diffusion_model,
                                        test_external_keypoint=test_external_keypoint, external_keypoint_file=external_keypoint_file)
    if num_gpus > 1:
        torch.distributed.barrier()
    if rank == 0 and num_gpus > 1:
        if task in ['generation', 'keypoint_conditional_generation']:
            num_points = trainset_config['npoints']
        elif task == 'keypoint_generation':
            num_points = trainset_config['num_keypoints']
        gather_generated_results(trainset_config['dataset'], save_dir, num_gpus, num_points=num_points, ckpt_info=ckpt_info)


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
    global diffusion_config
    diffusion_config = config["diffusion_config"]    # basic hyperparameters
    
    global trainset_config
    if train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet_psr_dataset':
        trainset_config = config['shapenet_psr_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    global diffusion_hyperparams 
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    if 'standard_diffusion_config' in config.keys():
        global standard_diffusion_config
        standard_diffusion_config = config['standard_diffusion_config']
        diffusion_model = Diffusion(standard_diffusion_config)
    else:
        diffusion_model = None

    print('Visible GPUs are', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
    num_gpus = torch.cuda.device_count()
    print('%d GPUs are available' % num_gpus, flush=True)
    if num_gpus > 1:
        assert args.group_name != ''
    else:
        assert args.rank == 0
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.config, args.rank, args.group_name, diffusion_model=diffusion_model, **train_config)
