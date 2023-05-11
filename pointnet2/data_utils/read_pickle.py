import pickle
import numpy as np
import argparse
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../exps/exp_shapenet_psr_generation/refine_and_upsampling_exps/T1000_betaT0.02_shapenet_dpsr_upsample_10_noise_0.02_symmetry/eval_result/shapenet_psr_dpsr_eval_result.pkl', help='the pickle file that contains the eval results')
    args = parser.parse_args()
    
    file_name = args.file
    handle = open(file_name, 'rb')
    data = pickle.load(handle)

    loss_name = 'dpsr_grid_L2_loss'
    y = np.array(data[loss_name])
    idx = np.argmin(y)

    print('The lowest %s is at' % loss_name)
    for key in data.keys():
        print(key, data[key][idx])
    pdb.set_trace()