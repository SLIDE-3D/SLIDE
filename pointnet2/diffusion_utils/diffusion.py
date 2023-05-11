import numpy as np
import torch
import argparse
import json
import pdb

import sys
sys.path.append('../')
from metrics_point_cloud.chamfer_and_f1 import calc_cd


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a).float().to(device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,)+(1,)*(len(x_shape)-1))
    return out


def diffusion_step(x, t, *,
                   noise=None,
                   sqrt_alphas,
                   sqrt_one_minus_alphas):
    """
    Sample from q(x_t | x_{t-1}) (eq. (2))
    """
    if noise is None:
        noise = torch.randn_like(x)
    assert noise.shape == x.shape
    return (
        extract(sqrt_alphas, t, x.shape) * x +
        extract(sqrt_one_minus_alphas, t, x.shape) * noise
    )


def denoising_step(x, t, model, logvar, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
            posterior_mean_coef1, posterior_mean_coef2, return_pred_xstart=False,
            data_clamp_range=1, label=None,
            local_resampling=False, complete_x0=None, keypoint_mask=None):
    """
    Sample from p(x_{t-1} | x_t)
    """
    # instead of using eq. (11) directly, follow original implementation which,
    # equivalently, predicts x_0 and uses it to compute mean of the posterior
    # 1. predict eps via model
    model_output = model(x, ts=t, label=label)
    # 2. predict clipped x_0
    # (follows from x_t=sqrt_alpha_cumprod*x_0 + sqrt_one_minus_alpha*eps)
    pred_xstart = (extract(sqrt_recip_alphas_cumprod, t, x.shape)*x -
                   extract(sqrt_recipm1_alphas_cumprod, t, x.shape)*model_output)
        
    if data_clamp_range > 0:
        pred_xstart = torch.clamp(pred_xstart, -data_clamp_range, data_clamp_range)
    if local_resampling:
        for _ in range(len(complete_x0.shape)-2):
            keypoint_mask = keypoint_mask.unsqueeze(2)
        pred_xstart = pred_xstart * keypoint_mask + complete_x0 * (1-keypoint_mask)

    # 3. compute mean of q(x_{t-1} | x_t, x_0) (eq. (6))
    mean = (extract(posterior_mean_coef1, t, x.shape)*pred_xstart +
            extract(posterior_mean_coef2, t, x.shape)*x)

    logvar = extract(logvar, t, x.shape)

    # sample - return mean for t==0
    noise = torch.randn_like(x)
    mask = 1-(t==0).float()
    mask = mask.reshape((x.shape[0],)+(1,)*(len(x.shape)-1))
    sample = mean + mask*torch.exp(0.5*logvar)*noise
    sample = sample.float()
    if return_pred_xstart:
        return sample, pred_xstart
    return sample


class Diffusion(object):
    def __init__(self, diffusion_config, device=None):
        self.init_diffusion_parameters(diffusion_config)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def train_loss(self, net, x, label, normal_loss_type='cos', loss_type='cd_p'):
        B,N,F = x.shape
        diffusion_steps = torch.randint(self.num_timesteps, size=(B,)).to(self.device)
        alpha_bar = extract(self.alphas_cumprod, diffusion_steps, x.shape)
        z = torch.randn_like(x)
        transformed_X = torch.sqrt(alpha_bar) * x + torch.sqrt(1-alpha_bar) * z
        model_output = net(transformed_X, ts=diffusion_steps, label=label)
        model_output = model_output * self.model_output_scale_factor
        # predict clipped x_0
        # (follows from x_t=sqrt_alpha_cumprod*x_0 + sqrt_one_minus_alpha*eps)
        pred_xstart = (extract(self.sqrt_recip_alphas_cumprod, diffusion_steps, x.shape)*transformed_X.detach() -
                        extract(self.sqrt_recipm1_alphas_cumprod, diffusion_steps, x.shape)*model_output)
        if self.scale_loss_terms:
            pred_xstart = pred_xstart / extract(self.sqrt_recipm1_alphas_cumprod, diffusion_steps, x.shape)
            x = x / extract(self.sqrt_recipm1_alphas_cumprod, diffusion_steps, x.shape)
        loss_dict = calc_cd(pred_xstart, x.detach(), calc_f1=True, f1_threshold=0.0001, 
                        normal_loss_type=normal_loss_type)
        loss_dict['x0_mse'] = torch.sum((pred_xstart - x)**2, dim=2).mean(dim=1) # of shape (B)
        loss_dict['epsilon_mse'] = torch.sum((model_output - z)**2, dim=2).mean(dim=1) # of shape (B)
        # pdb.set_trace()

        # if self.scale_loss_terms:
        #     loss_dict['x0_mse'] = loss_dict['x0_mse'] / (extract(self.sqrt_recipm1_alphas_cumprod, diffusion_steps, loss_dict['x0_mse'].shape))**2

        if not 'cd_feature_p' in loss_dict.keys():
            loss_dict['cd_feature_p'] = torch.tensor([0]).to(loss_dict['cd_p'].device).float()
        if not 'cd_feature_t' in loss_dict.keys():
            loss_dict['cd_feature_t'] = torch.tensor([0]).to(loss_dict['cd_t'].device).float()

        if loss_type == 'cd_p':
            loss = loss_dict['cd_p'] + loss_dict['cd_feature_p']
        elif loss_type == 'cd_t':
            loss = loss_dict['cd_t'] + loss_dict['cd_feature_t']
        elif loss_type == 'x0_mse':
            loss = loss_dict['x0_mse']
        elif loss_type == 'epsilon_mse':
            loss = loss_dict['epsilon_mse']
        elif loss_type == 'mixed_cd_p_epsilon_mse':
            small_ts = (diffusion_steps < self.t_trunction).float()
            loss = small_ts * (loss_dict['cd_p'] + loss_dict['cd_feature_p']) + (1-small_ts) * loss_dict['epsilon_mse']
        elif loss_type == 'mixed_cd_t_epsilon_mse':
            small_ts = (diffusion_steps < self.t_trunction).float()
            loss = small_ts * (loss_dict['cd_t'] + loss_dict['cd_feature_t']) + (1-small_ts) * loss_dict['epsilon_mse']
        else:
            raise Exception('loss type %s is not supported yet' % standard_diffusion_config['loss_type'])
        loss_dict['training_loss'] = loss

        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key].mean()
            
        return loss_dict


    def init_diffusion_parameters(self, config):
        self.model_var_type = config.get("model_var_type", "fixedsmall")
        betas=get_beta_schedule(
            beta_schedule=config['beta_schedule'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            num_diffusion_timesteps=config['num_diffusion_timesteps']
        )
        self.num_timesteps = betas.shape[0]
        self.data_clamp_range = config['data_clamp_range']
        self.model_output_scale_factor = config['model_output_scale_factor']
        self.scale_loss_terms = config.get('scale_loss_terms', False)
        print('Scale loss terms is', self.scale_loss_terms, flush=True)

        alphas = 1.0-betas
        alphas_cumprod = np.cumprod(alphas, axis=0) # \bar{alpha}_t
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1]) # \bar{alpha}_t-1
        posterior_variance = betas*(1.0-alphas_cumprod_prev) / (1.0-alphas_cumprod) # \tilde{beta}_t
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        # they are all numpy arrays of shape (T,)
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_one_minus_alphas = np.sqrt(1. - alphas)

        if config['loss_type'] in ['mixed_cd_p_epsilon_mse', 'mixed_cd_t_epsilon_mse']:
            if 't_trunction' in config.keys():
                self.t_trunction = config['t_trunction']
                print('t trunction is %d' % (self.t_trunction))
            else:
                self.xt_coefficient_trunction = config['xt_coefficient_trunction']
                self.t_trunction = (sqrt_recip_alphas_cumprod < self.xt_coefficient_trunction).sum()
                print('xt_coefficient_trunction is %.4f, and t trunction is %d' % (self.xt_coefficient_trunction, self.t_trunction))
            # we have sqrt_recip_alphas_cumprod[t_trunction-1] < xt_coefficient_trunction < sqrt_recip_alphas_cumprod[t_trunction]

        if self.model_var_type == "fixedlarge":
            # this may be experimentally good
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
        elif self.model_var_type == 'fixedsmall':
            # this is the standard formulation, it is the one used in the pdr paradigm
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
        else:
            raise Exception('the variance type %s is not supported' % self.model_var_type)


    def denoise(self, n, model, shape, label=None, n_steps=None, x=None, curr_step=None,
                callback=lambda x, i, x0=None: None):
        # n is the batchsize
        with torch.no_grad():
            if curr_step is None:
                curr_step = self.num_timesteps

            assert curr_step > 0, curr_step

            if n_steps is None or curr_step-n_steps < 0:
                n_steps = curr_step

            if x is None:
                assert curr_step == self.num_timesteps, curr_step
                # start the chain with x_T from normal distribution
                x = torch.randn(n, *shape)
                x = x.to(self.device)

            # when start from scratch
            # curr_step = num_timesteps = n_steps
            # i range from 0 to num_timesteps-1
            # reversed i range from num_timesteps-1 to 0
            for i in reversed(range(curr_step-n_steps, curr_step)):
                t = (torch.ones(n)*i).to(self.device)
                x, x0 = denoising_step(x,
                                       t=t,
                                       model=model,
                                       logvar=self.logvar,
                                       sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                       sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                       posterior_mean_coef1=self.posterior_mean_coef1,
                                       posterior_mean_coef2=self.posterior_mean_coef2,
                                       return_pred_xstart=True,
                                       data_clamp_range=self.data_clamp_range,
                                       label=label)
                callback(x, i, x0=x0)

            return x


    def diffuse(self, n, n_steps=None, x=None, curr_step=None,
                progress_bar=lambda i, total=None: i,
                callback=lambda x, i: None):
        with torch.no_grad():
            if curr_step is None:
                curr_step = 0

            assert curr_step < self.num_timesteps, curr_step

            if n_steps is None or curr_step+n_steps > self.num_timesteps:
                n_steps = self.num_timesteps-curr_step

            assert x is not None

            for i in progress_bar(range(curr_step, curr_step+n_steps), total=n_steps):
                t = (torch.ones(n)*i).to(self.device)
                x = diffusion_step(x,
                                   t=t,
                                   sqrt_alphas=self.sqrt_alphas,
                                   sqrt_one_minus_alphas=self.sqrt_one_minus_alphas)
                callback(x, i+1)

            return x

    def diffuse_t_steps(self, x0, t):
        # x is a torch tensor of shape (B,*shape)
        # t is an interger range from 0 to T-1
        alpha_bar = self.alphas_cumprod[t]
        xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1-alpha_bar) * torch.randn_like(x0)
        return xt

class LatentDiffusion(Diffusion):
    def __init__(self, diffusion_config, autoencoder, device=None):
        self.init_diffusion_parameters(diffusion_config)
        self.keypoint_position_loss_weight = diffusion_config.get('keypoint_position_loss_weight', 1.0)
        self.feature_loss_weight = diffusion_config.get('feature_loss_weight', 1.0)
        self.keypoint_conditional = diffusion_config.get('keypoint_conditional', False)
        if self.keypoint_conditional:
            # if keypoint_conditional, the latent ddpm only generates features at given key points
            # do not need to generate keypoints themselves 
            self.keypoint_position_loss_weight = 0
        print('keypoint_position_loss_weight', self.keypoint_position_loss_weight)
        print('feature_loss_weight', self.feature_loss_weight)
        print('keypoint_conditional', self.keypoint_conditional)

        self.autoencoder = autoencoder
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
    
    def encode(self, x, keypoint, label):
        # pdb.set_trace()
        # x is of shape B,N,C
        # keypoint is of shape B,N2,C2
        # label is of shape B
        # feature_at_keypoint is of shape B,N2,C3
        feature_at_keypoint = self.autoencoder.encode(x, keypoint, ts=None, label=label, sample_posterior=True)
        feature_and_keypoint = torch.cat([keypoint, feature_at_keypoint], dim=2) # shape B,N2,C2+C3
        return feature_and_keypoint
    
    def decode(self, latent, keypoint_dim, label):
        # latent is of shape B,N2,C2+C3
        keypoint = latent[:,:,0:keypoint_dim]
        feature_at_keypoint = latent[:,:,keypoint_dim:]
        reconstructed_pointcloud = self.autoencoder.decode(keypoint, feature_at_keypoint, ts=None, label=label)
        return reconstructed_pointcloud


    def train_loss(self, net, x, keypoint, label):
        with torch.no_grad():
            x = self.encode(x, keypoint, label)
        B,N,F = x.shape
        keypoint_dim = keypoint.shape[2]
        diffusion_steps = torch.randint(self.num_timesteps, size=(B,)).to(self.device)
        # diffusion_steps = torch.randint(10, size=(B,)).to(self.device)
        alpha_bar = extract(self.alphas_cumprod, diffusion_steps, x.shape)
        z = torch.randn_like(x)
        transformed_X = torch.sqrt(alpha_bar) * x + torch.sqrt(1-alpha_bar) * z
        if self.keypoint_conditional:
            # if keypoint_conditional, the keypoint positions will not be diffused, the model infers the features based on keypoint positions
            transformed_X = torch.cat([keypoint, transformed_X[:,:,keypoint_dim:]], dim=2)
        model_output = net(transformed_X, ts=diffusion_steps, label=label)
        model_output = model_output * self.model_output_scale_factor
        # loss = torch.sum((model_output - z)**2, dim=2).mean(dim=1) # of shape (B)
        mse = (model_output - z)**2
        
        # put more emphasize on the keypoint location instead of feartures
        loss = (self.keypoint_position_loss_weight * mse[:,:,0:keypoint_dim].sum(dim=2) + 
                self.feature_loss_weight * mse[:,:,keypoint_dim:].mean(dim=2))
        loss = loss.mean(dim=1)
        return loss


    # def denoise_and_reconstruct(self, n, model, keypoint_dim, shape, label=None, n_steps=None, x=None, curr_step=None,
    #             callback=lambda x, i, x0=None: None, keypoint=None, return_keypoint_feature=False):
    def denoise_and_reconstruct(self, n, model, keypoint_dim, shape, label=None, n_steps=None, x=None, curr_step=None,
                                keypoint=None, return_keypoint_feature=False, local_resampling=False, 
                                complete_x0=None, keypoint_mask=None):
        # n is the batchsize
        # x could be the initial points with features that we want to sampling from if it's not None
        # it could also be the points with features that we want to perform local re-sampling for some features
        if local_resampling:
            # in this case, we resample features on some points in complete_x0 (B,N,3+F) 
            # keypoint_mask is a tensor of shape (B,N) that contain 0 and 1, 
            # 1 indicates the points we want to resample features for
            assert self.keypoint_conditional
            assert x is None
            complete_x0 = complete_x0.to(self.device)
            keypoint_mask = keypoint_mask.to(self.device)

        with torch.no_grad():
            if curr_step is None:
                curr_step = self.num_timesteps

            assert curr_step > 0, curr_step

            if n_steps is None or curr_step-n_steps < 0:
                n_steps = curr_step

            if x is None:
                assert curr_step == self.num_timesteps, curr_step
                # start the chain with x_T from normal distribution
                x = torch.randn(n, *shape)
                x = x.to(self.device)

            # when start from scratch
            # curr_step = num_timesteps = n_steps
            # i range from 0 to num_timesteps-1
            # reversed i range from num_timesteps-1 to 0
            for i in reversed(range(curr_step-n_steps, curr_step)):
                # print(i, flush=True)
                t = (torch.ones(n)*i).to(self.device)
                if self.keypoint_conditional:
                    assert keypoint.shape[2] == keypoint_dim
                    x = torch.cat([keypoint, x[:,:,keypoint_dim:]], dim=2)
                x, x0 = denoising_step(x, t=t, model=model, logvar=self.logvar, 
                    sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                    sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                    posterior_mean_coef1=self.posterior_mean_coef1,
                    posterior_mean_coef2=self.posterior_mean_coef2,
                    return_pred_xstart=True, label=label,
                    data_clamp_range=self.data_clamp_range,
                    local_resampling=local_resampling, complete_x0=complete_x0, keypoint_mask=keypoint_mask)
                # callback(x, i, x0=x0)
            if self.keypoint_conditional:
                assert keypoint.shape[2] == keypoint_dim
                x = torch.cat([keypoint, x[:,:,keypoint_dim:]], dim=2)
            keypoint = x[:,:,0:keypoint_dim]
            keypoint_feature = x[:,:,keypoint_dim:]
            reconstructed_pointcloud = self.decode(x, keypoint_dim, label)
            if return_keypoint_feature:
                return reconstructed_pointcloud, keypoint, keypoint_feature
            else:
                return reconstructed_pointcloud, keypoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', help='JSON file for configuration')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    standard_diffusion_config = config['standard_diffusion_config']
    diffusion_model = Diffusion(standard_diffusion_config)
    pdb.set_trace()
