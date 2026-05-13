# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import sys
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import torch
from Sample_dataloader import *
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
import json
# Define the directory as a Path object
from util.util import *
from util.Network import *
import time
class BeamTraining:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device


    def get_unique_save_dir(self, base_dir, pilot_num, feature):
        base_path = Path(base_dir)
        save_path = base_path.with_name(f"{base_path.name}f_{feature}_p{pilot_num}")
        if not save_path.exists():
            save_path.mkdir(parents=True)
            return save_path

        index = 1
        while True:
            new_path = base_path.with_name(f"{base_path.name}f_{feature}_p{pilot_num}_{index}")
            if not new_path.exists():
                new_path.mkdir(parents=True)
                return new_path
            index += 1
    def save_setup_parameters(self):
        # Automatically capture and save all attributes of the instance
        with open(self.save_dir / 'setup_parameters.json', 'w') as f:
            json.dump(vars(self), f, indent=4, default=str)  # Using default=str to handle non-serializable objects like 'device'
        print("Setup parameters saved to:", self.save_dir / 'setup_parameters.json')
    def setup(self, training=True,testing=False):
        self.set_communication_setting()
        self.training=training
        self.testing=testing
        self.restore=0
        self.dropout_prob =0
        if self.training:
            self.save_dir = self.get_unique_save_dir(Path(opt.save_dir), opt.pilot_num, opt.feature)
            self.weights_dir = self.save_dir / "weights"
            self.log_file_path = self.save_dir / "train.log"
            sys.stdout = beam_Logger(self.log_file_path)
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            self.last_weight, self.best_weight = self.weights_dir / "last_beam.pt", self.weights_dir / "best_beam.pt"

        self.mode=self.opt.feature
        self.train_dataset = CustomDataset(self.opt.train_file, seed=0,  mode="train",  input_type=self.mode)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=4, pin_memory=True,collate_fn=self.train_dataset.custom_collate)
        self.clip_gradient_label = True
        self.clip_gradient_value = 200
        if training:
            self.valid_dataset = CustomDataset(self.opt.valid_file, seed=0,  mode="test",  input_type=self.mode)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=4, pin_memory=True,collate_fn=self.valid_dataset.custom_collate)
        else:
            self.test_dataset = CustomDataset(self.opt.test_file, seed=0,  mode="test",  input_type=self.mode)
            self.valid_loader = DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=8, pin_memory=True,collate_fn=self.test_dataset.custom_collate)
        self.nbs = 256      # nominal batch size
        self.accumulate = max(round(self.nbs / self.opt.batch_size), 1)
        self.nb = len(self.train_loader)
        self.last_opt_step = -1
        self.min_loss = float('inf')
        self.learning_rate = 1e-3
        self.GNN_layers =2
        self.output_channel_est_size = self.antenna_sum * 2
        self.p_drop=0.2
        self.set_up_pilot_models()
        self.set_up_lidar_models()
        self.set_up_multimodal_models()

        self.models = [  self.LNNs,self.MNNs, self.PNNs  ]  # ← wrap in list
        if self.training:
            self.save_setup_parameters()
            # self.create_directory(self.save_dir)
    def aggregation_block(self, feature, f_s,f_a,f_c):
        batchsize=feature.size(0)
        s_output = f_s(feature.view(batchsize * self.UE_num, -1)).view(batchsize, self.UE_num, -1)
        a_output = f_a(feature.view(batchsize * self.UE_num, -1)).view(batchsize, self.UE_num, -1)
        a_output = torch.mean(a_output, dim=1, keepdim=True)-1/self.UE_num*a_output
        c_output = f_c(
            torch.cat([s_output.view(batchsize * self.UE_num, -1), a_output.view(batchsize * self.UE_num, -1)],
                      dim=1)).view(batchsize, self.UE_num, -1)
        return c_output
    def set_up_lidar_models(self):
        feature_layer1 = [512];self.lidar_output_feature=512;input_feature=4
        self.lidar_user_dnn = Designed_DNN(input_feature, feature_layer1, self.opt.Batch_normal, self.lidar_output_feature,
                                           self.dropout_prob).to(device).float()
        self.lidar_obstacle_dnn1 = Designed_DNN(input_feature, feature_layer1, self.opt.Batch_normal,self.lidar_output_feature,
                                             self.dropout_prob).to(device).float()
        self.lidar_obstacle_dnn2 = Designed_DNN(self.lidar_output_feature, feature_layer1, self.opt.Batch_normal,self.lidar_output_feature,
                                             self.dropout_prob).to(device).float()
        self.lidar_linear = torch.nn.Linear(self.lidar_output_feature, self.output_channel_est_size).to(self.device).float()

        self.LNNs = [self.lidar_user_dnn, self.lidar_obstacle_dnn1 , self.lidar_obstacle_dnn2 , self.lidar_linear]


    def set_up_multimodal_models(self):
        m_feat_layer = [512, 512];self.m_feat_size=512
        if self.mode == "multimodal":
            input_dim = self.m_feat_size * 3
        elif self.mode == "UE_and_pilot":
            input_dim = self.m_feat_size * 2
        else:
            input_dim = self.m_feat_size
        self.m_f_a0 = Designed_DNN(input_dim, m_feat_layer, self.opt.Batch_normal, self.m_feat_size // 2,
                                   self.dropout_prob).to(self.device).float()

        self.m_f_s0 = Designed_DNN(input_dim, m_feat_layer, self.opt.Batch_normal, self.m_feat_size // 2,
                                   self.dropout_prob).to(self.device).float()

        self.m_f_c0 = Designed_DNN(self.m_feat_size, m_feat_layer, self.opt.Batch_normal, self.m_feat_size,
                                   self.dropout_prob).to(self.device).float()

        self.m_f_a1 = Designed_DNN(self.m_feat_size, m_feat_layer, self.opt.Batch_normal, self.m_feat_size // 2,
                                   self.dropout_prob).to(self.device).float()
        self.m_f_s1 = Designed_DNN(self.m_feat_size, m_feat_layer, self.opt.Batch_normal, self.m_feat_size // 2,
                                   self.dropout_prob).to(self.device).float()
        self.m_f_c1 = Designed_DNN(self.m_feat_size, m_feat_layer, self.opt.Batch_normal, self.m_feat_size,
                                   self.dropout_prob).to(self.device).float()
        self.m_linear = torch.nn.Linear(self.m_feat_size, self.output_channel_est_size).to(self.device).float()
        self.MNNs = [self.m_f_a0, self.m_f_s0, self.m_f_c0, self.m_f_a1, self.m_f_s1, self.m_f_c1,self.m_linear]


    def set_up_pilot_models(self):
        self.p_feat_size=512;p_feat_layer=[512]
        self.P_feature_NN = Designed_DNN(self.antenna_sum * 2 + 1, p_feat_layer, self.opt.Batch_normal,
                                         self.p_feat_size, self.dropout_prob).to(self.device).float()

        self.p_f_a0 = Designed_DNN(self.p_feat_size, p_feat_layer, self.opt.Batch_normal, self.p_feat_size // 2,
                                   self.dropout_prob).to(self.device).float()

        self.p_f_s0 = Designed_DNN(self.p_feat_size, p_feat_layer, self.opt.Batch_normal, self.p_feat_size // 2,
                                   self.dropout_prob).to(self.device).float()

        self.p_f_c0 = Designed_DNN(self.p_feat_size, p_feat_layer, self.opt.Batch_normal, self.p_feat_size,
                                   self.dropout_prob).to(self.device).float()

        self.PNNs=[self.P_feature_NN,self.p_f_a0, self.p_f_s0, self.p_f_c0 ]

    def set_communication_setting(self):
        self.P_limit=1
        self.UE_num=3
        self.x_antenna=16
        self.z_antenna=2
        self.antenna_sum=self.x_antenna*self.z_antenna
        self.pilot_num=self.opt.pilot_num
        self.antenna_array=np.array([self.x_antenna,self.z_antenna])
        self.frac_d_lambda=0.5
        # Scaling factor used to enlarge the channel magnitude to a suitable range.
        self.snr_linear = 10 ** (70 / 10)  # This value can be interpreted as the linear SNR computed from the difference between transmit power and noise power.
        self.noise_power_std=opt.noise_power_std
        self.frequency = 3.5e9


    import torch

    def min_rate_ignore_no_path(self,w, h, noise=1e-3, tol=1e-12):

        ch_pow = (h.abs() ** 2).sum(dim=2)  # [B, K]
        valid = ch_pow > tol  # [B, K]

        # Effective channels: h_k^H w_j
        eff = torch.einsum('bkn,bjn->bkj', h.conj(), w)  # [B, K, K]
        pwr = eff.abs() ** 2  # |h_k^H w_j|^2
        signal = pwr.diagonal(dim1=1, dim2=2)  # desired power [B,K]
        interf = pwr.sum(dim=2) - signal  # interference [B,K]

        # SINR and rate
        sinr = signal / (interf + noise + 1e-12)
        rate = torch.log2(1 + sinr.clamp_min(0))  # [B, K]
        # Mask invalid users (no-path)
        rate_masked = rate.clone()
        rate_masked[~valid] = float('inf')  # ignored in min()

        # Minimum rate per batch
        min_rate = rate_masked.min(dim=1).values  # [B]

        # If a sample has no valid user, set 0 (or nan)
        no_valid = ~valid.any(dim=1)
        min_rate[no_valid] = 0.0

        return min_rate

    def generate_position_grid(self, frac_d_lambda):
        x_positions = (torch.arange(self.x_antenna, dtype=torch.float32) * frac_d_lambda).to(self.device)
        x_positions_new = x_positions - torch.mean(x_positions)
        z_positions = (torch.arange(self.z_antenna, dtype=torch.float32) * frac_d_lambda).to(self.device)
        z_positions_new = z_positions - torch.mean(z_positions)
        x_grid, z_grid = torch.meshgrid(x_positions_new, z_positions_new, indexing='ij')
        return x_grid, z_grid
    def uplink_pilot(self,commun_data, no_noise=0):
        amplitude,phi_angles, theta_angles,tau=torch.squeeze(commun_data[1]),torch.squeeze(commun_data[2]),torch.squeeze(commun_data[3]),torch.squeeze(commun_data[4])
        scalor_factor = math.sqrt(self.snr_linear)

        x_grid, z_grid = self.generate_position_grid(self.frac_d_lambda)
        phase_shifts_x = (2 * torch.pi) * x_grid.view(1, 1, *x_grid.shape) * torch.sin(
            theta_angles.view(*theta_angles.shape, 1, 1)) * torch.cos(phi_angles.view(*phi_angles.shape, 1, 1))
        phase_shifts_z = (2 * torch.pi) * z_grid.view(1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape, 1, 1))
        phase_shifts = phase_shifts_x + phase_shifts_z
        phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2], self.x_antenna * self.z_antenna)
        channel = amplitude.unsqueeze(-1) * phase_shifts_e* scalor_factor
        channel_expanded = channel.sum(dim=2)  # Shape: (batch_size, path_num, antenna_num, 1)
        noise_size =(*channel_expanded.shape,1)
        noise = 1 / math.sqrt(2.0*self.pilot_num) * (torch.randn(noise_size, device=self.device) + 1j * torch.randn(noise_size, device=self.device)) * self.noise_power_std
        pilot_frequency_with_noise=channel_expanded.unsqueeze(-1)+noise
        # received_signal_with_noise_frequency=frequency_channel_sum_path.unsqueeze(-1)+noise
        return pilot_frequency_with_noise,channel_expanded
    def train_epoch(self, epoch):
        losses = []  # Initialize list to store losses
        RSS=[]
        for model_group in self.models:  # Iterate over groups of models
            for model in model_group:  # Iterate over individual models within each group
                model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.opt.epochs}", leave=True)
        for pbar_index, ( commun, lidar_raw,keypoint_benchmark,paths) in enumerate(pbar):
            (location, amplitude, phi_r, theta_r,tau) = [x.to(self.device) for x in commun]
            self.paths=paths
            ni = pbar_index + self.nb * epoch  # number integrated batches (since train start)
            self.train_batch = len(location)
            pilot_with_noise,real_channel=self.uplink_pilot([location,amplitude,phi_r,theta_r,tau])
            P_input_feature=torch.cat([pilot_with_noise.real,pilot_with_noise.imag],dim=2).permute(0,1,3,2)
            pilot_input=torch.cat([P_input_feature.reshape(-1 ,self.antenna_sum*2),torch.ones([self.train_batch*self.UE_num,1],device=self.device)*self.pilot_num],dim=-1)
            pilot_feature=self.P_feature_NN(pilot_input).view(self.train_batch,self.UE_num,-1)
            pilot_feature = self.aggregation_block(pilot_feature, self.p_f_s0, self.p_f_a0, self.p_f_c0)
            if self.mode != "pilot":
                lidar_raw = lidar_raw.to(self.device)
                lidar_normaliezed = lidar_raw / torch.tensor([100, 100, 25, 7], device=self.device,
                                                                 dtype=torch.float32).view(1, 1, 1,
                                                                                           4)  # R_inv_K_inv_uv1:torch.Size([256, 4, 6, 8, 3])

                self.obstacle_num = lidar_normaliezed.size(1) - self.UE_num
                sample_point = lidar_normaliezed.size(2)
                lidar_obstalce_feature1 = torch.mean(
                    self.lidar_obstacle_dnn1(lidar_normaliezed[:, self.UE_num:].reshape(-1, 4)).reshape(
                        self.train_batch, self.obstacle_num,sample_point, -1), dim=2)  # batchsize,512
                lidar_obstalce_feature = torch.sum(
                    self.lidar_obstacle_dnn2(lidar_obstalce_feature1.reshape(self.train_batch*self.obstacle_num,-1)).reshape(
                        self.train_batch, self.obstacle_num, -1), dim=1)  # batchsize,512
                lidar_transmitter_feature = torch.mean(
                    self.lidar_user_dnn(lidar_normaliezed[:, :self.UE_num].reshape(-1, 4)).reshape(
                        self.train_batch, self.UE_num, sample_point, -1), dim=2)  # batchsize,UE,256,512

                drop_mask = (torch.rand(self.train_batch, 1, 1, device=self.device) > self.p_drop).float()
                lidar_transmitter_feature = lidar_transmitter_feature * drop_mask
                lidar_obstalce_feature = lidar_obstalce_feature.view(self.train_batch, 1, -1) * drop_mask

                if self.mode == "UE_and_pilot":
                    final_feature = torch.cat([lidar_transmitter_feature,pilot_feature], dim=-1)
                if self.mode == "multimodal":
                    final_feature = torch.cat([lidar_transmitter_feature ,lidar_obstalce_feature.view(self.train_batch, 1, -1).expand(-1,self.UE_num,-1) , pilot_feature],dim=-1)
            else:
                final_feature = pilot_feature

            multimodal_fc0_output = self.aggregation_block(final_feature, self.m_f_s0, self.m_f_a0, self.m_f_c0)
            multimodal_fc1_output = self.aggregation_block(multimodal_fc0_output, self.m_f_s1, self.m_f_a1, self.m_f_c1)
            output = self.m_linear(multimodal_fc1_output)
            beamforming=(output[:,:,:self.antenna_sum ]+1j*output[:,:,self.antenna_sum:])
            power = (beamforming.real ** 2 + beamforming.imag ** 2).sum(dim=(1, 2), keepdim=True)  # [B,1,1]
            beamforming_normalized = beamforming / torch.sqrt(power / self.P_limit)
            minimum_rate=self.min_rate_ignore_no_path(beamforming_normalized,real_channel)

            train_loss =- torch.mean(minimum_rate)
            losses.append(train_loss)
            train_loss.backward()
            if ni - self.last_opt_step >= self.accumulate:
                if self.clip_gradient_label:
                    # Clip gradients for all models in INNs, MNNs, and PNNs
                    for model_group in [ self.LNNs,self.MNNs, self.PNNs ]:
                        for model in model_group:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_gradient_value)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.last_opt_step = ni
        mean_loss = sum(losses) / len(losses)
        print(f"Mean Loss for Epoch {epoch+1}: {mean_loss:.4f}")

    def train(self):
        # Set all models to training mode
        all_parameters = [param for model_group in self.models for model in model_group for param in model.parameters()]
        # Example optimizer creation (optional, adjust based on your optimizer choice)
        self.optimizer = torch.optim.Adam(all_parameters,lr=self.learning_rate)
        self.optimizer.zero_grad()
        if self.restore:
             self.restore_model(f"Train_noise2_complex/trainf_multimodal_p16_2/weights/best_beam.pt")
        for epoch in range(self.opt.epochs):
            self.epoch = epoch
            self.train_epoch( epoch)
            self.validate()
    def test(self):
        # Set all models to training mode
        all_parameters = [param for model_group in self.models for model in model_group for param in model.parameters()]
        self.optimizer = torch.optim.Adam(all_parameters,lr=self.learning_rate)
        self.optimizer.zero_grad()
        # self.restore_model(f"Revised_noise{self.noise_power_std}_complex/trainf_multimodal_p{self.pilot_num}/weights/best_beam.pt")
        self.restore_model(f"/home/lyh/multimodal_code/Train_noise2_complex/trainf_multimodal_p16/weights/best_beam.pt")
        self.validate()
    def validate(self):
        for model_group in self.models:  # Iterate over groups of models
            for model in model_group:  # Iterate over individual models within each group
                model.eval()
        valid_loss = []
        with torch.no_grad():  # Disable gradient calculation
            for  commun, lidar_raw,keypoint_benchmark, paths in self.valid_loader:
                self.training=False
                self.valid_batch = len(paths)
                (location, amplitude, phi_r, theta_r, tau) = [x.to(self.device) for x in commun]
                self.paths = paths
                pilot_with_noise, real_channel = self.uplink_pilot([location, amplitude, phi_r, theta_r, tau])
                P_input_feature = torch.cat([pilot_with_noise.real, pilot_with_noise.imag], dim=2).permute(0, 1, 3, 2)
                pilot_input = torch.cat([P_input_feature.reshape(-1, self.antenna_sum * 2),
                                         torch.ones([self.valid_batch * self.UE_num, 1],
                                                    device=self.device) * self.pilot_num], dim=-1)
                pilot_feature = self.P_feature_NN(pilot_input).view(self.valid_batch, self.UE_num, -1)
                pilot_feature = self.aggregation_block(pilot_feature, self.p_f_s0, self.p_f_a0, self.p_f_c0)
                if self.mode!= "pilot":
                    lidar_raw = lidar_raw.to(self.device)
                    lidar_normlized = lidar_raw / torch.tensor([100, 100, 25, 7], device=self.device,
                                                                     dtype=torch.float32).view(1, 1, 1, 4)  # R_inv_K_inv_uv1:torch.Size([256, 4, 6, 8, 3])
                    self.obstacle_num = lidar_normlized.size(1) - self.UE_num
                    sample_point = lidar_normlized.size(2)
                    lidar_obstalce_feature1 = torch.mean(
                        self.lidar_obstacle_dnn1(lidar_normlized[:, self.UE_num:].reshape(-1, 4)).reshape(
                            self.valid_batch, self.obstacle_num, sample_point, -1), dim=2)  # batchsize,512
                    lidar_obstalce_feature = torch.sum(
                        self.lidar_obstacle_dnn2(
                            lidar_obstalce_feature1.reshape(self.valid_batch * self.obstacle_num, -1)).reshape(
                            self.valid_batch, self.obstacle_num, -1), dim=1)  # batchsize,512
                    lidar_user_feat = torch.mean(
                        self.lidar_user_dnn(lidar_normlized[:, :self.UE_num].reshape(-1, 4)).reshape(
                            self.valid_batch, self.UE_num, sample_point, -1), dim=2)  # batchsize,UE,256,512
                    if self.mode=="UE_and_pilot":
                        final_feature = torch.cat([lidar_user_feat,pilot_feature], dim=-1)
                    if self.mode == "multimodal":
                        final_feature = torch.cat([lidar_user_feat,
                                                   lidar_obstalce_feature.view(self.valid_batch, 1, -1).expand(-1,self.UE_num,-1),pilot_feature], dim=-1)
                else:
                    final_feature = pilot_feature

                multimodal_fc0_output = self.aggregation_block(final_feature, self.m_f_s0, self.m_f_a0, self.m_f_c0)
                multimodal_fc1_output = self.aggregation_block(multimodal_fc0_output, self.m_f_s1, self.m_f_a1, self.m_f_c1)
                output = self.m_linear(multimodal_fc1_output)

                beamforming = (output[:, :, :self.antenna_sum] + 1j * output[:, :, self.antenna_sum:])
                power = (beamforming.real ** 2 + beamforming.imag ** 2).sum(dim=(1, 2), keepdim=True)  # [B,1,1]
                beamforming_normalized = beamforming / torch.sqrt(power / self.P_limit)
                minimum_rate = self.min_rate_ignore_no_path(beamforming_normalized, real_channel)
                loss = -torch.mean(minimum_rate)  # nmse = torch.sum(torch.square(torch.abs(complex_output - time_domain_channel)),dim=-1)/torch.sum(torch.square(torch.abs( channel)),dim=-1)
                valid_loss.append(loss)
                self.training = True
        flat_tensor = torch.cat([t.flatten() for t in valid_loss])
        new_loss = torch.mean(flat_tensor.float())
        print('test/val epoch:',self.epoch,'loss:',new_loss)
        if new_loss < self.min_loss and self.training and (not self.testing):
            self.min_loss = new_loss
            # Prepare checkpoint data
            all_model_groups = {
                "MNNs": self.MNNs,
                "PNNs": self.PNNs,
                "LNNs": self.LNNs,}
            # Create checkpoint dictionary
            ckpt = {
                "epoch": self.epoch,
                "models": {name: [model.state_dict() for model in models] for name, models in all_model_groups.items()},
                "optimizer": self.optimizer.state_dict(),}
            torch.save(ckpt, self.best_weight)


    def restore_model(self, checkpoint_path, load_groups=None, freeze=False):
        """
        load_groups: list of model groups to load, e.g., ['LNNs', 'PNNs']
        freeze: set True to freeze parameters after loading
        """
        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        saved_groups = checkpoint.get("models", {})
        self.epoch = checkpoint.get("epoch", 0)
        # Define model groups in the current class
        model_groups = {
            "MNNs": self.MNNs,
            "PNNs": self.PNNs,
            "LNNs": self.LNNs,
        }

        if load_groups is None:
            load_groups = list(model_groups.keys())

        for group_name in load_groups:
            if group_name in saved_groups:
                for i, model in enumerate(model_groups[group_name]):
                    try:
                        model.load_state_dict(saved_groups[group_name][i], strict=False)
                        print(f"[INFO] Restored {group_name}[{i}]")
                        if freeze:
                            for p in model.parameters():
                                p.requires_grad = False
                    except Exception as e:
                        print(f"[WARNING] Failed to load {group_name}[{i}]: {e}")
            else:
                print(f"[WARNING] {group_name} not found in checkpoint")

        print("[INFO] Partial model restoration done.")

    def create_directory(self, directory):
        base_directory = directory
        i = 1
        while os.path.exists(directory):
            directory = f"{base_directory}_{i}"
            i += 1
        if self.training:
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        return directory


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_label =1
    noise_power_std=2*np.sqrt(2)
    perfect_channel=0

    for pilot_num in [ 16]:
        opt = argparse.Namespace(
            workers=6,
            batch_size=128,
            pilot_num=pilot_num,
            perfect_channel=perfect_channel,
            noise_power_std=noise_power_std,
            save_dir=f"Train_noise{int(noise_power_std)}_complex/train/",
            feature="multimodal", #  #"pilot", "lidar","multimodal","UE_and_pilot"
            epochs=150,
            Batch_normal='BN',  # 'GN', 'LN', 'BN', 'None'
            weights=None,  # Specify path to weights if needed
            train_file='/home/lyh/Multi_user_data_generation/data//train3.txt',
            valid_file='/home/lyh/Multi_user_data_generation/data//val3.txt',
            test_file='/home/lyh/Multi_user_data_generation/data//test3_5000.txt'
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU instead.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training = BeamTraining(opt, device)
        training.setup(training=training_label,testing=1-training_label)
        if training_label==1:
            training.train()
        else:
            training.test()

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
