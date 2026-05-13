import torch
from torch.utils.data import Dataset
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
# from pytorch3d.ops import sample_farthest_points


class CustomDataset(Dataset):
    def __init__(self, txt_file, seed=None,  device='cpu', mode='train',input_type="multimodal"):
        # Set seed if provided
        # seed=2323
        self.num_points=256
        self.max_object = 18
        self.UE_num=3
        self.block_size=10
        if seed is not None:
            self.set_seed(seed)

        with open(txt_file, 'r') as file:
            self.file_paths = [path.strip() for path in file.readlines()]
        self.commun_paths = [
            file_path.replace('detection', 'sample_data').replace('.pth', '.npz').replace('det', 'communication_2d')
            for file_path in self.file_paths]
        self.lidar_path = [
            file_path.replace('detection_', 'bin').replace('data2', 'data').replace('.pth', '.bin').replace('det','filtered_lidar_points')
            for file_path in self.file_paths]

        self.label_2d_benchmark=[
            file_path.replace('detection_', 'cam4_sample_').replace('.pth', '.txt').replace('det','label')
            for file_path in self.file_paths]
        self.cache = False
        self.data_cache = {}
        self.label_3d_cache = {}
        self.lidar_cache={}
        self.device = torch.device(device)  # Specify the device (default is CPU)
        self.mode = mode.lower()
        self.input_type=input_type
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load from file system if not cached
        file_path = self.file_paths[idx] #image
        commun_path = self.commun_paths[idx] #pilot
        if self.input_type=="multimodal" or self.input_type=="lidar" or self.input_type=="UE_and_pilot":
            lidar_path=self.lidar_path[idx] #lidar
            lidar_np =  torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :5])
            device = lidar_np.device
            output = torch.full((self.max_object, self.num_points, 5), -1.0, device=device)
            # --- UE objects ---
            for obj_id in range(self.UE_num):
                points = lidar_np[lidar_np[:, 3] == obj_id + 1]
                n = points.size(0)
                if n == 0:
                    continue
                idx = torch.randint(0, n, (self.num_points,), device=device) if n < self.num_points \
                    else torch.randperm(n, device=device)[:self.num_points]
                output[obj_id] = points[idx]

            # --- Other objects ---
            other_points = lidar_np[lidar_np[:, 3] > self.UE_num]
            unique_labels = torch.unique(other_points[:, 4].long())
            index_temp = self.UE_num

            for lbl in unique_labels:
                if index_temp >= self.max_object:
                    break
                points = other_points[other_points[:, 4].long() == lbl]
                n = points.size(0)
                if n == 0:
                    continue
                idx = torch.randint(0, n, (self.num_points,), device=device) if n < self.num_points \
                    else torch.randperm(n, device=device)[:self.num_points]
                output[index_temp] = points[idx]
                index_temp += 1
            output=output[:,:,:4]
        else:
            output=0
        commun_npz = np.load(commun_path)
        location = torch.from_numpy(commun_npz['location'])
        amplitude = self.pad_or_truncate1(torch.from_numpy(commun_npz['applitude']), target_cols=30)
        phi_r = self.pad_or_truncate1(torch.from_numpy(commun_npz['phi_r']), target_cols=30)
        theta_r = self.pad_or_truncate1(torch.from_numpy(commun_npz['theta_r']), target_cols=30)
        tau = self.pad_or_truncate1(torch.from_numpy(commun_npz['tau']), target_cols=30)
        communs = (location, amplitude, phi_r, theta_r, tau)
        #
        # data = torch.load(file_path, weights_only=True)
        # bb_box = data["bbox_2d"][2:3]
        # keypoint_from_image = data["keypoint"][2:3]
        return  communs,output,0, file_path

    def custom_collate(self, batch):
        communs, lidar_list,keypoint_benchmark,  paths = zip(*batch)

        communs = [torch.stack(c) for c in zip(*communs)]
        if self.input_type=="multimodal" or self.input_type=="lidar" or self.input_type=="UE_and_pilot":
            lidar_padded = pad_sequence(lidar_list, batch_first=True, padding_value=0)  # shape: [B, N_max, 3]
            return  communs,lidar_padded, keypoint_benchmark,paths
        else:
            return  communs,0, keypoint_benchmark,paths

    def pad_or_truncate1(self,tensor, target_cols, pad_value=0):

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(1)  # Make it 2D if 1D

        # Truncate columns if necessary
        if tensor.size(1) == 2:
            tensor = tensor[:, :target_cols]

        # Pad columns if necessary
        pad_cols = max(0, target_cols - tensor.size(1))
        tensor = F.pad(tensor, (0, pad_cols), mode='constant', value=pad_value)

        return tensor
    def pad_or_truncate(self,tensor, target_cols, pad_value=0):
        #tensor:2,3 user,X object
        if tensor.size(1)==2:
            print(tensor)
        if tensor.dim()<3:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)  # Make it 2D if 1D

            # Truncate columns if necessary
            if tensor.size(1) == 2:
                tensor = tensor[:, :target_cols]

            # Pad columns if necessary
            pad_cols = max(0, target_cols - tensor.size(1))
            tensor = F.pad(tensor, (0, pad_cols), mode='constant', value=pad_value)
        else:
            if tensor.size(2) > target_cols:
                tensor = tensor[:, :, :target_cols]
            pad_cols = max(0, target_cols - tensor.size(2))
            tensor = F.pad(tensor, (0, pad_cols), mode='constant', value=pad_value)

        return tensor
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

