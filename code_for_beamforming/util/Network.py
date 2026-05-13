
import torch.nn as nn

import torch



class Designed_DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, Batch_normal, output_size,dropout_prob=0.0, num_groups=32 ):
        super(Designed_DNN, self).__init__()
        self.layers = nn.ModuleList()  # Stores all the linear layers
        self.norms = nn.ModuleList()  # Stores all the normalization layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.erelu = nn.ELU()
        previous_size = input_size

        # Initialize layers dynamically based on hidden_sizes
        for i, size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(previous_size, size))
            if Batch_normal == 'GN':
                self.norms.append(nn.GroupNorm(num_groups=min(num_groups, size), num_channels=size))
            elif Batch_normal == 'LN':
                self.norms.append(nn.LayerNorm(size))
            elif Batch_normal == 'BN':
                self.norms.append(nn.BatchNorm1d(size))
            elif Batch_normal == 'None':
                self.norms.append(nn.Identity())
            previous_size = size

        # Output layer without normalization

        # self.final_normal = self.make_norm(Batch_normal, output_size, num_groups)
        self.final_normal = nn.BatchNorm1d(output_size)

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def make_norm(self, Batch_normal, size, num_groups):
        if Batch_normal == 'GN':
            # Group number must divide channel number
            group_num = min(num_groups, size)
            while size % group_num != 0:
                group_num -= 1
            return nn.GroupNorm(num_groups=group_num, num_channels=size)

        elif Batch_normal == 'LN':
            return nn.LayerNorm(size, track_running_stats=False)

        elif Batch_normal == 'BN':
            return nn.BatchNorm1d(size)

        elif Batch_normal == 'None':
            return nn.Identity()

        else:
            raise ValueError(f"Unknown normalization type: {Batch_normal}")
    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    def forward(self, x):

        for layer, norm in zip(self.layers, self.norms):
            # x = self.relu(norm(layer(x)))
            x = norm(self.relu(layer(x)))
            if self.dropout.p > 0:
                x = self.dropout(x)
        x = self.final_normal(self.output_layer(x))  # Apply output layer without activation and normalization
        # x =  (self.output_layer(x))  # Apply output layer without activation and normalization
        # x =  self.output_layer(x)   # Apply output layer without activation and normalization
        return x



class EMA:
    def __init__(self, models, decay=0.999):
        self.models = models  # List of models
        self.decay = decay  # EMA decay rate
        self.shadow = {}  # Dictionary to store the shadow parameters

        # Initialize the shadow parameters
        for model_idx, model in enumerate(self.models):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Use model index as a prefix to ensure unique names
                    shadow_name = f"model{model_idx}.{name}"
                    self.shadow[shadow_name] = param.clone().detach()
                    self.shadow[shadow_name].requires_grad = False

    def update(self):
        for model_idx, model in enumerate(self.models):
            for name, param in model.named_parameters():
                # Skip BatchNorm running statistics
                if "running_mean" in name or "running_var" in name:
                    continue

                if param.requires_grad:
                    shadow_name = f"model{model_idx}.{name}"

                    if shadow_name not in self.shadow:
                        print(f"Initializing shadow for {shadow_name}")
                        self.shadow[shadow_name] = param.clone().detach()
                        self.shadow[shadow_name].requires_grad = False

                    if param.data.size() == self.shadow[shadow_name].data.size():
                        self.shadow[shadow_name].data = self.decay * self.shadow[shadow_name].data + (1 - self.decay) * param.data
                    else:
                        print(f"Skipping update for {shadow_name} due to shape mismatch.")

    def apply(self):
        for model_idx, model in enumerate(self.models):
            for name, param in model.named_parameters():
                # Skip BatchNorm running statistics
                if "running_mean" in name or "running_var" in name:
                    continue

                shadow_name = f"model{model_idx}.{name}"
                if shadow_name in self.shadow:
                    if param.data.size() == self.shadow[shadow_name].data.size():
                        param.data.copy_(self.shadow[shadow_name].data)
                    else:
                        print(f"Skipping update for {shadow_name} due to shape mismatch.")



import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True)
            ]
            prev = h

        layers += [nn.Linear(prev, output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionFusionNet(nn.Module):
    def __init__(self, lidar_dim, pilot_dim, output_dim, ue_num, antenna_sum):
        super().__init__()

        self.ue_num = ue_num
        self.antenna_sum = antenna_sum
        self.output_dim = output_dim

        fusion_dim = 512

        # modality encoders
        self.lidar_encoder = MLPBlock(
            lidar_dim,
            [1024, 512],
            fusion_dim
        )

        self.pilot_encoder = MLPBlock(
            pilot_dim,
            [1024, 512],
            fusion_dim
        )

        # attention weight generator
        self.attn_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        # output head
        self.output_head = MLPBlock(
            fusion_dim,
            [512, 512],
            output_dim
        )

    def forward(self, lidar_x, pilot_x):

        f_lidar = self.lidar_encoder(lidar_x)
        f_pilot = self.pilot_encoder(pilot_x)

        attn_logits = self.attn_mlp(
            torch.cat([f_lidar, f_pilot], dim=-1)
        )

        attn = F.softmax(attn_logits, dim=-1)

        alpha_lidar = attn[:, 0:1]
        alpha_pilot = attn[:, 1:2]

        fused = alpha_lidar * f_lidar + alpha_pilot * f_pilot

        out = self.output_head(fused)

        return out
class UserTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads=4, num_layers=2, ff_dim=1024, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu',
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x: [B, UE, D]
        out = self.encoder(x)
        out = self.norm(out + x)   # residual
        return out