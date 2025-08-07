"""
SLPF (Spatial-temporal Long-term Partial sensing Forecast model) Model

This module implements the main SLPF model for multi-stage spatial-temporal prediction:
1. ADP (Adaptation): Learn to predict unsensed locations from sensed history
2. Forecast: Learn to predict future values for sensed locations
3. Aggregation: Combine historical and future information for final prediction

The model uses graph neural networks with enhanced node embeddings and spatial
transfer matrices to capture complex spatial-temporal dependencies.
"""

import torch
from torch import nn


class MultiLayerPerceptron_(nn.Module):
    """
    Multi-Layer Perceptron with residual connection and intermediate skip connection.
    
    This variant adds a skip connection from the first layer directly to the output,
    providing better gradient flow for training deeper networks.
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the MLP with residual connections.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, 
            kernel_size=(1, 1), bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, 
            kernel_size=(1, 1), bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            input_data: Input tensor [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Output with residual connections applied
        """
        # Standard MLP forward pass
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        
        # Add skip connection from first layer (residual + MLP)
        hidden = hidden + self.act(self.fc1(input_data))
        
        return hidden


class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron with standard residual connection.
    
    This is the standard residual MLP used in most layers of the SLPF model.
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the standard MLP with residual connection.
        
        Args:
            input_dim: Input feature dimension (must equal hidden_dim for residual connection)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, 
            kernel_size=(1, 1), bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, 
            kernel_size=(1, 1), bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with standard residual connection.
        
        Args:
            input_data: Input tensor [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Output with residual connection applied
        """
        # MLP forward pass
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        
        # Add residual connection (input must have same dimensions as hidden)
        hidden = hidden + input_data
        
        return hidden

class SLPF(nn.Module):
    def __init__(self, sensed_locations,unsensed_locations,A,args):
        super(SLPF, self).__init__()
        # attributes
        self.num_locs = args.num_locs
        self.num_unsensed_locs = args.num_unsensed_locs
        self.input_len = args.lag
        self.input_dim = args.input_dim
        self.embed_dim = args.embed_dim
        self.output_len = args.horizon
        self.num_layer = args.num_layer
        self.temp_dim_tid = args.embed_dim
        self.temp_dim_diw = args.embed_dim
        self.time_of_day_size = 288 # 24*12 (5mins per time step)
        self.day_of_week_size = 7 # 7 days in a week

        self.sensed_locations=sensed_locations
        self.unsensed_locations=unsensed_locations
        # embedding banks
        self.full_locs_emb = nn.Parameter(
            torch.empty(self.num_locs, self.embed_dim))
        nn.init.xavier_uniform_(self.full_locs_emb)
        # rank embedding bank
        self.rank_emb = nn.Parameter(
            torch.empty(self.num_locs, self.embed_dim))
        nn.init.xavier_uniform_(self.rank_emb)
        # time-of-day embedding bank
        self.time_in_day_emb = nn.Parameter(
            torch.empty(self.time_of_day_size, self.temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        # day-of-week embedding bank
        self.day_in_week_emb = nn.Parameter(
            torch.empty(self.day_of_week_size, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_MLP = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.time_series_emb_MLP_prime = nn.Conv2d(
            in_channels=self.input_dim * self.output_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        self.hidden_dim = self.embed_dim * 5

        # encoder
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        self.encoder_ = MultiLayerPerceptron_(self.hidden_dim, self.hidden_dim)

        # regression
        self.MLP = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.input_len, kernel_size=(1, 1), bias=True)
        self.MLP_prime = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.stage='adp'

        # adjacency matrix
        self.A=nn.Parameter(A,requires_grad=False)
        self.combine=nn.Linear(self.input_len,1) # only flow rate 

    def forward(self, *args):
        if self.stage=='adp' or self.stage=='forecast':
            # adp stage: X_T shape [bs, T, M, C] batchsize, T=12, M= # of sensed locations, C=4 (data, time-of-day, day-of-week, rank)
            # forecast stage: X_T shape [bs, T, N, C] batchsize, T=12, N = # of all locations= M+M', C=4 (data, time-of-day, day-of-week, rank)
            X_T = args[0]
        else:
            # agg stage: X_T shape [bs, T, N, C] batchsize, T=12, N = # of all locations= M+M', C=4 (data, time-of-day, day-of-week, rank)
            # agg stage: X_MT_prime shape [bs, T', M', C] batchsize, T'=96, M' = # of unsensed locations, C=4 (data, time-of-day, day-of-week, rank)
            X_T=args[0]
            X_MT_prime = args[1]
        if self.stage=='adp':
            locs_emb=self.full_locs_emb[self.sensed_locations]
        else:
            locs_emb = self.full_locs_emb
        unsensed_locs_emb=self.full_locs_emb[self.unsensed_locations]

        # time-of-the-day embedding
        t_i_d_data = X_T[..., 1]
        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]

        # day-of-the-week embedding
        d_i_w_data = X_T[..., 2]
        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]

        tem_emb = []
        tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # rank embedding
        full_rank_data = X_T[..., -1]
        full_rank_list = []
        if self.stage=='adp':
            sensing_ratio=(self.num_locs-self.num_unsensed_locs)/self.num_locs  # adp stage only contains sensed locations
        else:
            sensing_ratio=1.0 # forecast stage only contains sensed and unsensed locations; agg stage contains all locations for the first branch
        full_rank_data = torch.floor(full_rank_data / sensing_ratio).type(torch.LongTensor)  # ensure scale up to the full rank
        for i in range(self.input_len):
            full_rank_list.append(self.rank_emb[full_rank_data[:, i, :].type(torch.LongTensor)])
        full_rank_emb = torch.stack(full_rank_list, dim=-1)
        # aggregate full rank embedding
        full_rank_emb = self.combine(full_rank_emb).squeeze(-1)


        # time series embedding
        X_T[..., 1] = X_T[..., 1] / self.time_of_day_size
        X_T[..., 2] = X_T[..., 2] / self.day_of_week_size
        if self.stage=='adp':
            X_T[..., 3] = X_T[..., 3] / (self.num_locs-self.num_unsensed_locs)
        else:
            X_T[..., 3] = X_T[..., 3] / self.num_locs
        batch_size, _, num_locs, _ = X_T.shape
        X_T = X_T.transpose(1, 2).contiguous()
        X_T = X_T.view(batch_size, num_locs, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_MLP(X_T)

        # node embedding
        locs_emb_feature = []
        locs_emb_feature.append(locs_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + locs_emb_feature + tem_emb+[full_rank_emb.transpose(1, 2).unsqueeze(-1)], dim=1)

        # get high-dimensional feature
        hidden = self.encoder(hidden)
        hidden = self.encoder_(hidden)

        # Node Embedding Enhanced Spatial Transfer Matrix
        if self.stage=='adp':
            dynamic = torch.matmul(full_rank_emb + locs_emb, unsensed_locs_emb.t())
            A_MM_prime = self.A[self.sensed_locations][:, self.unsensed_locations] + dynamic # partial adjacency matrix added with node feature
            hidden = torch.matmul(hidden.transpose(2, 3).squeeze(-2), A_MM_prime).unsqueeze(-1)
            prediction = self.MLP(hidden) #X_M_prime_T
        elif self.stage=='forecast':
            dynamic = torch.matmul(full_rank_emb + locs_emb, self.full_locs_emb[self.sensed_locations].t())
            A_NM = self.A[:, self.sensed_locations] + dynamic # partial adjacency matrix added with node feature
            hidden = torch.matmul(hidden.transpose(2, 3).squeeze(-2), A_NM).unsqueeze(-1)
            prediction = self.MLP_prime(hidden) #X_M_T_prime
        else: #agg step
            dynamic = torch.matmul(full_rank_emb + self.full_locs_emb, unsensed_locs_emb.t())
            A_NM_prime = self.A[:, self.unsensed_locations] + dynamic # partial adjacency matrix added with node feature
            hidden = torch.matmul(hidden.transpose(2, 3).squeeze(-2), A_NM_prime).unsqueeze(-1)
            prediction1 = self.MLP_prime(hidden) # get the final prediction from the history all locations data

            # time-of-the-day embedding
            t_i_d_data = X_MT_prime[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]

            # day-of-the-week embedding
            d_i_w_data = X_MT_prime[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]

            tem_emb = []
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

            # rank embedding
            full_rank_data = X_MT_prime[..., -1]
            full_rank_list = []


            # agg stage second branch only contain sensed locations, need scale up the rank data
            sensing_ratio=(self.num_locs-self.num_unsensed_locs)/self.num_locs
            full_rank_data = torch.floor(full_rank_data / sensing_ratio).type(torch.LongTensor)    # ensure scale up to the full rank

            for i in range(self.input_len):
                full_rank_list.append(self.rank_emb[full_rank_data[:, i, :].type(torch.LongTensor)])
            full_rank_emb = torch.stack(full_rank_list, dim=-1)
            # aggregate full rank embedding
            full_rank_emb = self.combine(full_rank_emb).squeeze(-1)

            # time series embedding
            X_MT_prime[..., 1] = X_MT_prime[..., 1] / self.time_of_day_size
            X_MT_prime[..., 2] = X_MT_prime[..., 2] / self.day_of_week_size
            X_MT_prime[..., 3] = X_MT_prime[..., 3] / (self.num_locs-self.num_unsensed_locs)
            batch_size, _, num_sensed_locs, _ = X_MT_prime.shape
            X_MT_prime = X_MT_prime.transpose(1, 2).contiguous()
            X_MT_prime = X_MT_prime.view(
                batch_size, num_sensed_locs, -1).transpose(1, 2).unsqueeze(-1)
            time_series_emb = self.time_series_emb_MLP_prime(X_MT_prime)

            # node embedding
            locs_emb_feature = []
            locs_emb_feature.append(self.full_locs_emb[self.sensed_locations].unsqueeze(0).expand(
                    batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

            # concate all embeddings
            hidden = torch.cat([time_series_emb] + locs_emb_feature + tem_emb + [full_rank_emb.transpose(1, 2).unsqueeze(-1)],
                               dim=1)

            # get high-dimensional feature
            hidden = self.encoder(hidden)
            hidden = self.encoder_(hidden)
            
            # Node Embedding Enhanced Spatial Transfer Matrix
            dynamic = torch.matmul(full_rank_emb + self.full_locs_emb[self.sensed_locations], unsensed_locs_emb.t())
            A_MM_prime = self.A[self.sensed_locations][:, self.unsensed_locations] + dynamic # partial adjacency matrix added with node feature
            hidden = torch.matmul(hidden.transpose(2, 3).squeeze(-2), A_MM_prime).unsqueeze(-1)
            prediction2 = self.MLP_prime(hidden) # get the final prediction from the future sensed data
            prediction=0.5*prediction1+0.5*prediction2 #X_M_prime_T_prime

        return prediction
