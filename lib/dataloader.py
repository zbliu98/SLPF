import torch
import numpy as np
import torch.utils.data
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
from lib.TrainInits import init_seed
import csv
from torch.utils.data import Dataset


def load_graph(pkl_filename,num_locs):
    if ('BAY' in pkl_filename) or ('LA' in pkl_filename):
        n_vertex = num_locs
        id_list = []
        with open(pkl_filename, 'r') as fp:
            file = csv.reader(fp)
            for line in file:
                start = int(line[0])
                end = int(line[1])
                if start not in id_list:
                    id_list.append(start)
                if end not in id_list:
                    id_list.append(end)
        id_list.sort()
        convert_id = dict()
        for id, value in enumerate(id_list):
            convert_id[value] = id
        with open(pkl_filename, 'r') as fp:
            dist_matrix = np.zeros((n_vertex, n_vertex))
            file = csv.reader(fp)
            for line in file:
                start = convert_id[int(line[0])]
                end = convert_id[int(line[1])]
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
        dist_std = np.std(dist_matrix)

        with open(pkl_filename, 'r') as fp:
            adj = np.zeros((n_vertex, n_vertex))
            file = csv.reader(fp)
            for line in file:
                start = convert_id[int(line[0])]
                end = convert_id[int(line[1])]
                adj[start][end] = np.exp(-(float(line[2]) ** 2) / (dist_std ** 2))
                adj[end][start] = np.exp(-(float(line[2]) ** 2) / (dist_std ** 2))
        for i in range(n_vertex):
            adj[i][i] = 1
        return torch.Tensor(adj)
    else:
        n_vertex = num_locs
        with open(pkl_filename, 'r') as fp:
            dist_matrix = np.zeros((n_vertex, n_vertex))
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
        dist_std = np.std(dist_matrix)

        with open(pkl_filename, 'r') as fp:
            adj = np.zeros((n_vertex, n_vertex))
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                adj[start][end] = np.exp(-(float(line[2]) ** 2) / (dist_std ** 2))
                adj[end][start] = np.exp(-(float(line[2]) ** 2) / (dist_std ** 2))
        for i in range(n_vertex):
            adj[i][i] = 1
        return torch.Tensor(adj)


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler



class MyDataset(Dataset):
    def __init__(self, args,data,scaler,sensed_locations,unsensed_locations,start_idx=0,end_idx=0):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.args=args
        self.scaler = scaler
        self.sensed_locations = sensed_locations
        self.unsensed_locations = unsensed_locations
        self.data = data
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        real_idx = self.start_idx + idx
        sensed_x=self.data[real_idx:real_idx+self.args.lag,self.sensed_locations]
        unsensed_x=self.data[real_idx:real_idx+self.args.lag,self.unsensed_locations]
        unsensed_x[...,0]=self.scaler.inverse_transform(unsensed_x[...,0])  # inverse transform unsensed data to rank
        sensed_y=self.data[real_idx+self.args.lag:real_idx+self.args.lag+self.args.horizon,self.sensed_locations]
        sensed_y[...,0]=self.scaler.inverse_transform(sensed_y[...,0])  # inverse transform sensed data to rank
        unsensed_y=self.data[real_idx+self.args.lag:real_idx+self.args.lag+self.args.horizon,self.unsensed_locations]
        unsensed_y[...,0]=self.scaler.inverse_transform(unsensed_y[...,0])
        return torch.tensor(sensed_x).float().to(self.args.device), torch.tensor(unsensed_x).float().to(self.args.device), torch.tensor(sensed_y).float().to(self.args.device), torch.tensor(unsensed_y).float().to(self.args.device)


def create_rank_index(data,rank_locations):
    data = data.squeeze(-1)
    sensed_data=data[...,rank_locations]
    all_spatial_rank = np.empty(data.shape)
    sensed_spatial_rank = np.empty(sensed_data.shape)
    for i in range(sensed_data.shape[0]):
        sensed_spatial_rank[i] = np.argsort(sensed_data[i])
    all_spatial_rank[..., rank_locations] = sensed_spatial_rank
    return np.expand_dims(all_spatial_rank, axis=-1)

def get_feature(data1, data2, stage, sensed_locations, unsensed_locations, scaler):
    """
    Construct feature tensors for model input, supporting both history (adp) and future stages.

    Args:
        data1: For 'adp', sensed features [bs, T, M, C]; for 'pred', history features.
        data2: For 'adp', unsensed output [bs, T, M', 1]; for 'pred', future output [bs, T', M, 1].
        stage: 'adp' for history, otherwise for prediction/future.
        sensed_locations: Indices of sensed locations.
        unsensed_locations: Indices of unsensed locations.
        scaler: Normalization scaler.

    Returns:
        Tensor containing all features for the given stage.
    """
    if stage == 'adp':  # History feature construction
        his_sensed_features = data1  # [bs, T, M, C]
        his_unsensed_output = data2  # [bs, T, M', 1]
        bs, window, num_sensed_locs, dim = his_sensed_features.shape

        # Create empty tensor for all locations (sensed + unsensed), excluding last feature (time_idx)
        empty = torch.zeros(bs, window, num_sensed_locs + len(unsensed_locations), dim - 1).to('cuda')

        # Fill sensed locations with their features (except time_idx)
        empty[..., sensed_locations, :] = his_sensed_features[..., :4]

        # Inverse transform the main data value for rank calculation
        empty[..., sensed_locations, 0] = scaler.inverse_transform(his_sensed_features[..., 0])

        # Fill unsensed locations with predicted values and repeat day/week features
        empty[..., unsensed_locations, 0] = his_unsensed_output[..., 0]
        empty[..., unsensed_locations, 1] = his_sensed_features[..., :1, 1].repeat(1, 1, len(unsensed_locations))
        empty[..., unsensed_locations, 2] = his_sensed_features[..., :1, 2].repeat(1, 1, len(unsensed_locations))

        # Compute spatial rank index for all locations
        spatial_index = create_rank_index(
            empty[..., :1].detach().cpu().numpy(),
            torch.arange(len(sensed_locations) + len(unsensed_locations))
        )

        # Normalize the main data value
        empty[..., 0] = scaler.transform(empty[..., 0])

        # Add spatial rank index as a feature
        empty[..., 3:4] = torch.from_numpy(spatial_index).to('cuda')

        return empty

    else:  # Future feature construction
        his_sensed_features = data1  # [bs, T, M, C]
        fu_sensed_output = data2     # [bs, T', M, 1]
        bs, window, num_sensed_locs, dim = his_sensed_features.shape
        horizon = fu_sensed_output.shape[1]

        # Create empty tensor for future features (excluding time_idx)
        empty = torch.zeros(bs, horizon, num_sensed_locs, dim - 1).to('cuda')

        # Fill with predicted values for future
        empty[..., 0] = fu_sensed_output[..., 0]

        # Generate day-of-week feature for each future step
        day_feature_start_id = (his_sensed_features[:, -1, 0, 1] + 1) % 288
        day_offset = torch.arange(horizon).to('cuda')
        day_feature = ((day_feature_start_id.unsqueeze(1) + day_offset) % 288).unsqueeze(-1).repeat(1, 1, num_sensed_locs)
        empty[..., 1] = day_feature

        # Generate week feature for each future step
        time_idx_start_id = (his_sensed_features[:, -1, 0, -1] + 1)
        week_feature = (((time_idx_start_id.unsqueeze(1) + day_offset) // 288) % 7).unsqueeze(-1).repeat(1, 1, num_sensed_locs)
        empty[..., 2] = week_feature

        # Compute spatial rank index for future predictions
        spatial_index = create_rank_index(
            fu_sensed_output.detach().cpu().numpy(),
            torch.arange(len(sensed_locations))
        )
        empty[..., 3:4] = torch.from_numpy(spatial_index).to('cuda')

        # Normalize the main data value
        empty[..., 0] = scaler.transform(empty[..., 0])

        return empty




def get_dataloader(args, normalizer='std'):
    # Set random seed for reproducibility
    init_seed(args.seed)

    # Load raw spatio-temporal dataset: shape [T, N, D]
    data = load_st_dataset(args.dataset)
    feature_list = [data]

    # Add time-of-day feature: [T, N, 1]
    time_ind = np.arange(data.shape[0]) % 288
    time_in_day = np.tile(time_ind, [1, data.shape[1], 1]).transpose((2, 1, 0))
    feature_list.append(time_in_day)

    # Add day-of-week feature: [T, N, 1]
    dow = (np.arange(data.shape[0]) // 288) % 7
    dow_tiled = np.tile(dow, [1, data.shape[1], 1]).transpose((2, 1, 0))
    feature_list.append(dow_tiled)

    # Calculate mean for each location and use as weights for selection
    mean_data = np.mean(data[..., 0], axis=0)
    weights = mean_data / np.sum(mean_data)
    location_list = np.arange(args.num_locs)

    # Weighted random selection of weighted locations
    sensed_locations = np.random.choice(
        location_list, args.num_locs - args.num_unsensed_locs, p=weights, replace=False)

    # sensed_locations = np.random.choice(
    #     location_list, args.num_locs - args.num_unsensed_locs, replace=False)
    
    sensed_locations = np.sort(sensed_locations)
    unsensed_locations = np.sort(list(set(location_list) - set(sensed_locations)))
    print('unsensed_locations: ', unsensed_locations)

    # Add spatial rank index feature
    rank_index = create_rank_index(data, sensed_locations)
    feature_list.append(rank_index)

    # Concatenate all features along the last dimension
    data = np.concatenate(feature_list, axis=-1)

    # Add time index feature: [T, N, 1]
    time_idx = np.arange(data.shape[0])
    time_idx = np.tile(time_idx, [1, data.shape[1], 1]).transpose((2, 1, 0))
    data = np.concatenate([data, time_idx], axis=-1)

    # Normalize using only sensed locations' data
    _, scaler = normalize_dataset(data[:, sensed_locations, 0], normalizer, args.column_wise)
    data[..., 0] = scaler.transform(data[..., 0])

    # Calculate dataset split lengths
    total_len = data.shape[0] - (args.lag + args.horizon - 1)
    train_len = int(total_len * (1 - args.val_ratio - args.test_ratio))
    val_len = int(total_len * args.val_ratio)
    test_len = total_len - train_len - val_len

    # Define split indices
    train_start, train_end = 0, train_len
    val_start, val_end = train_end, train_end + val_len
    test_start, test_end = val_end, val_end + test_len

    # Create dataset objects for each split
    train_dataset = MyDataset(args, data, scaler, sensed_locations, unsensed_locations, train_start, train_end)
    val_dataset = MyDataset(args, data, scaler, sensed_locations, unsensed_locations, val_start, val_end)
    test_dataset = MyDataset(args, data, scaler, sensed_locations, unsensed_locations, test_start, test_end)

    # Create DataLoader objects
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)

    # Load and normalize adjacency graph
    graph = load_graph('../data/' + args.dataset + '/' + args.dataset + '.csv', args.num_locs)

    return train_loader, val_loader, test_loader, scaler, sensed_locations, unsensed_locations, graph


