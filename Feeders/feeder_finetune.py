import numpy as np
import torch
from torch.utils.data import Dataset

class FeederMultiClass(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        window_size=64,
        p_interval=[0.85, 1.15],         
        random_choose=True,
        random_shift=True,
        random_move=False,            
        random_rot=False,               
        random_spatial_flip=True,
        random_temporal_flip=True,       
        temporal_mask=True,
        temporal_dropout=0.03,
        vel=False,
        bone=False,
        normalization=False,             
        debug=False
    ):
        # Load data
        try:
            self.data = np.load(data_path)     
            self.label = np.load(label_path)    
        except Exception as e:
            print(f"Error loading npy files: {e}")
            raise

        self.window_size = window_size
        self.p_interval = p_interval
        self.vel = vel
        self.bone = bone
        self.normalization = normalization
        self.debug = debug
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.random_spatial_flip = random_spatial_flip
        self.random_temporal_flip = random_temporal_flip
        self.temporal_mask = temporal_mask
        self.temporal_dropout = temporal_dropout

        if debug:
            self.data = self.data[:200]
            self.label = self.label[:200]
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.label_names = ["Normal", "Fall", "Fight"]
        self.bone_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4),   
            (1, 5), (5, 6), (6, 7),           
            (1, 8), (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),      
            (0, 15), (0, 16), (15, 17)        
        ]
        unique, counts = np.unique(self.label, return_counts=True)
        print(f"\n[Feeder] Dataset Loaded: {self.N} samples")
        for idx, count in zip(unique, counts):
            print(f"  - {self.label_names[idx]:7s}: {count:4d} samples")
        print("-" * 30)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data_np = self.data[index].copy() 
        label = int(self.label[index])
        data_np = self.temporal_crop_resize(data_np)
        if self.random_choose:
            data_np = self._random_choose(data_np)
        if self.random_shift:
            data_np = self._random_shift(data_np)
        if self.random_spatial_flip:
            data_np = self._spatial_flip(data_np)
        if self.random_temporal_flip and np.random.rand() < 0.5:
            data_np = data_np[:, ::-1, :, :].copy()
        if self.temporal_mask:
            data_np = self._temporal_mask(data_np)
        joint = data_np.copy()

        bone = np.zeros_like(joint)
        if self.bone:
            for i, (v1, v2) in enumerate(self.bone_pairs):
                if i < self.V:
                    bone[:, :, i, :] = joint[:, :, v1, :] - joint[:, :, v2, :]

        motion = np.zeros_like(joint)
        if self.vel:
            motion[:, :-1, :, :] = joint[:, 1:, :, :] - joint[:, :-1, :, :]

        return {
            'joint': torch.from_numpy(joint).float(),
            'bone': torch.from_numpy(bone).float(),
            'motion': torch.from_numpy(motion).float()
        }, label

    def temporal_crop_resize(self, x):
        C, T, V, M = x.shape
        if T == self.window_size: return x
        
        scale = np.random.uniform(self.p_interval[0], self.p_interval[1])
        target_len = int(self.window_size / scale)
        
        if T > target_len:
            start = np.random.randint(0, T - target_len + 1)
            x = x[:, start:start + target_len, :, :]
        
        indices = np.linspace(0, x.shape[1] - 1, self.window_size).astype(int)
        return x[:, indices, :, :]

    def _random_choose(self, x):
        if np.random.rand() < 0.5:
            C, T, V, M = x.shape
            new_size = np.random.randint(int(T*0.9), T+1)
            start = np.random.randint(0, T - new_size + 1)
            x = x[:, start:start+new_size, :, :]
            indices = np.linspace(0, x.shape[1]-1, T).astype(int)
            x = x[:, indices, :, :]
        return x

    def _random_shift(self, x):
        C, T, V, M = x.shape
        shift = np.random.randint(-2, 3)
        return np.roll(x, shift, axis=1)

    def _spatial_flip(self, x):
        if np.random.rand() < 0.5:
            x[0] = 1.0 - x[0] 
            lr_pairs = [(2,5), (3,6), (4,7), (9,12), (10,13), (11,14), (15,16)]
            for l, r in lr_pairs:
                temp = x[:, :, l, :].copy()
                x[:, :, l, :] = x[:, :, r, :]
                x[:, :, r, :] = temp
        return x

    def _temporal_mask(self, x):
        if np.random.rand() < 0.4:
            C, T, V, M = x.shape
            mask_len = np.random.randint(4, 12)
            start = np.random.randint(0, T - mask_len)
            x[:, start:start+mask_len, :, :] = 0
        return x
