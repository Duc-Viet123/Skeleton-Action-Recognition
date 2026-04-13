import numpy as np
import torch
from torch.utils.data import Dataset


class FeederPretrain(Dataset):

    def __init__(
        self,
        data_path,
        split='train',
        window_size=64,
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        p_interval=(0.5, 1.0),
        random_scale=False,
        random_flip=False,
        random_noise=False,
        noise_std=0.01,
        temporal_mask=False,
        temporal_mask_ratio=0.1,  
        spatial_mask=False,
        spatial_mask_ratio=0.1,      
        normalization=False,
        vel=False,
        bone=False,
        use_mixup=False,
        mixup_alpha=0.2,

        contrastive_mode=True,
    ):
        self.data = np.load(data_path)
        self.split = split
        self.window_size = window_size
        self.debug = debug
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.p_interval = p_interval
        self.random_scale = random_scale
        self.random_flip = random_flip
        self.random_noise = random_noise
        self.noise_std = noise_std
        self.temporal_mask = temporal_mask
        self.temporal_mask_ratio = temporal_mask_ratio
        self.spatial_mask = spatial_mask
        self.spatial_mask_ratio = spatial_mask_ratio
        self.contrastive_mode = contrastive_mode

        assert self.data.ndim == 5, \
            f"Expected (N,C,T,V,M), got {self.data.shape}"

        if self.debug:
            self.data = self.data[:200]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

        print(f"[FeederPretrain] Loaded {split}: {self.data.shape}")

        if split == 'train':
            aug_list = []
            for name, flag in [
                ('random_choose', random_choose),
                ('random_shift', random_shift),
                ('random_move', random_move),
                ('random_rot', random_rot),
                ('random_scale', random_scale),
                ('random_flip', random_flip),
                ('random_noise', random_noise),
                ('temporal_mask', temporal_mask),
                ('spatial_mask', spatial_mask),
            ]:
                if flag:
                    aug_list.append(name)

            if aug_list:
                print(f"[FeederPretrain] Augmentations: {', '.join(aug_list)}")

            if contrastive_mode:
                print("[FeederPretrain] Contrastive mode: Returning 2 independent augmented views")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x_orig = self.data[idx].copy() 

        if self.contrastive_mode:
            view1 = self._apply_augmentations(x_orig.copy())
            view2 = self._apply_augmentations(x_orig.copy())

            view1, index_t1 = self.temporal_crop_resize(view1)
            view2, _ = self.temporal_crop_resize(view2)

            return (
                torch.from_numpy(view1).float(),
                torch.from_numpy(view2).float(),
                torch.from_numpy(index_t1).long(),
                idx
            )
        else:
            x = self._apply_augmentations(x_orig)
            x, index_t = self.temporal_crop_resize(x)
            y = int(self.label[idx])
            return (
                torch.from_numpy(x).float(),
                torch.from_numpy(index_t).long(),
                y,
                idx
            )

    # AUGMENTATION 

    def _apply_augmentations(self, x):
        if self.split != 'train':
            return x

        if self.random_choose:
            x = self._random_choose(x)
        if self.random_shift:
            x = self._random_shift(x)
        if self.random_move:
            x = self._random_move(x)
        if self.random_rot:
            x = self._random_rot(x)
        if self.random_scale:
            x = self._random_scale(x)
        if self.random_flip:
            x = self._random_flip(x)
        if self.temporal_mask:
            x = self._temporal_mask(x)
        if self.spatial_mask:
            x = self._spatial_mask(x)
        if self.random_noise:
            x = self._random_noise(x)

        return x

    # BASIC AUGS 

    def _random_choose(self, x):
        C, T, V, M = x.shape
        p = np.random.uniform(*self.p_interval)
        num_frames = max(int(T * p), 1)
        if num_frames < T:
            start = np.random.randint(0, T - num_frames + 1)
            x = x[:, start:start + num_frames]
        return x

    def _random_shift(self, x):
        C, T, V, M = x.shape
        shift = np.random.randint(-T // 10, T // 10 + 1)
        return np.roll(x, shift, axis=1) if shift != 0 else x

    def _random_move(self, x):
        x[0] += np.random.uniform(-0.1, 0.1)
        x[1] += np.random.uniform(-0.1, 0.1)
        return x

    def _random_rot(self, x):
        theta = np.random.uniform(-np.pi / 18, np.pi / 18)
        cos, sin = np.cos(theta), np.sin(theta)
        x0 = x[0] * cos - x[1] * sin
        x1 = x[0] * sin + x[1] * cos
        x[0], x[1] = x0, x1
        return x

    def _random_scale(self, x):
        x[:2] *= np.random.uniform(0.8, 1.2)
        return x

    def _random_flip(self, x):
        if np.random.rand() > 0.5:
            x[0] = -x[0]
        return x

    def _random_noise(self, x):
        return x + np.random.normal(0, self.noise_std, x.shape)

    # MASKING

    def _sample_ratio(self, ratio):
        if isinstance(ratio, (list, tuple)):
            r = np.random.uniform(ratio[0], ratio[1])
        else:
            r = ratio
        return float(np.clip(r, 0.0, 0.999))

    def _temporal_mask(self, x):
        C, T, V, M = x.shape
        ratio = self._sample_ratio(self.temporal_mask_ratio)
        num_mask = int(T * ratio)

        if 0 < num_mask < T:
            idx = np.random.choice(T, num_mask, replace=False)
            x[:, idx] = 0
        return x

    def _spatial_mask(self, x):
        C, T, V, M = x.shape
        ratio = self._sample_ratio(self.spatial_mask_ratio)
        num_mask = int(V * ratio)

        if 0 < num_mask < V:
            idx = np.random.choice(V, num_mask, replace=False)
            x[:, :, idx] = 0
        return x

    #TEMPORAL RESIZE

    def temporal_crop_resize(self, x):
        C, T, V, M = x.shape

        if T == self.window_size:
            return x, np.arange(T)

        if T > self.window_size:
            start = np.random.randint(0, T - self.window_size + 1)
            x = x[:, start:start + self.window_size]
            index_t = np.arange(start, start + self.window_size)
        else:
            pad = self.window_size - T
            x = np.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
            index_t = np.concatenate([np.arange(T), np.full(pad, T - 1)])

        return x, index_t