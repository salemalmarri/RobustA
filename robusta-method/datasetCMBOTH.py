import torch.utils.data as data
import numpy as np
from utils import process_feat

class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False, modality=None):
        self.args = args
        self.modality = modality if modality is not None else args.modality

        # NEW: Allow multiple corruptions
        if isinstance(args.corrupt_modality, list):
            self.corrupt_modalities = args.corrupt_modality
        elif isinstance(args.corrupt_modality, str):
            self.corrupt_modalities = [args.corrupt_modality]
        else:
            self.corrupt_modalities = []

        self.corruption_percentage = args.corruption_percentage

        self.D_rgb = 2048
        self.D_flow = 2048
        self.D_audio = 2048

        self.group_size = 10
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.normal_flag = '_label_A'

        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.flow_list_file = args.test_flow_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.flow_list_file = args.flow_list
            self.audio_list_file = args.audio_list

        self.corrupted_list_file = args.test_corrupted_list if test_mode and 'RGB' in self.corrupt_modalities else None
        self.corrupted_audio_list_file = args.test_corrupted_audio_list if test_mode and 'AUDIO' in self.corrupt_modalities else None

        self._parse_list()

        num_samples = len(self.list)
        if num_samples % self.group_size != 0:
            raise ValueError("Number of samples is not a multiple of 10.")

        self.num_groups = num_samples // self.group_size

        if test_mode:
            np.random.seed(0)

        self.group_permutation = np.random.permutation(self.num_groups)
        self.inv_group_permutation = np.empty(self.num_groups, dtype=np.int64)
        self.inv_group_permutation[self.group_permutation] = np.arange(self.num_groups)

    def _parse_list(self):
        if self.modality == 'MIX2':
            self.list = list(open(self.rgb_list_file))
            self.audio_list = list(open(self.audio_list_file))
        else:
            raise ValueError('Only MIX2 modality is implemented.')

        if self.corrupted_list_file:
            self.corrupted_list = list(open(self.corrupted_list_file))
        if self.corrupted_audio_list_file:
            corrupted_audio = list(open(self.corrupted_audio_list_file))
            expanded = []
            for line in corrupted_audio:
                expanded.extend([line.strip()] * 10)
            self.corrupted_audio_list = expanded

    def corrupt_features(self, features, index, branch_modality=None):
        if not self.corrupt_modalities or self.corruption_percentage <= 0.0:
            return features

        group_id = index // self.group_size
        group_rank = self.inv_group_permutation[group_id]
        cutoff = int(self.corruption_percentage * self.num_groups)
        if group_rank >= cutoff:
            return features

        if branch_modality in self.corrupt_modalities:
            features[:] = 0

        return features

    def __getitem__(self, index):
        filename = self.list[index].strip()
        label = 0.0 if self.normal_flag in filename else 1.0

        group_id = index // self.group_size
        group_rank = self.inv_group_permutation[group_id]
        cutoff = int(self.corruption_percentage * self.num_groups)

        use_corrupted_rgb = 'RGB' in self.corrupt_modalities and self.test_mode and group_rank < cutoff and hasattr(self, 'corrupted_list')
        use_corrupted_audio = 'AUDIO' in self.corrupt_modalities and self.test_mode and group_rank < cutoff and hasattr(self, 'corrupted_audio_list')

        if use_corrupted_rgb:
            filename = self.corrupted_list[index].strip()
        features1 = np.load(filename).astype(np.float32)

        audio_filename = self.audio_list[index // 10].strip()
        if use_corrupted_audio:
            audio_filename = self.corrupted_audio_list[index // 10].strip()
        features2 = np.load(audio_filename).astype(np.float32)

        features1, features2 = self._trim_features(features1, features2, extend_audio=True)

        if not use_corrupted_rgb:
            features1 = self.corrupt_features(features1, index, branch_modality='RGB')
        if not use_corrupted_audio:
            features2 = self.corrupt_features(features2, index, branch_modality='AUDIO')

        if self.transform:
            features1 = self.transform(features1)
            features2 = self.transform(features2)

        if self.args.single_branch:
            if self.test_mode:
                return features1, features2, filename
            else:
                features1 = process_feat(features1, self.max_seqlen, is_random=False)
                features2 = process_feat(features2, self.max_seqlen, is_random=False)
                return features1, features2, label, filename
        else:
            features_cat = np.concatenate((features1, features2), axis=1)
            if self.test_mode:
                return features_cat, filename
            else:
                features_cat = process_feat(features_cat, self.max_seqlen, is_random=False)
                return features_cat, label, filename

    def _trim_features(self, features1, features2, extend_audio=False):
        len1, len2 = features1.shape[0], features2.shape[0]
        if len1 == len2:
            return features1, features2
        if len1 > len2:
            if extend_audio:
                last_frame = features2[-1:]
                features2 = np.concatenate((features2, np.repeat(last_frame, len1 - len2, axis=0)), axis=0)
            else:
                features1 = features1[:len2]
        else:
            features2 = features2[:len1]
        return features1, features2

    def __len__(self):
        return len(self.list)
