import torch.utils.data as data
import numpy as np
from utils import process_feat

class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False, modality=None):
        self.args = args
        self.modality = modality if modality is not None else args.modality
        self.corrupt_modality = args.corrupt_modality
        self.corruption_percentage = args.corruption_percentage

        # Hardcoded feature dimensions
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

        self._parse_list()

        num_samples = len(self.list)
        if num_samples % self.group_size != 0:
            raise ValueError("Number of samples is not a multiple of 10.")

        self.num_groups = num_samples // self.group_size
        np.random.seed(0)
        self.group_permutation = np.random.permutation(self.num_groups)
        self.inv_group_permutation = np.empty(self.num_groups, dtype=np.int64)
        self.inv_group_permutation[self.group_permutation] = np.arange(self.num_groups)

    def _parse_list(self):
        """
        Reads the appropriate file list(s) depending on self.modality
        and populates self.list (and possibly self.audio_list, self.flow_list).
        """
        if self.modality == 'AUDIO':
            self.list = list(open(self.audio_list_file))
            # Expand each audio file 10 times to match RGB sample count if needed
            audio_expanded = []
            for line in self.list:
                line = line.strip()
                audio_expanded.extend([line] * 10)
            self.list = audio_expanded

        elif self.modality == 'RGB':
            self.list = list(open(self.rgb_list_file))

        elif self.modality == 'FLOW':
            self.list = list(open(self.flow_list_file))

        elif self.modality == 'MIX':
            self.list = list(open(self.rgb_list_file))
            self.flow_list = list(open(self.flow_list_file))

        elif self.modality == 'MIX2':
            self.list = list(open(self.rgb_list_file))
            self.audio_list = list(open(self.audio_list_file))

        elif self.modality == 'MIX3':
            self.list = list(open(self.flow_list_file))
            self.audio_list = list(open(self.audio_list_file))

        elif self.modality == 'MIX_ALL':
            self.list = list(open(self.rgb_list_file))
            self.flow_list = list(open(self.flow_list_file))
            self.audio_list = list(open(self.audio_list_file))

        else:
            raise ValueError('Modality is wrong!')

    def _trim_features(self, features1, features2, extend_audio=False):
        """
        Ensures features1 and features2 have the same temporal dimension by either
        trimming or duplicating frames (if extend_audio=True).
        """
        len1, len2 = features1.shape[0], features2.shape[0]
        if len1 == len2:
            return features1, features2

        if len1 > len2:
            if extend_audio:
                diff = len1 - len2
                last_frame = features2[-1:]
                repeated_frames = np.repeat(last_frame, diff, axis=0)
                features2 = np.concatenate((features2, repeated_frames), axis=0)
            else:
                features1 = features1[:len2]
        else:
            if extend_audio:
                features2 = features2[:len1]
            else:
                features2 = features2[:len1]

        return features1, features2

    def corrupt_features(self, features, index, branch_modality=None):
        """
        Zeroes out 'features' if:
          1) self.corrupt_modality is set AND
          2) this sample's group is in the fraction to corrupt (group_rank < cutoff) AND
          3) branch_modality matches self.corrupt_modality (when branch_modality is specified).
        
        If branch_modality is None, fallback to older column-based logic for single or 
        concatenated features. 
        """
        if self.corrupt_modality is None or self.corruption_percentage <= 0.0:
            return features

        # Determine if this particular group is in the corruption range
        group_id = index // self.group_size
        group_rank = self.inv_group_permutation[group_id]
        cutoff = int(self.corruption_percentage * self.num_groups)
        if group_rank >= cutoff:
            return features

        # If we know exactly which branch we're corrupting (RGB or AUDIO), zero out entirely if it matches
        if branch_modality is not None:
            if branch_modality == self.corrupt_modality:
                features[:] = 0
            return features

        # Otherwise (branch_modality=None), use older logic for single-modality or multi-concat
        if self.modality == 'RGB' and self.corrupt_modality == 'RGB':
            features[:] = 0
        elif self.modality == 'FLOW' and self.corrupt_modality == 'FLOW':
            features[:] = 0
        elif self.modality == 'AUDIO' and self.corrupt_modality == 'AUDIO':
            features[:] = 0
        elif self.modality == 'MIX':
            if self.corrupt_modality == 'RGB':
                features[:, :self.D_rgb] = 0
            elif self.corrupt_modality == 'FLOW':
                features[:, self.D_rgb:self.D_rgb+self.D_flow] = 0
        elif self.modality == 'MIX2':
            if self.corrupt_modality == 'RGB':
                features[:, :self.D_rgb] = 0
            elif self.corrupt_modality == 'AUDIO':
                features[:, self.D_rgb:self.D_rgb+self.D_audio] = 0
        elif self.modality == 'MIX3':
            if self.corrupt_modality == 'FLOW':
                features[:, :self.D_flow] = 0
            elif self.corrupt_modality == 'AUDIO':
                features[:, self.D_flow:self.D_flow+self.D_audio] = 0
        elif self.modality == 'MIX_ALL':
            if self.corrupt_modality == 'RGB':
                features[:, :self.D_rgb] = 0
            elif self.corrupt_modality == 'FLOW':
                features[:, self.D_rgb:self.D_rgb+self.D_flow] = 0
            elif self.corrupt_modality == 'AUDIO':
                features[:, self.D_rgb+self.D_flow:self.D_rgb+self.D_flow+self.D_audio] = 0

        return features

    def __getitem__(self, index):
        filename = self.list[index].strip()
        label = 0.0 if self.normal_flag in filename else 1.0

        # ----------------------------------
        # Single-Modality Cases (RGB, AUDIO)
        # ----------------------------------
        if self.modality == 'AUDIO':
            features = np.load(filename).astype(np.float32)
            if self.test_mode:
                return features, filename
            else:
                features = process_feat(features, self.max_seqlen, is_random=False)
                return features, label, filename

        elif self.modality == 'RGB':
            features = np.load(filename).astype(np.float32)
            # We pass branch_modality=None to use older logic 
            # (since it's a single-modality dataset).
            features = self.corrupt_features(features, index)  
            if self.transform is not None:
                features = self.transform(features)
            if self.test_mode:
                return features, filename
            else:
                features = process_feat(features, self.max_seqlen, is_random=False)
                return features, label, filename

        # -----------------------
        # MIX2 = RGB + AUDIO
        # -----------------------
        elif self.modality == 'MIX2':
            # Load RGB (2048-D)
            features1 = np.load(filename).astype(np.float32)
            # Load Audio (2048-D in your config)
            audio_filename = self.audio_list[index // 10].strip()
            features2 = np.load(audio_filename).astype(np.float32)

            # Ensure matching temporal length
            features1, features2 = self._trim_features(features1, features2, extend_audio=True)

            # Option A: Corrupt each branch only if requested 
            # (no need to call on the other if we only want to corrupt one).
            if self.corrupt_modality == 'RGB':
                features1 = self.corrupt_features(features1, index, branch_modality='RGB')
            elif self.corrupt_modality == 'AUDIO':
                features2 = self.corrupt_features(features2, index, branch_modality='AUDIO')

            # If you'd prefer to always call both, you could do:
            #
            # features1 = self.corrupt_features(features1, index, branch_modality='RGB')
            # features2 = self.corrupt_features(features2, index, branch_modality='AUDIO')
            #
            # That would still only zero out whichever matches self.corrupt_modality.

            # Optional transform
            if self.transform is not None:
                features1 = self.transform(features1)
                features2 = self.transform(features2)

            # Return them either separately or concatenated
            if self.args.single_branch:
                # Return them separately so we can feed them into different branches
                if self.test_mode:
                    return features1, features2, filename
                else:
                    features1 = process_feat(features1, self.max_seqlen, is_random=False)
                    features2 = process_feat(features2, self.max_seqlen, is_random=False)
                    return features1, features2, label, filename
            else:
                # Concatenate along feature dimension
                features = np.concatenate((features1, features2), axis=1)
                if self.test_mode:
                    return features, filename
                else:
                    features = process_feat(features, self.max_seqlen, is_random=False)
                    return features, label, filename

        else:
            # For other modalities (MIX, MIX3, MIX_ALL), handle similarly or raise
            raise ValueError('Modality is not implemented in this snippet.')

    def __len__(self):
        return len(self.list)
