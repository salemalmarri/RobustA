import torch.utils.data as data
import numpy as np
from utils import process_feat

class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False, modality=None):
        self.args = args
        self.modality = modality if modality is not None else args.modality
        cm = args.corrupt_modality
        if isinstance(cm, list):
            self.corrupt_modality = cm[0] if cm else None
        else:
            self.corrupt_modality = cm
        self.corruption_percentage = args.corruption_percentage

        # Hardcoded feature dimensions
        self.D_rgb = 2048
        self.D_flow = 2048
        self.D_audio = 2048  # Audio also projected to 2048 in your setup

        self.group_size = 10
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.normal_flag = '_label_A'

        # Normal lists (RGB, FLOW, AUDIO)
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.flow_list_file = args.test_flow_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.flow_list_file = args.flow_list
            self.audio_list_file = args.audio_list

        # If we plan to corrupt RGB files by loading a separate (pre-corrupted) list,
        # define a place to store that list. We'll parse it only if test_mode & corrupt_modality=='RGB'.
        self.corrupted_list_file = None
        if test_mode and (self.corrupt_modality == 'RGB') and (self.corruption_percentage > 0.0):
            # Suppose we've added --test-corrupted-list to option.py
            if hasattr(args, 'test_corrupted_list'):
                self.corrupted_list_file = args.test_corrupted_list

        # ---------------------------
        # NEW: Support for Pre-Corrupted AUDIO
        # ---------------------------
        self.corrupted_audio_list_file = None
        if test_mode and (self.corrupt_modality == 'AUDIO') and (self.corruption_percentage > 0.0):
            # Suppose we've added --test_corrupted_audio_list to option.py
            if hasattr(args, 'test_corrupted_audio_list'):
                self.corrupted_audio_list_file = args.test_corrupted_audio_list
        # ---------------------------

        self._parse_list()

        num_samples = len(self.list)
        if num_samples % self.group_size != 0:
            raise ValueError("Number of samples is not a multiple of 10.")

        self.num_groups = num_samples // self.group_size

        # ----------------------------------------------
        # Only fix the seed for test mode
        # ----------------------------------------------
        if test_mode:
            np.random.seed(0)

        self.group_permutation = np.random.permutation(self.num_groups)
        self.inv_group_permutation = np.empty(self.num_groups, dtype=np.int64)
        self.inv_group_permutation[self.group_permutation] = np.arange(self.num_groups)

    def _parse_list(self):
        """
        Reads the appropriate file list(s) depending on self.modality
        and populates self.list (and possibly self.audio_list, self.flow_list).
        Also reads a corrupted_list if we are in test_mode and corrupt_modality=RGB.
        """
        if self.modality == 'AUDIO':
            self.list = list(open(self.audio_list_file))
            # Expand each audio file 10 times to match sample count if needed
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

        # Now, if test_mode + corrupt_modality=RGB + corruption_percentage>0,
        # we read the "pre-corrupted" list from self.corrupted_list_file
        if self.test_mode and self.corrupted_list_file is not None:
            self.corrupted_list = list(open(self.corrupted_list_file))
            if len(self.corrupted_list) != len(self.list):
                raise ValueError(
                    f"Corrupted list (len={len(self.corrupted_list)}) "
                    f"does not match main list (len={len(self.list)})."
                )

        # ---------------------------
        # NEW: If test_mode + corrupt_modality=AUDIO + corruption_percentage>0,
        # read the "pre-corrupted" audio list
        # ---------------------------
        if self.test_mode and self.corrupted_audio_list_file is not None:
            self.corrupted_audio_list = list(open(self.corrupted_audio_list_file))

            # Expand each corrupted audio file 10 times to match sample count
            audio_corrupted_expanded = []
            for line in self.corrupted_audio_list:
                line = line.strip()
                audio_corrupted_expanded.extend([line] * 10)
            self.corrupted_audio_list = audio_corrupted_expanded

            if len(self.corrupted_audio_list) != len(self.list):
                raise ValueError(
                    f"Corrupted audio list (len={len(self.corrupted_audio_list)}) "
                    f"does not match main list (len={len(self.list)})."
                )
        # ---------------------------

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
          3) branch_modality matches self.corrupt_modality (if branch_modality != None).

        If branch_modality is None, use older column-based logic for single or
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

        # If branch_modality is specified, zero out only if it matches corrupt_modality
        if branch_modality is not None:
            if branch_modality == self.corrupt_modality:
                features[:] = 0
            return features

        # Otherwise (branch_modality=None), fallback to older logic
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
                features[:, self.D_rgb : self.D_rgb + self.D_flow] = 0
        elif self.modality == 'MIX2':
            if self.corrupt_modality == 'RGB':
                features[:, :self.D_rgb] = 0
            elif self.corrupt_modality == 'AUDIO':
                features[:, self.D_rgb : self.D_rgb + self.D_audio] = 0
        elif self.modality == 'MIX3':
            if self.corrupt_modality == 'FLOW':
                features[:, :self.D_flow] = 0
            elif self.corrupt_modality == 'AUDIO':
                features[:, self.D_flow : self.D_flow + self.D_audio] = 0
        elif self.modality == 'MIX_ALL':
            if self.corrupt_modality == 'RGB':
                features[:, :self.D_rgb] = 0
            elif self.corrupt_modality == 'FLOW':
                features[:, self.D_rgb : self.D_rgb + self.D_flow] = 0
            elif self.corrupt_modality == 'AUDIO':
                features[:, self.D_rgb + self.D_flow : self.D_rgb + self.D_flow + self.D_audio] = 0

        return features

    def __getitem__(self, index):
        """
        If corrupt_modality=RGB, we load from a pre-corrupted list for "corrupted" groups
        (test mode only), skipping on-the-fly zeroing for RGB. If corrupt_modality=AUDIO,
        we load the normal file and do the usual zero-out approach (unless you implement
        pre-corrupted audio logic below).
        """
        filename = self.list[index].strip()
        label = 0.0 if self.normal_flag in filename else 1.0

        # Figure out which group this belongs to
        group_id = index // self.group_size
        group_rank = self.inv_group_permutation[group_id]
        cutoff = int(self.corruption_percentage * self.num_groups)

        # ---------------------------------------------------
        # 1) If corrupt_modality=RGB: load pre-corrupted file if group is "marked"
        # ---------------------------------------------------
        use_pre_corrupted_rgb = (
            self.test_mode
            and self.corrupt_modality == 'RGB'
            and self.corruption_percentage > 0.0
            and hasattr(self, 'corrupted_list')
            and group_rank < cutoff
        )

        if use_pre_corrupted_rgb:
            filename = self.corrupted_list[index].strip()

        # ---------------------------------------------------
        # NEW: If corrupt_modality=AUDIO: load pre-corrupted file if group is "marked"
        # ---------------------------------------------------
        use_pre_corrupted_audio = (
            self.test_mode
            and self.corrupt_modality == 'AUDIO'
            and self.corruption_percentage > 0.0
            and hasattr(self, 'corrupted_audio_list')
            and group_rank < cutoff
        )
        # ---------------------------------------------------

        # ----------------------------------
        # Single-Modality Cases
        # ----------------------------------
        if self.modality == 'AUDIO':
            # If we decided to use pre-corrupted audio for this group, pick from self.corrupted_audio_list
            if use_pre_corrupted_audio:
                filename = self.corrupted_audio_list[index].strip()

            features = np.load(filename).astype(np.float32)

            # If we physically replaced it above, you could skip zeroing
            if not use_pre_corrupted_audio:
                features = self.corrupt_features(features, index, branch_modality='AUDIO')

            if self.test_mode:
                return features, filename
            else:
                features = process_feat(features, self.max_seqlen, is_random=False)
                return features, label, filename

        elif self.modality == 'RGB':
            features = np.load(filename).astype(np.float32)

            # If corrupt_modality=RGB, we skip zero-out because we already loaded
            # a physically corrupted file. If you DO want both, remove the condition below:
            if self.corrupt_modality != 'RGB':
                features = self.corrupt_features(features, index, branch_modality='RGB')

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
            # The main list is presumably the RGB file
            features1 = np.load(filename).astype(np.float32)

            # The audio list is aligned in blocks of 10
            audio_filename = self.audio_list[index // 10].strip()

            # NEW: If we decided to use pre-corrupted audio for this group, pick from self.corrupted_audio_list
            if use_pre_corrupted_audio:
                audio_filename = self.corrupted_audio_list[index // 10].strip()

            features2 = np.load(audio_filename).astype(np.float32)

            # Ensure matching temporal length
            features1, features2 = self._trim_features(features1, features2, extend_audio=True)

            # If corrupt_modality=RGB => we might have loaded a physically corrupted RGB above
            # so let's skip zero-out for features1 if that's the case.
            # If corrupt_modality=AUDIO => on-the-fly zeroing for features2.
            if use_pre_corrupted_rgb:
                pass  # Already physically corrupted
            else:
                features1 = self.corrupt_features(features1, index, branch_modality='RGB')

            if use_pre_corrupted_audio:
                pass  # Already physically corrupted audio
            else:
                if self.corrupt_modality == 'AUDIO':
                    features2 = self.corrupt_features(features2, index, branch_modality='AUDIO')

            # Optional transform
            if self.transform is not None:
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
                # Concatenate along feature dimension
                features_cat = np.concatenate((features1, features2), axis=1)
                if self.test_mode:
                    return features_cat, filename
                else:
                    features_cat = process_feat(features_cat, self.max_seqlen, is_random=False)
                    return features_cat, label, filename

        else:
            # For other modalities (MIX, MIX3, MIX_ALL) or anything not implemented
            raise ValueError('Modality is not implemented in this snippet.')

    def __len__(self):
        return len(self.list)
