from torch.utils.data import Dataset
import torch
import random

class VideoCaptionDataset(Dataset):
    def __init__(self, total_data, video, caption, train=True):
        # video = [total_data, 80, 4096]
        # caption = [total_data, captions_num, sequence_length]
        super(VideoCaptionDataset, self).__init__()
        self._video = video
        self._caption = caption
        self._total_data = total_data
        self._train = train
        if self._train:
            self._caption_len = [len(one_data) for one_data in self._caption]
            self._total_captions = sum(self._caption_len)

    def __len__(self):
        return self._total_data

    def __getitem__(self, i):
        if self._train:
            correct_caption, wrong_caption = self._rand_two_captions(i)
            return self._video[i], correct_caption, wrong_caption
        else:
            return self._video[i], self._caption[i]

    def _rand_two_captions(self, correct_i):
        correct_caption = random.sample(self._caption[correct_i], 1)
        flattened_caption = [one_caption \
                for i, one_data in enumerate(self._caption)\
                for one_caption in one_data if i != correct_i]
        wrong_caption = random.sample(flattened_caption, 1)
        return correct_caption[0], wrong_caption[0]

def customed_collate_fn_for_training(batch):
    # sort by sentence length
    video, correct_caption, wrong_caption = zip(*batch)
    video = torch.tensor(video, dtype=torch.float)
    correct_caption, correct_length = _padding(correct_caption)
    correct_caption, correct_length, correct_indices = \
            _sort_and_get_indices(correct_caption, correct_length)
    wrong_caption, wrong_length = _padding(wrong_caption)
    wrong_caption, wrong_length, wrong_indices = \
            _sort_and_get_indices(wrong_caption, wrong_length)
    return video, (correct_caption, correct_length, correct_indices),\
            (wrong_caption, wrong_length, wrong_indices)

def customed_collate_fn_for_testing(batch):
    video, caption = zip(*batch)
    video = torch.tensor(video, dtype=torch.float)
    # Flatten [batch, 5, sequence_len] to [batch*5, sequence_len]
    caption = [line for one_data in caption for line in one_data]
    caption, length = _padding(caption)
    caption, length, indices = _sort_and_get_indices(caption, length)
    return video, (caption, length, indices)

def _padding(caption_list):
    length = [len(line) for line in caption_list]
    max_length = max(length)
    padded_caption = [line+[2]*(max_length-len(line)) for line in caption_list]
    return padded_caption, length
            
def _sort_and_get_indices(caption, length):
    caption = torch.tensor(caption, dtype=torch.long)
    length = torch.tensor(length, dtype=torch.long)
    length, indices = torch.sort(length, descending=True)
    caption = caption[indices]
    reversed_indices = torch.zeros(indices.shape[0], dtype=torch.long)
    for i, index in enumerate(indices):
        reversed_indices[index] = i
    return caption, length, reversed_indices
