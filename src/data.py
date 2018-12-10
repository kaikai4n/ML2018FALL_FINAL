import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import os
##############################################
# Need to check if json package is available #
import json
##############################################

class CaptionDataLoader():
    def __init__(self,
            video_filenames,
            create_word_dict=True, 
            word_dict_filename=None,
            save_word_dict=False):
        # video base filename is given so that the returned video captioning 
        # is ordered by the filename order
        self._video_filenames = video_filenames
        self.__create_word_dict = create_word_dict
        self._word_dict_filename = word_dict_filename
        self._save_word_dict = save_word_dict
        if self.__create_word_dict == False:
            if word_dict_filename is None:
                raise Exception('"word_dict_filename" for loading word \
                        dictionary index is not given')
            self._load_word_dict(word_dict_filename)

    def _save_word_dict(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._word_dict, f)

    def _load_word_dict(self, filename):
        with open(filename, 'rb') as f:
            self._word_dict = pickle.load(f)

    def _create_word_dict(self, content):
        self._word_dict = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
        for one_data in content:
            for line in one_data:
                for ele in line:
                    if ele not in self._word_dict:
                        self._word_dict[ele] = len(self._word_dict)
    
    def get_word_dict_len(self):
        return len(self._word_dict)

    def _load_from_json(self, filename):
        with open(filename, 'r') as f:
            content = json.load(f)
        return content

    def load_caption(self, caption_filename):
        print('Loading Caption Data...')
        json_content = self._load_from_json(caption_filename)
        content_dict = {one_data['id']: one_data['caption'] \
                for one_data in json_content} 
        content = [content_dict[filename.rstrip('.npy')] for filename in self._video_filenames]
        print('Caption proccessing...')
        self._data_processing(content)
        if self._create_word_dict:
            self._create_word_dict(content)
        padded_content = self._to_one_hot_value(content)
        import pdb
        pdb.set_trace()
        exit()
        if self._save_word_dict:
            if self._word_dict_filename is None:
                raise Exception('"word_dict_filename" saving word dict\
                        is not given')
            self._save_word_dict(word_dict_filename)
        return content

    def _data_processing(self, content):
        for i, _ in enumerate(content):
            for j, _ in enumerate(content[i]):
                processed_sentence = self._process_sentence(content[i][j])
                content[i][j] = processed_sentence
    
    def _process_sentence(self, sentence):
        sentence = sentence.lower()
        puntuations = ['.', ',', '!', '?', '/', '\\', '@', '#', '$', '%', '&', '*', '(', ')', '+', '-', '=', '~', '"', '[', ']', '{', '}']
        for punt in puntuations:
            sentence = sentence.replace(punt, ' '+punt+' ')
        replaces = [('\'ll', ' will'), ('it\'s', 'it is'), ('\'re', ' are'), ('i\'m', 'i am'), ('can\'t', 'can not'), ('don\'t', 'do not'), ('didn\'t', 'did not'), ('haven\'t', 'have not'), ('won\'t', 'will not'), ('isn\'t', 'is not'), ('shouldn\'t', 'should not'), ('wouldn\'t', 'would not')]
        for old, new in replaces:
            sentence = sentence.replace(old, new)
        sentence = '<SOS> ' + sentence + ' <EOS>'
        return sentence.split()

    
    def load_data_y(self, filename, encoding='utf-8'):
        raise NotImplementedError

    def _to_one_hot_value(self, content, max_sentence_len=500):
        print('To word dictionary value...')
        transformed_content = [[[self._word_dict[word] for word in line] \
                for line in one_data] for one_data in content]
        print('Start padding...')
        self._sentence_length = [len(sentence) \
                for sentence in transformed_content]
        max_length = max(self._sentence_length)
        padded_content = self._pad_equal_length(transformed_content,\
                max_length)
        return padded_content

    def _pad_equal_length(self, content, length):
        padded_content = [ele + (length-len(ele))*[self._word_dict['<PAD>']] \
                for ele in content]
        return padded_content

    def _to_numpy(self, content):
        return np.asarray(content)

    def get_sentence_length(self):
        return self._sentence_length

class VideoDataLoader():
    def __init__(self):
        pass

    def load_training_data(self, data_path):
        self._parse_filenames(data_path)
        data_list = [self._load_from_numpy(filename) \
                for filename in self._full_filenames]
        return data_list

    def load_testing_data(self, filename):
        raise NotImplementedError

    def _parse_filenames(self, data_path):
        try:
            self._base_filenames = os.listdir(data_path)
        except FileNotFoundError:
            raise Exception('Data directory "%s" not found.' % data_path)
        self._full_filenames = [os.path.join(data_path, filename)\
                for filename in self._base_filenames]

    def _load_from_numpy(self, filename):
        with open(filename, 'rb') as f:
            content = np.load(f)
        return content

    def get_base_filenames(self):
        return self._base_filenames

class DcardDataset(Dataset):
    def __init__(self, total_data, x, y, length):
        super(DcardDataset, self).__init__()
        self._x = x
        self._y = y
        self._total_data = total_data
        self._length = length

    def __len__(self):
        return self._total_data

    def __getitem__(self, i):
        return self._x[i], self._y[i], self._length[i]

def customed_collate_fn(batch):
    # sort by sentence length
    batch = sorted(batch, key=lambda x: -x[2])
    x, y, length = zip(*batch)
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    length = torch.tensor(length, dtype=torch.long)
    return x, y, length

def cut_validation(total_data, data_list, shuffle=True, propotion=0.95):
    # data_list contain [x, y, length]
    # propotion is [0, 1]
    if shuffle:
        shuffle_indexes = list(range(total_data))
        random.shuffle(shuffle_indexes)
        data_list = [[data_[i] for i in shuffle_indexes] for data_ in data_list]
    total_train = int(total_data * propotion)
    train, valid = zip(*[(data_[:total_train], data_[total_train:]) \
            for data_ in data_list])
    return (total_train,)+train, (total_data-total_train,)+valid

if __name__ == '__main__':
    vdl = VideoDataLoader()
    train_video_x = vdl.load_training_data('data/training_data/feat')
    video_filenames = vdl.get_base_filenames()
    # example for loading word dictionary from file
    """
    dl = CaptionDataLoader(
            video_filenames=video_filenames,
            create_word_dict=False,
            word_dict_filename='word_dict.pkl') 
    """
    # example to create own word dictionary via data
    dl = CaptionDataLoader(
            video_filenames=video_filenames,
            create_word_dict=True, 
            save_word_dict=True,
            word_dict_filename='word_dict.pkl')
    train_x = dl.load_caption('data/training_label.json')
    #print(train_x.shape)
    print(train_x[:5][:20])
    
