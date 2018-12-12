import pickle
import numpy as np
import random
import os
##############################################
# Need to check if json package is available #
import json
##############################################
import pandas as pd

class CaptionDataLoader():
    def __init__(self,
            video_filenames,
            create_word_dict=True, 
            word_dict_filename=None,
            save_word_dict=False):
        # video base filename is given so that the returned video captioning 
        # is ordered by the filename order
        self._video_filenames = video_filenames
        self._create_word_dict_bool = create_word_dict
        self._word_dict_filename = word_dict_filename
        # save word_dict after calling load_caption()
        self._save_word_dict_bool = save_word_dict
        if self._create_word_dict_bool == False:
            if word_dict_filename is None:
                raise Exception('"word_dict_filename" for loading word \
                        dictionary index is not given')
            self._load_word_dict(word_dict_filename)
        # else, create word_dict after calling load_caption()

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

    def load_training_caption(self, caption_filename):
        print('Loading Caption Data...')
        json_content = self._load_from_json(caption_filename)
        content_dict = {one_data['id']: one_data['caption'] \
                for one_data in json_content} 
        content = [content_dict[filename.rstrip('.npy')] for filename in self._video_filenames]
        print('Caption proccessing...')
        self._data_processing(content)
        if self._create_word_dict_bool:
            self._create_word_dict(content)
        transformed_content = self._to_one_hot_value(content)
        if self._save_word_dict_bool:
            if self._word_dict_filename is None:
                raise Exception('"word_dict_filename" saving word dict\
                        is not given')
            self._save_word_dict(self._word_dict_filename)
        return transformed_content

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
        
    def _to_one_hot_value(self, content, max_sentence_len=500):
        print('To word dictionary value...')
        transformed_content = [[[self._word_dict[word] \
                if word in self._word_dict else self._word_dict['<UNK>'] \
                for word in line] \
                for line in one_data] for one_data in content]
        return transformed_content
    
    def load_testing_caption(self, filename):
        # testing caption filetype is csv, not json
        if self._word_dict is None:
            raise Exception('Word dictionary file not loaded error.')
        # I don't use pandas.read_csv() to read since the symbol for
        # seperating the column is ambiguous in the csv file,
        # so I do my best to parse the file myself
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        content = list(filter(None, content.split('\n')))
        video_filenames, options = [], []
        for line in content:
            video_filename, option = line.split(',', 1)
            option_seperated = self._split_options(option, 5)
            if len(option_seperated) != 5:
                print(option_seperated)
                raise Exception('Loading testing data error. Expect to get \
                         5 options but got %d options for line %s' % \
                         (len(option_seperated), line))
            video_filenames.append(video_filename)
            options.append(option_seperated)
        self._data_processing(options)
        transformed_options = self._to_one_hot_value(options)
        return video_filenames, transformed_options

    def _split_options(self, options, max_options):
        def _go_split(options, sorted_indices):
            sorted_indices.append(len(options))
            return [options[sorted_indices[i]:sorted_indices[i+1]].lstrip(',')\
                    for i in range(len(sorted_indices)-1)]
        split_indices = set([0])
        for char_index, one_char in enumerate(options[:-1]):
            if options[char_index] == ',' and options[char_index+1].istitle():
                split_indices.add(char_index)
        if len(split_indices) >= max_options:
            return _go_split(options, sorted(list(split_indices)))
        for char_index, one_char in enumerate(options[:-1]):
            if options[char_index] == '.' and options[char_index+1] == ',':
                split_indices.add(char_index+1)
        if len(split_indices) >= max_options:
            return _go_split(options, sorted(list(split_indices)))
        while len(split_indices) < max_options:
            value_changed = False
            for char_index, one_char in enumerate(options):
                if options[char_index] == ',':
                    if char_index not in split_indices:
                        split_indices.add(char_index)
                        value_changed = True
                        break
            if value_changed == False:
                print('Warning: Find one data has less than 5 options.')
                break
        return _go_split(options, sorted(list(split_indices)))

class VideoDataLoader():
    def __init__(self):
        pass

    def load_training_data(self, data_path):
        print('Loading video data...')
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

class DataLoader():
    def __init__(self,
            video_dir,
            caption_filename,
            load_word_dict=True,
            word_dict_filename=None,):
        self._caption_filename = caption_filename
        self._vdl = VideoDataLoader()
        self._video_x = self._vdl.load_training_data(video_dir)
        self._video_filenames = self._vdl.get_base_filenames()
        if load_word_dict:
            self._cdl = CaptionDataLoader(
                    video_filenames=self._video_filenames,
                    create_word_dict=False,
                    word_dict_filename=word_dict_filename)
        else:
            self._cdl = CaptionDataLoader(
                    video_filenames=self._video_filenames,
                    create_word_dict=True, 
                    save_word_dict=True,
                    word_dict_filename=word_dict_filename)

    def load_data(self, train=True):
        if train:
            self._caption_x = \
                    self._cdl.load_training_caption(self._caption_filename)
        else:
            self._ordered_video_filenames, self._caption_x = \
                    self._cdl.load_testing_caption(self._caption_filename)
            video_dict = {filename.rstrip('.npy'): value for filename, value \
                    in zip(self._video_filenames, self._video_x)}
            self._video_x = [video_dict[filename] for filename in self._ordered_video_filenames]
        return self._video_x, self._caption_x

    def get_word_dict_len(self):
        return self._cdl.get_word_dict_len()

def cut_validation(total_data, data_list, shuffle=True, propotion=0.9):
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
    # a higher level loading class
    dl = DataLoader(
            video_dir='data/training_data/feat',
            caption_filename='data/training_label.json',
            load_word_dict=True,
            word_dict_filename='word_dict.pkl')
    train_video_x, train_caption_x = dl.load_data()

    # if one tries to load video and caption seperately
    vdl = VideoDataLoader()
    train_video_x = vdl.load_training_data('data/training_data/feat')
    video_filenames = vdl.get_base_filenames()
    # example for loading word dictionary from file
    dl = CaptionDataLoader(
            video_filenames=video_filenames,
            create_word_dict=False,
            word_dict_filename='word_dict.pkl') 
    # example to create own word dictionary via data
    dl = CaptionDataLoader(
            video_filenames=video_filenames,
            create_word_dict=True, 
            save_word_dict=True,
            word_dict_filename='word_dict.pkl')
    train_caption_x = dl.load_caption('data/training_label.json')
    
