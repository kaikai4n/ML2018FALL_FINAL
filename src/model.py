import torch
import torch.nn.functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self._args = args
        for key, value in self._args.items():
            exec('self._'+ key + ' = value')
    
    def _gru(self, input_dim, hidden_size):
        gru = torch.nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=self._rnn_layers,
                dropout=self._dropout,
                bidirectional=self._bidirectional,)
        return gru
    
    def _relu(self):
        return torch.nn.ReLU()

    def _softmax(self):
        return torch.nn.Softmax(dim=1)

class VideoRNN(BaseModel):
    def __init__(self, args, train=True):
        # args is a dictionary containing required arguments:
        # load_model_filename: the filename of init parmeters
        # word_dict_len: the length of word dictionary
        # embed_dim: embedding dimension
        # hidden_size: hidden size of RNN
        # dropout: dropout rate of RNN 
        # bidirectional: RNN is bidirectional or not
        super(VideoRNN, self).__init__(args)
        self._init_network()

    def _init_network(self):
        self._rnn = self._gru(4096, self._hidden_size)
        self._hidden_multiply = self._rnn_layers
        if self._bidirectional:
            self._hidden_multiply *= 2

    def forward(self, x):
        # x is a tensor of video features: [batch, 80, 4096]
        batch_size = x.shape[0]
        _, hidden = self._rnn(x, None)
        trans_hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        return trans_hidden

class CaptionRNN(BaseModel):
    def __init__(self, args, train=True):
        # args is a dictionary containing required arguments:
        # load_model_filename: the filename of init parmeters
        # word_dict_len: the length of word dictionary
        # embed_dim: embedding dimension
        # hidden_size: hidden size of RNN
        # dropout: dropout rate of RNN 
        # bidirectional: RNN is bidirectional or not
        super(CaptionRNN, self).__init__(args)
        self._init_network()

    def _init_network(self):
        self._embedding = torch.nn.Embedding(
                self._vocabulary_size, self._embed_dim)
        self._rnn = self._gru(self._embed_dim, self._hidden_size)
        self._hidden_multiply = self._rnn_layers
        if self._bidirectional:
            self._hidden_multiply *= 2

    def forward(self, x, x_length):
        # x is a tensor of sentence: [batch, max_sentence_length]
        # x_length is the length number of each sentence: [batch]
        batch_size = x.shape[0]
        x_embed = self._embedding(x)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_embed, x_length, batch_first=True)
        _, hidden = self._rnn(x_packed, None)
        trans_hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        return trans_hidden

class VideoCaption(BaseModel):
    def __init__(self, args, train=True):
        super(VideoCaption, self).__init__(args)
        self._video_rnn = VideoRNN(args, train=train)
        self._caption_rnn = CaptionRNN(args, train=train)
        if train == False:
            self.load(self._load_model_filename)

    def forward(self, video, c_caption, c_length, w_caption, w_length):
        pred_video = self._video_rnn(video)
        pred_c = self._caption_rnn(c_caption, c_length)
        pred_w = self._caption_rnn(w_caption, w_length)
        return pred_video, pred_c, pred_w

    def count_triplet(self, video, c_caption, w_caption):
        c_distance = F.pairwise_distance(video, c_caption)
        w_distance = F.pairwise_distance(video, w_caption)
        return c_distance, w_distance

    def infer(self, video, caption, length):
        # infer testing data
        # video = [batch, 80, 4096]
        # caption = [batch*5, max_length]
        pred_video = self._video_rnn(video)
        pred_captions = self._caption_rnn(caption, length)
        return pred_video, pred_captions

    def count_argmin_distance(self, pred_video, pred_caption):
        # input arguments:
        # pred_video = [batch, hidden_size]
        # pred_caption = [batch*5, hidden_size]
        # output:
        # output = [batch]
        batch_size = pred_video.shape[0]
        caption_len = pred_caption.shape[0]
        if caption_len % 5 != 0 or caption_len / 5 != batch_size:
            raise Exception('Predicted caption shape should be multiple of 5')
        pred_video = pred_video.repeat(1, 5).view(5*batch_size, -1)
        distances = F.pairwise_distance(pred_video, pred_caption)
        distances = distances.view(-1, 5)
        output = torch.argmin(distances, dim=1)
        return output

    def save(self, filename):
        state_dict = {name:value.cpu() for name, value \
                in self.state_dict().items()}
        status = {'state_dict':state_dict,}
        with open(filename, 'wb') as f_model:
            torch.save(status, f_model)
    
    def load(self, filename):
        status = torch.load(filename)
        self.load_state_dict(status['state_dict'])
    
