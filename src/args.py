import argparse
import model

def check_model(model_name):
    try:
        model_object = getattr(model, model_name)
    except AttributeError:
        print('Model not found:', model_name)
        exit()

def get_args(train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',
            default='data/training_data/feat',
            help='The video feat directory where \
                    stores .npy files.')
    parser.add_argument('--caption_filename',
            default='data/training_label.json',
            help='The caption label json filename.')
    parser.add_argument('--no_load_word_dict',
            default=True,
            action='store_false',
            help='If to create new word dict, then\
                    --no_load_word_dict to do so.\
                    Else, there should be a word dict\
                    file given by user.')
    parser.add_argument('--word_dict_filename',
            default='word_dict.pkl',
            help='If to create new word dict, the given\
                    filename is saved. Else, program will\
                    load from the given filename.')
    parser.add_argument('--model',
            required=True,
            default=None,
            help='The model user would like to use.')
    parser.add_argument('--no_cuda',
            default=True,
            action='store_false',
            help='Force not to use GPU.')
    parser.add_argument('--seed',
            default=7122,
            type=int,
            help='Random seed for numpy and torch.')
    parser.add_argument('-b', '--batch_size',
            default=512,
            type=int,
            help='The batch size for training.')
    if train:
        parser.add_argument('--validation',
                action='store_true',
                default=False,
                help='To split validation or not.')
        parser.add_argument('-e', '--epoches',
                type=int,
                default=50)
        parser.add_argument('-lr', '--learning_rate',
                type=float,
                default=0.001)
        parser.add_argument('--save_intervals',
                default=5,
                type=int,
                help='The epoch intervals to save models')
        parser.add_argument('--prefix',
                required=True,
                help='The prefix of saving name')
        parser.add_argument('--load_model_filename',
                default=None,
                type=str,
                help='The initialization parameters \
                        from a given model name.')
        parser.add_argument('--hidden_size',
                default=256,
                type=int,
                help='The hidden size of RNN.')
        parser.add_argument('--rnn_layers',
                default=1,
                type=int,
                help='The rnn layers.')
        parser.add_argument('--embed_dim',
                default=128,
                type=int,
                help='The word embedding dimension.')
        parser.add_argument('--dropout_rate',
                default=0.0,
                type=float,
                help='The dropout rate for RNN')
        parser.add_argument('--no_bidirectional',
                default=True,
                action='store_false',
                help='To specify not to use bidirectional\
                        for RNN.')
                
    else:
        parser.add_argument('--ensemble',
                default=False,
                action='store_true',
                help='To ensemble models or not, if True, \
                        type in multiple model names and \
                        model filenames at \
                        models and model_filenames.')
        parser.add_argument('--models',
                nargs='*',
                help='When ensemble is True, this argument\
                        is required.')
        parser.add_argument('--model_filenames',
                nargs='*',
                help='When ensemble is True, this argument\
                        is required.')
        parser.add_argument('--args_filename',
                required=True,
                help='When initializing model, it is neccessary\
                        to give the training arguments from a file.')
        parser.add_argument('--output',
                default='ans.csv',
                help='When inferencing, the designated output\
                        filename.')
        parser.add_argument('--model_filename',
                default=None,
                type=str,
                help='When inferencing one model, \
                        the designated model.')
    args = parser.parse_args()
    
    if train:
        check_model(args.model)
    else:
        if args.ensemble:
            if args.models is None or len(args.models) <= 1 or \
                    args.model_filenames is None or len(args.model_filenames) <= 1:
                raise Exception("Ensemble set true, expect to have\
                        models and model_filenames arguments at least two")
            elif len(args.models) != len(args.model_filenames):
                raise Exception("Receive different length of models\
                        and corresponding model_filenames.")
            for model_name in args.models:
                check_model(model_name)
        else:
            if args.model is None or args.model_filename is None:
                raise Exception("Expect to have model and model_filename\
                        arguments, but not given.")
            check_model(args.model)
    return args


if __name__ == '__main__':
    args = get_args(False)
