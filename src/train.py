from data import DataLoader
from dataset import VideoCaptionDataset
from dataset import customed_collate_fn_for_training as collate_fn
from data import cut_validation
import model
from args import get_args
import torch
from utils import save_training_args
from utils import check_save_path
from utils import set_random_seed
import os
import sys
import time

def train(
        total_data,
        train_video,
        train_caption,
        prefix, 
        validation,
        batch_size,
        collate_fn,
        margin,
        model_name,
        vocabulary_size,
        embed_dim,
        hidden_size,
        rnn_layers,
        dropout_rate,
        bidirectional,
        learning_rate,
        epoches,
        save_intervals,
        use_cuda=True):
    print('Training preprocessing...')
    # processing saving path
    log_save_path, model_path, save_args_path = \
            check_save_path(prefix, validation)
    
    # processing validation data
    if validation:
        train_data, valid_data = cut_validation(
                total_data, 
                [train_video, train_caption],
                shuffle=True)
        total_train, train_video, train_caption = train_data
        total_valid, valid_video, valid_caption = valid_data
    else:
        total_train = total_data

    # make dataset
    train_dataset = VideoCaptionDataset(
            total_train, train_video, train_caption)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
    if validation:
        valid_dataset = VideoCaptionDataset(
                total_valid, valid_video, valid_caption)
        valid_loader = torch.utils.data.DataLoader(
                    dataset=valid_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn
                )

    # Initialize model
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model %s not found in model.py' % model_name)
    
    # training arguments to send
    # load_model_filename: the filename of init parmeters
    # word_dict_len: the length of word dictionary
    # embed_dim: embedding dimension
    # hidden_size: hidden size of RNN
    # dropout: dropout rate of RNN 
    # bidirectional: RNN is bidirectional or not
    training_args = {
            'load_model_filename': None,
            'vocabulary_size': vocabulary_size,
            'embed_dim': embed_dim,
            'hidden_size': hidden_size,
            'rnn_layers': rnn_layers,
            'dropout': dropout_rate,
            'bidirectional': bidirectional}
    save_training_args(training_args, save_args_path)
    my_model = model_class_object(training_args, train=True)
    my_model = my_model.cuda() if use_cuda else my_model

    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MarginRankingLoss(margin)
    
    print('Start training...')
    for epoch in range(epoches):
        start_time = time.time()
        total_loss, total_steps = 0.0, 0.0
        for step, (video, (c_caption, c_length, c_indices), \
                (w_caption, w_length, w_indices)) in enumerate(train_loader):
            duration = time.time() - start_time
            sys.stdout.write('\rduration: %05.1f, step: %02d ' \
                    % (duration, step))
            sys.stdout.flush()
            this_batch_size = video.shape[0]
            triplet_y = torch.zeros(this_batch_size) - 1
            if use_cuda:
                video = video.cuda()
                c_caption, c_length, c_indices = \
                        c_caption.cuda(), c_length.cuda(), c_indices.cuda()
                w_caption, w_length, w_indices = \
                        w_caption.cuda(), w_length.cuda(), w_indices.cuda()
                triplet_y = triplet_y.cuda()
            optimizer.zero_grad()
            pred_video, pred_c, pred_w = my_model.forward(video,\
                    c_caption, c_length, w_caption, w_length)
            pred_c, pred_w = pred_c[c_indices], pred_w[w_indices]
            c_distance, w_distance = \
                    my_model.count_triplet(pred_video, pred_c, pred_w)
            loss = loss_func(c_distance, w_distance, triplet_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.cpu())
            total_steps += 1

        total_loss /= total_steps
        if validation:
            with torch.no_grad():
                my_model.eval()
                total_valid_loss, total_valid_steps = 0, 0
                for step, (video, (c_caption, c_length, c_indices), \
                        (w_caption, w_length, w_indices)) in enumerate(valid_loader):
                    this_batch_size = video.shape[0]
                    triplet_y = torch.zeros(this_batch_size) - 1
                    if use_cuda:
                        video = video.cuda()
                        c_caption, c_length, c_indices = \
                                c_caption.cuda(), c_length.cuda(), c_indices.cuda()
                        w_caption, w_length, w_indices = \
                                w_caption.cuda(), w_length.cuda(), w_indices.cuda()
                        triplet_y = triplet_y.cuda()
                    optimizer.zero_grad()
                    pred_video, pred_c, pred_w = my_model.forward(video,\
                            c_caption, c_length, w_caption, w_length)
                    pred_c, pred_w = pred_c[c_indices], pred_w[w_indices]
                    c_distance, w_distance = \
                            my_model.count_triplet(pred_video, pred_c, pred_w)
                    loss = loss_func(c_distance, w_distance, triplet_y)
                    total_valid_loss += float(loss.cpu())
                    total_valid_steps += 1
                total_valid_loss /= total_valid_steps
                my_model.train()
            progress_msg = 'epoch:%3d, loss:%.3f, valid:%.3f'\
                    % (epoch, total_loss, total_valid_loss)
            log_msg = '%d,%.4f,%.4f\n' % \
                    (epoch, total_loss, total_valid_loss)
        else:
            progress_msg = 'epoch:%3d, loss:%.3f'\
                    % (epoch, total_loss)
            log_msg = '%d,%.4f\n' % (epoch, total_loss)
        print(progress_msg)
        with open(log_save_path, 'a') as f_log:
            f_log.write(log_msg)
        if (epoch + 1) % save_intervals == 0:
            model_save_path = os.path.join(model_path, 'models_e%d.pt' % (epoch+1))
            my_model.save(model_save_path)

def main():
    args = get_args(train=True)
    set_random_seed(args.seed)
    dl = DataLoader(
            video_dir=args.video_dir,
            caption_filename=args.caption_filename,
            load_word_dict=args.no_load_word_dict,
            word_dict_filename=args.word_dict_filename)
    train_video, train_caption = dl.load_data()
    word_dict_len = dl.get_word_dict_len()
    train(
            total_data=len(train_video),
            train_video=train_video,
            train_caption=train_caption,
            prefix=args.prefix,
            validation=args.validation,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            margin=args.margin,
            model_name=args.model,
            vocabulary_size=word_dict_len,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            rnn_layers=args.rnn_layers,
            dropout_rate=args.dropout_rate,
            bidirectional=args.no_bidirectional,
            learning_rate=args.learning_rate,
            epoches=args.epoches,
            save_intervals=args.save_intervals,
            use_cuda=args.no_cuda)


if __name__ == '__main__':
    main()
