import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

import numpy as np

from .model import BERT
from .trainer import BERTTrainer, FineTuningTrainer
from .dataset import BERTDataset, WordVocab, LabeledDataset


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-n", "--name", type=str, default='', help="exp name")
    parser.add_argument("-nl", "--num_labels", type=int, default=None, help="number of mult-label")

    parser.add_argument("-m", "--mode", type=int, default=0, help="pre-training(0) or fine-tuning(1)")
    parser.add_argument("-p", "--pre_train_model", type=str, default=None, help="pre-trained model")
    parser.add_argument("-emb", "--pre_train_embed", type=str, default=None, help="pre-trained token embeddings")

    parser.add_argument("-hs", "--hidden", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=23, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=16, help="dataloader worker size")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, help="dropout ratio")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.00, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    exp_name = f'{args.name}-{args.vocab_path.split(".")[0].split("/")[-1]}-m_{args.mode}-hs_{args.hidden}-l_{args.layers}-a_{args.attn_heads}-b_{args.batch_size}-lr_{args.lr}-d_{args.dropout}'
    logger_name = 'runs/' + exp_name
    output_path = 'output/' + exp_name + '.model'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if os.path.exists(logger_name):
        for file_name in os.listdir(logger_name):
            if file_name.startswith("events.out.tfevents"):
                print(f"Event file {file_name} already exists")
                if input('Remove this file? (y/n) ') == 'y':
                    os.remove(os.path.join(logger_name, file_name))
                    print(f"Event file {file_name} removed")
                else:
                    sys.exit(1)

    print("Training Mode:", "Pre-training" if args.mode == 0 else "Fine-tuning")

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    if args.mode == 0:
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                    corpus_lines=args.corpus_lines, on_memory=args.on_memory)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
            if args.test_dataset is not None else None

        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) \
            if test_dataset is not None else None

        print("Building BERT model")
        bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
        if args.pre_train_embed:
            pretrain_embed = np.load(args.pre_train_embed)
            # bert.embedding.token.weight.requires_grad = False
            with torch.no_grad():
                for i in range(pretrain_embed.shape[0]):
                    bert.embedding.token.weight[5 + i] = torch.from_numpy(pretrain_embed[int(vocab.itos[5 + i])])

        print("Creating BERT Trainer")
        trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                              lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                              with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                              logger_name=logger_name)

    else:
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = LabeledDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                       corpus_lines=args.corpus_lines, on_memory=args.on_memory)

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = LabeledDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
            if args.test_dataset is not None else None

        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) \
            if test_dataset is not None else None

        print("Building BERT model")
        if args.pre_train_model:
            bert = torch.load(args.pre_train_model)
        else:
            bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
            if args.pre_train_embed:
                pretrain_embed = np.load(args.pre_train_embed)
                # bert.embedding.token.weight.requires_grad = False
                with torch.no_grad():
                    for i in range(pretrain_embed.shape[0]):
                        bert.embedding.token.weight[5 + i] = torch.from_numpy(pretrain_embed[int(vocab.itos[5 + i])])

        print("Creating BERT Trainer")
        trainer = FineTuningTrainer(bert, args.hidden, args.num_labels, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                                    lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                    with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                                    logger_name=logger_name)


    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        if (epoch + 1) % 20 == 0:
            trainer.save(epoch, output_path)

        if test_data_loader is not None:
            trainer.test(epoch)
