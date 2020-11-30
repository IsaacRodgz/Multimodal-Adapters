#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader

from mmbt.data.dataset import JsonlDataset
from mmbt.data.vocab import Vocab


def get_transforms(args):
    '''
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]

    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["mmadapter", "adapter", "mmadapterfull"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):

    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    video_tensor = None
    if args.visual in ["video", "both"]:
        video_tensor = torch.stack([row[4] for row in batch])
        
    genres = None
    if args.task == "mpaa":
        genres = torch.stack([row[4] for row in batch])
        
    img_tensor = None
    if args.visual in ["image", "both"]:
        img_tensor = torch.stack([row[2] for row in batch])
        
    audio_tensor = None
    if args.audio == "spectrogram":
        audio_lens = [row[5].shape[1] for row in batch]
        audio_min_len = min(audio_lens)
        audio_tensor = torch.stack([row[5][..., :audio_min_len] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        if args.model == "mmbt3":
            mask_tensor[i_batch, :length+1] = 1
            mm_mask_tensor[i_batch, :length+args.num_image_embeds] = 1
        else:
            mask_tensor[i_batch, :length] = 1

    if args.task == "moviescope":
        return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, video_tensor, audio_tensor
    else:
        return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, genres


def get_data_loaders(args, data_all=None, partition_index=None):
    if args.model in ["mmadapter", "adapter", "mmadapterfull"]:
        tokenizer = (
            BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        )
    else:
        tokenizer = (str.split)

    transforms = get_transforms(args)

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl")
    )
    if args.task == "mpaa":
        genres = [g for line in open(os.path.join(args.data_path, args.task, "train.jsonl")) for g in json.loads(line)["genre"]]
        args.genres = list(set(genres))
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    
    train = JsonlDataset(
        os.path.join(args.data_path, args.task, "train.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    args.train_data_len = len(train)

    dev = JsonlDataset(
        os.path.join(args.data_path, args.task, "dev.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader