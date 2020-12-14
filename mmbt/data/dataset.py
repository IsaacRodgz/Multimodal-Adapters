#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import pickle

import torch
from torch.utils.data import Dataset

from mmbt.utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms_, vocab, args, data_dict=None):
        if data_dict is not None:
            self.data = data_dict
        else:
            self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        self.max_seq_len = args.max_seq_len
        if args.model in ["mmbt", "mmbtp", "mmdbt", "mmbt3"]:
            self.max_seq_len -= args.num_image_embeds
            
        if self.args.meta:
            split = data_path.split('/')[-1].split('.')[0]
            split = split if split != 'dev' else 'val'
            metadata_dir = os.path.join(self.data_dir, 'Metadata_matrices', f'{split}_metadata.npy')
            self.metadata_matrix = np.load(metadata_dir)
            metadata_dir = os.path.join(self.data_dir, 'Metadata_matrices', f'{split}_ids.pickle')
            with open(metadata_dir, 'rb') as handle:
                self.metadata_dict = pickle.load(handle)

        self.transforms = transforms_
        self.vilbert_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406, 0.456, 0.485],
                    std=[1., 1., 1.],
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.args.task == "mpaa":
            sentence = self.tokenizer(self.data[index]["script"])
            segment = torch.zeros(len(sentence))
        elif self.args.task == "moviescope":
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["synopsis"])[:(self.args.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))
        else:
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["text"])[:(self.args.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        video = None
        if self.args.visual != "none":
            if self.args.task == "moviescope":
                if self.args.visual in ["video", "both"]:
                    file = open(os.path.join(self.data_dir, '200F_VGG16', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    video = torch.from_numpy(data).squeeze(0)
                
                if self.args.visual in ["image", "both"]:
                    '''
                    image_dir = os.path.join(self.data_dir, 'Raw_Poster', f'{str(self.data[index]["id"])}.jpg')
                    image = image = Image.open(image_dir).convert("RGB")
                    image = self.transforms(image)
                    '''
                    file = open(os.path.join(self.data_dir, 'PosterFeatures', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    image = torch.from_numpy(data).squeeze(0)
                    #'''
            else:
                image = Image.open(
                    os.path.join(self.data_dir, self.data[index]["img"])
                ).convert("RGB")

                image = self.transforms(image)
                
        audio = None
        if self.args.audio == "spectrogram":
            file = open(os.path.join(self.data_dir, 'MelgramPorcessed', f'{str(self.data[index]["id"])}.p'), 'rb')
            data = pickle.load(file, encoding='bytes')
            data = torch.from_numpy(data).type(torch.FloatTensor).squeeze(0)
            audio = torch.cat([frame for frame in data[:4]], dim=1)
            
        metadata = None
        if self.args.meta:
            example_id = self.data[index]["id"]
            metadata_idx = self.metadata_dict[example_id]
            metadata = self.metadata_matrix[metadata_idx]
            metadata = torch.from_numpy(metadata).type(torch.FloatTensor)
            
        if self.args.task == "mpaa":
            genres = torch.zeros(len(self.args.genres))
            genres[[self.args.genres.index(tgt) for tgt in self.data[index]["genre"]]] = 1

        if self.args.task == "mpaa":
            return sentence, segment, image, label, genres
        elif self.args.task == "moviescope":
            return sentence, segment, image, label, video, audio, metadata
        else:
            return sentence, segment, image, label