import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from pytorch_pretrained_bert.modeling import BertLayerNorm
#from transformers import BertModel as HuggingBertModel

from mmbt.models.modeling_bert import BertModel
from mmbt.models.mmadapter_modeling import Activation_Function_Class


class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        self.args = args
        
        conv_layers = []
        conv_layers.append(nn.Conv1d(128, 128, 128, stride=2))
        conv_layers.append(nn.Conv1d(128, 128, 128, stride=2))
        conv_layers.append(nn.AdaptiveAvgPool1d(200))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class BertMultimodalAdapterFusionEncoder(nn.Module):
    def __init__(self, args):
        super(BertMultimodalAdapterFusionEncoder, self).__init__()
        self.args = args
        
        self.bert = BertModel.from_pretrained(args.bert_model, adapter_args=vars(args))
        directory = "/".join(args.savedir.split("/")[:-1])
        self.bert.load_adapter("image", os.path.join(directory, "HoulsbyMoviescopeImageSeed2_model_run"))
        self.bert.load_adapter("video", os.path.join(directory, "HoulsbyMoviescopeVideoSeed4_model_run"), "image")
        self.bert.add_fusion_layer(["image", "video"], args)
        self.bert.train_fusion(["image", "video"])
        
        self.img_project = nn.Linear(in_features=args.img_hidden_sz, out_features=args.modality_size)
        self.vid_project = nn.Linear(in_features=args.img_hidden_sz, out_features=args.modality_size)

    def forward(self, input_txt, attention_mask, segment, img, vid):
        img = self.img_project(img)
        vid = self.vid_project(torch.mean(vid, dim=1))
        
        out = self.bert(input_ids=input_txt, token_type_ids=segment, attention_mask=attention_mask, mod=[img, vid], adapter_names=["image", "video"])
        
        return out[1]


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Dropout(dropout),
            Activation_Function_Class("gelu"),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class MultimodalAdapterFusionClf(nn.Module):
    def __init__(self, args):
        super(MultimodalAdapterFusionClf, self).__init__()
        self.args = args
        self.enc = BertMultimodalAdapterFusionEncoder(args)
        self.clf = SimpleClassifier(args.hidden_sz, args.hidden_sz, args.n_classes, 0.0)

    def forward(self, txt, mask, segment, img, vid):
        x = self.enc(txt, mask, segment, img, vid)
        return self.clf(x)
