import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_pretrained_bert.modeling import BertLayerNorm
#from transformers import BertModel as HuggingBertModel

from mmbt.models.modeling_bert import BertModel
from mmbt.models.mmadapter_modeling import Activation_Function_Class
from mmbt.models.image import ImageEncoder


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


class BertMultimodalAdapterEncoder(nn.Module):
    def __init__(self, args):
        super(BertMultimodalAdapterEncoder, self).__init__()
        self.args = args
        
        self.bert = BertModel.from_pretrained(args.bert_model, adapter_args=vars(args))
        self.bert.add_adapter("image")
        self.bert.train_adapter(["image"])
        
        self.modality_project = nn.Linear(in_features=args.img_hidden_sz, out_features=args.modality_size)
        #self.video_reduce = nn.Conv1d(args.img_hidden_sz, args.img_hidden_sz, args.img_ngram_sz, stride=args.img_ngram_sz)
        
        if self.args.adapter_modality_type == "audio":
            self.audio_enc = AudioEncoder(args)
        if args.task == "mmimdb":
            self.img_enc = ImageEncoder(args)

    def forward(self, input_txt, attention_mask, segment, mod=None):        
        if self.args.adapter_modality_type == "video":
            mod = self.modality_project(torch.mean(mod, dim=1))
        elif self.args.adapter_modality_type == "image":
            if self.args.task == "mmimdb":
                mod = self.img_enc(mod)
                mod = self.modality_project(torch.mean(mod, dim=1))
            else:
                mod = self.modality_project(mod)
        elif  self.args.meta:
            mod = self.modality_project(mod)
        elif self.args.adapter_modality_type == "audio":
            mod = self.audio_enc(mod)
            mod = self.modality_project(torch.mean(mod, dim=2))
        
        #mod = self.video_reduce(mod.transpose(1,2))
        #mod = self.modality_project(torch.mean(mod.transpose(1,2), dim=1))
        
        out = self.bert(input_ids=input_txt, token_type_ids=segment, attention_mask=attention_mask, mod=mod, adapter_names=["image"])
        
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


class MultimodalAdapterClf(nn.Module):
    def __init__(self, args):
        super(MultimodalAdapterClf, self).__init__()
        self.args = args
        self.enc = BertMultimodalAdapterEncoder(args)
        self.clf = SimpleClassifier(args.hidden_sz, args.hidden_sz, args.n_classes, 0.0)

    def forward(self, txt, mask, segment, img=None):
        x = self.enc(txt, mask, segment, img)
        return self.clf(x)
