import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_pretrained_bert.modeling import BertLayerNorm
#from transformers import BertModel as HuggingBertModel

from mmbt.models.modeling_bert import BertModel
from mmbt.models.mmadapter_modeling import Activation_Function_Class


class BertMultimodalAdapterEncoder(nn.Module):
    def __init__(self, args):
        super(BertMultimodalAdapterEncoder, self).__init__()
        self.args = args
        
        self.bert = BertModel.from_pretrained(args.bert_model, adapter_args=vars(args))
        self.bert.add_adapter("image")
        self.bert.train_adapter(["image"])
        
        self.modality_project = nn.Linear(in_features=args.img_hidden_sz, out_features=args.modality_size)
        
        #self.video_reduce = nn.Conv1d(args.img_hidden_sz, args.img_hidden_sz, args.img_ngram_sz, stride=args.img_ngram_sz)

    def forward(self, input_txt, attention_mask, segment, img=None):        
        if self.args.adapter_modality_type == "video":
            img = self.modality_project(torch.mean(img, dim=1))
        elif self.args.adapter_modality_type == "image":
            img = self.modality_project(img)
        
        #img = self.video_reduce(img.transpose(1,2))
        #img = self.modality_project(torch.mean(img.transpose(1,2), dim=1))
        
        out = self.bert(input_ids=input_txt, token_type_ids=segment, attention_mask=attention_mask, mod=img, adapter_names=["image"])
        
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
