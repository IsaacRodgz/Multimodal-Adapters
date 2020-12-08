import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_pretrained_bert.modeling import BertLayerNorm
#from transformers import BertModel as HuggingBertModel

from mmbt.models.adapter_modeling import BertModel, Activation_Function_Class
from mmbt.models.image import ImageEncoder


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


class MultimodalBertAdapterClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertAdapterClf, self).__init__()
        self.args = args
        
        self.enc = BertModel.from_pretrained(args.bert_model, adapter_args=vars(args))
        self.clf = SimpleClassifier(args.hidden_sz, args.hidden_sz, args.n_classes, 0.0)

    def forward(self, txt, mask, segment):
        x = out = self.enc(input_ids=txt, token_type_ids=segment, attention_mask=mask)
        return self.clf(x[1])
