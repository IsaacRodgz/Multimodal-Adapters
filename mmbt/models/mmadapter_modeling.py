import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.nn.functional.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


class BertMultimodalAdapter(nn.Module):
    """
    Multimodal adaptation based from "Parameter-Efficient Transfer Learning for NLP" paper:
    https://arxiv.org/pdf/1902.00751.pdf
    """

    def __init__(self, hidden_size, m_hidden_size, adapter_size, adapter_activation):
        super(BertMultimodalAdapter, self).__init__()
        seq_list = []
        m_seq_list = [] # Complementary modality
        
        # BERT input down projection
        self.layer_norm_before = nn.LayerNorm(hidden_size)
        seq_list.append(self.layer_norm_before)
        seq_list.append(nn.Linear(hidden_size, adapter_size))
        self.non_linearity = Activation_Function_Class(adapter_activation)
        seq_list.append(self.non_linearity)
        
        # Complementary modality down projection
        self.m_layer_norm_before = nn.LayerNorm(m_hidden_size)
        m_seq_list.append(self.m_layer_norm_before)
        m_seq_list.append(nn.Linear(m_hidden_size, adapter_size))
        self.m_non_linearity = Activation_Function_Class(adapter_activation)
        m_seq_list.append(self.m_non_linearity)
        
        # Down projection
        self.adapter_down = nn.Sequential(*seq_list)
        self.m_adapter_down = nn.Sequential(*m_seq_list)
        
        # Multimodality gated combination
        self.gate = nn.Linear(adapter_size*2, adapter_size, bias=False)
        
        # Up projection
        self.adapter_up = nn.Linear(adapter_size, hidden_size)
        #self.m_adapter_up = nn.Linear(adapter_size, m_hidden_size)
        #self.m_dropout = nn.Dropout(0.2)
        
        self.adapter_down.apply(self.init_bert_weights)
        self.adapter_up.apply(self.init_bert_weights)
        self.m_adapter_down.apply(self.init_bert_weights)
        #self.m_adapter_up.apply(self.init_bert_weights)

    def forward(self, hidden_states, mod=None):
        adapted_hidden_states = self.adapter_down(hidden_states)
        adapted_m_hidden_states = self.m_adapter_down(mod)
        
        seq_len = adapted_hidden_states.shape[1]
        adapted_m_hidden_states_rep = adapted_m_hidden_states.unsqueeze(1).repeat(1,seq_len,1)
        input_cat = torch.cat((adapted_hidden_states, adapted_m_hidden_states_rep), dim=2)
        scores = F.sigmoid(self.gate(input_cat))
        mixed = scores*adapted_hidden_states + (1-scores)*adapted_m_hidden_states_rep
        
        adapted_hidden_states = self.adapter_up(mixed)
        return adapted_hidden_states + hidden_states
    
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # TODO I set the std to default 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertMultimodalAttentionAdapter(nn.Module):
    """
    Multimodal adaptation based from "Parameter-Efficient Transfer Learning for NLP" paper:
    https://arxiv.org/pdf/1902.00751.pdf
    """

    def __init__(self, hidden_size, m_hidden_size, adapter_size, adapter_activation):
        super(BertMultimodalAttentionAdapter, self).__init__()
        
        # Query matrix used with BERT hidden vectors
        self.query = nn.Linear(hidden_size, adapter_size)
        self.query.apply(BertMultimodalAdapter.init_bert_weights)
        
        # Key matrix used with extra modality vectors
        self.key = nn.Linear(m_hidden_size, adapter_size)
        self.key.apply(BertMultimodalAdapter.init_bert_weights)
        
        # Value matrix used with extra modality vectors
        self.value = nn.Linear(m_hidden_size, adapter_size, bias=False)
        self.value.apply(BertMultimodalAdapter.init_bert_weights)
        
        self.dropout = nn.Dropout(0.1)
        self.scaling = adapter_size ** -0.5
        
        # Up projection
        self.adapter_up = nn.Linear(adapter_size, hidden_size)
        
        self.adapter_up.apply(self.init_bert_weights)

    def forward(self, hidden_states, mod=None):
        # Get projected queries, keys and values
        query_layer = self.query(hidden_states)
        query_layer *= self.scaling
        key_layer = self.key(mod)
        value_layer = self.value(mod)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = self.dropout(attention_scores)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # Add values with attention scores
        context_layer = torch.matmul(attention_probs, value_layer)

        adapted_hidden_states = self.adapter_up(context_layer)
        return adapted_hidden_states + hidden_states


class BertMultimodalMAGAdapter(nn.Module):
    """
    Multimodal adaptation based from "Parameter-Efficient Transfer Learning for NLP" paper:
    https://arxiv.org/pdf/1902.00751.pdf
    and MAG module
    """

    def __init__(self, hidden_size, m_hidden_size, adapter_size, adapter_activation):
        super(BertMultimodalMAGAdapter, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.W_hv = nn.Linear(m_hidden_size + hidden_size, hidden_size)
        self.W_v = nn.Linear(m_hidden_size, hidden_size)
        self.beta_shift = 1e-3

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, hidden_states, mod=None):
        eps = 1e-6
        device = self.dummy_param.device
        seq_len = hidden_states.shape[1]
        
        adapted_m_hidden_states_rep = mod.unsqueeze(1).repeat(1,seq_len,1)
        weight_v = F.relu(self.W_hv(torch.cat((adapted_m_hidden_states_rep, hidden_states), dim=-1)))

        h_m = weight_v * self.W_v(adapted_m_hidden_states_rep)

        em_norm = hidden_states.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + hidden_states)
        )

        return embedding_output