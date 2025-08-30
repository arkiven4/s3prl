import torch
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (N, T, 1)
        
        return:
        utter_rep: size (N, H)
        """
        seq_len = batch_rep.shape[1]
        softmax = nn.functional.softmax
        att_logits = self.W(batch_rep).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        
        self.agg_method = SelfAttentionPooling(input_dim)
        self.linear = nn.Linear(input_dim, output_class_num)          

    def forward(self, features, att_mask):
        features = self.agg_method(features, att_mask)
        predicted = self.linear(features)
        return predicted
