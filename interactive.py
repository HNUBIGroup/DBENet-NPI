
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention_head_size = int(feature_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.query_weights = nn.Linear(feature_dim, self.all_head_size)
        self.key_weights = nn.Linear(feature_dim, self.all_head_size)
        self.value_weights = nn.Linear(feature_dim, self.all_head_size)
        self.out = nn.Linear(feature_dim, feature_dim)

    def transpose_for_scores(self, x):
        assert self.feature_dim % self.num_heads == 0
        self.attention_head_size = self.feature_dim // self.num_heads
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        x = x.unsqueeze(2)
        x = x.permute(0, 1, 2, 3)
        return x

    def forward(self, query, key, value):
        query_layer = self.transpose_for_scores(self.query_weights(query))
        key_layer = self.transpose_for_scores(self.key_weights(key))
        value_layer = self.transpose_for_scores(self.value_weights(value))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output


class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, protein_feature_dim, rna_feature_dim, num_heads=4):
        super(CrossTransformerEncoder, self).__init__()
        self.protein_to_rna_fc = nn.Linear(protein_feature_dim, 512)
        self.rna_to_protein_fc = nn.Linear(rna_feature_dim, 512)
        self.multihead_crossattention = MultiHeadCrossAttention(512, num_heads)
        self.pro_norm1 = nn.LayerNorm(512)
        self.rna_norm1 = nn.LayerNorm(512)
        self.pro_feed_forward = FeedForward(512, 4)
        self.rna_feed_forward = FeedForward(512, 4)
        self.pro_norm2 = nn.LayerNorm(512)
        self.rna_norm2 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.2)

    def forward(self, protein_features, rna_features):

        # 对齐蛋白质和RNA的特征维度
        protein_features_aligned = F.relu(self.protein_to_rna_fc(protein_features))
        rna_features_aligned = F.relu(self.rna_to_protein_fc(rna_features))

        # shared_weights_cross_multihead_attention
        protein_attention_output = self.multihead_crossattention(
            query=protein_features_aligned,
            key=rna_features_aligned,
            value=rna_features_aligned
        )
        rna_attention_output = self.multihead_crossattention(
            query=rna_features_aligned,
            key=protein_features_aligned,
            value=protein_features_aligned
        )

        # add & norm
        protein_features_aligned = protein_features_aligned.unsqueeze(1)
        rna_features_aligned = rna_features_aligned.unsqueeze(1)
        pro_feature = self.dropout(self.pro_norm1(protein_attention_output + protein_features_aligned))
        rna_feature = self.dropout(self.rna_norm1(rna_attention_output + rna_features_aligned))
        pro_feature = pro_feature.squeeze(1)
        rna_feature = rna_feature.squeeze(1)

        # feed forward
        pro_feature_forward = self.pro_feed_forward(pro_feature)
        rna_feature_forward = self.rna_feed_forward(rna_feature)

        # add & norm
        pro_feature = self.dropout(self.pro_norm2(pro_feature_forward + pro_feature))
        rna_feature = self.dropout(self.rna_norm2(rna_feature_forward + rna_feature))

        return pro_feature, rna_feature
