"""
SindhAgriNet - PyTorch Neural Network Architecture
Multitask deep learning model for crop disease classification, agronomic intent analysis, and multilingual reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        energy = self.projection(encoder_outputs)  # [batch_size, seq_len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)  # [batch_size, seq_len]
        # Weighted representation: [batch_size, hidden_dim]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class SindhAgriNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_diseases=8, num_intents=6, num_crops=8, num_langs=3):
        super(SindhAgriNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_dim * 2)

        # Classification Heads
        self.disease_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_diseases)
        )

        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_intents)
        )

        self.crop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_crops)
        )

        self.lang_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, num_langs)
        )

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        context_vec, attn_weights = self.attention(lstm_out)
        normed = self.norm(self.dropout(context_vec))

        disease_logits = self.disease_head(normed)
        intent_logits = self.intent_head(normed)
        crop_logits = self.crop_head(normed)
        lang_logits = self.lang_head(normed)

        return {
            "disease_logits": disease_logits,
            "intent_logits": intent_logits,
            "crop_logits": crop_logits,
            "lang_logits": lang_logits,
            "attention": attn_weights
        }
