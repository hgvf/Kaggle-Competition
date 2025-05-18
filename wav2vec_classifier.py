import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoProcessor, AutoModelForPreTraining

class wav2vecConfig(PretrainedConfig):
    def __init__(self, n_classes, n_ffn, n_query=1, n_model=256, **kwargs):
        super(wav2vecConfig, self).__init__()

        self.n_classes = n_classes
        self.n_model = n_model
        self.n_ffn = n_ffn
        self.n_query = n_query

class wav2vecClassifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")

        self.attn = nn.MultiheadAttention(embed_dim=self.config.n_model, 
                                          num_heads=8, 
                                          batch_first=True,
                                          dropout=0.1)
        self.ffn = nn.Sequential(nn.Linear(self.config.n_model, self.config.n_ffn),
                                 nn.ReLU(),
                                 nn.Linear(self.config.n_ffn, self.config.n_model))
        
        self.query = nn.Parameter(torch.randn(1, self.config.n_query, self.config.n_model))
        self.q_attn = nn.MultiheadAttention(embed_dim=self.config.n_model, 
                                     num_heads=8, 
                                     batch_first=True,
                                     dropout=0.1)
        self.q_ffn = nn.Sequential(nn.Linear(self.config.n_model, self.config.n_ffn),
                                 nn.ReLU(),
                                 nn.Linear(self.config.n_ffn, self.config.n_model))

        self.out = nn.Linear(self.config.n_model, self.config.n_classes)

        self.init_params()

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def init_params(self):
        for param in self.wav2vec.parameters():
            param.requires_grad = False 

    def compute_metric(self, logits, labels):
        # logits: [B, n_classes]
        # labels: [B]
        
        loss = self.criterion(logits, labels)

        return loss

    def forward(self, data):
        audio = data['audio']
        label = data['label']

        # wav2vec2.0
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        hidden_states = self.wav2vec(**inputs).projected_states

        # self-attention
        attn_output, _ = self.attn(hidden_states, hidden_states, hidden_states)
        self.ffn_output = self.ffn(attn_output)

        # query the target token
        q_attn_output, _ = self.q_attn(self.query, hidden_states, hidden_states)
        q_ffn_output = self.q_ffn(q_attn_output)

        if self.config.n_query > 1:
            # average the query output
            q_ffn_output = torch.mean(q_ffn_output, dim=1)
            
        out = self.out(q_ffn_output)[:, 0]

        # compute loss
        loss = self.compute_metric(out, label)

        return {'output': out, 
                'loss': loss}

