import torch
import torch.nn as nn
import sys
sys.path.append('/unilm/beats')
from query_attn_classifier import QueryAttentionClassifier
from unilm.beats.BEATs import BEATs, BEATsConfig
from transformers import PreTrainedModel, PretrainedConfig

class beatsConfig(PretrainedConfig):
    def __init__(self, n_classes=206, n_ffn=1024, n_query=1, n_model=527, n_head=8, **kwargs):
        super(beatsConfig, self).__init__()

        self.n_classes = n_classes
        self.n_model = n_model
        self.n_ffn = n_ffn
        self.n_query = n_query
        self.n_head = n_head

class beatsClassifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        checkpoint = torch.load('./unilm/beats/checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.BEATs_model = BEATs(cfg)
        self.BEATs_model.load_state_dict(checkpoint['model'])
        self.BEATs_model.eval()

        self.classifier = QueryAttentionClassifier(
            n_model=self.config.n_model,
            n_ffn=self.config.n_ffn,
            n_head=self.config.n_head,
            n_query=self.config.n_query,
            n_classes=self.config.n_classes
        )

        self.init_params()

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def init_params(self):
        for param in self.BEATs_model.parameters():
            param.requires_grad = False 

    def compute_loss(self, logits, labels):
        # logits: [B, n_classes]
        # labels: [B]

        loss = self.criterion(logits, labels)

        return loss
    
    def forward(self, input_values, labels=None):
        audio = input_values
        label = labels.squeeze()

        # (B, L, 512)
        padding_mask = torch.zeros(audio.size(0), audio.size(1)).bool().to(audio.device)
        hidden_states = self.BEATs_model.extract_features(audio, padding_mask=padding_mask)[0]

        logits = self.classifier(hidden_states.unsqueeze(1))

        if labels is not None:
            loss = self.compute_loss(logits, label)
            return loss, logits
        else:
            return logits
        
