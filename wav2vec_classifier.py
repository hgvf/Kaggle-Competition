import torch
import torch.nn as nn

from query_attn_classifier import QueryAttentionClassifier

from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoProcessor, AutoModel

class wav2vecConfig(PretrainedConfig):
    def __init__(self, n_classes=206, n_ffn=1024, n_query=1, n_model=256, n_head=8, **kwargs):
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
        self.wav2vec = AutoModel.from_pretrained("facebook/wav2vec2-base")

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
        for param in self.wav2vec.parameters():
            param.requires_grad = False 

    def compute_loss(self, logits, labels):
        # logits: [B, n_classes]
        # labels: [B]

        loss = self.criterion(logits, labels)

        return loss

    def forward(self, input_values, labels=None):
        audio = input_values
        label = labels.squeeze()

        # wav2vec2.0
        audio_list = [a for a in audio.cpu().numpy()]
        inputs = self.processor(audio_list, sampling_rate=16000, return_tensors="pt").to(audio.device)
       
        # (B, L, 512)
        hidden_states = self.wav2vec(**inputs).extract_features
        
        out = self.classifier(hidden_states)

        # compute loss
        loss = self.compute_loss(out, label)

        return {'output': out, 
                'loss': loss}

