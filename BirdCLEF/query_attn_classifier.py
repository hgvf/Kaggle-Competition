import torch
import torch.nn as nn

class QueryAttentionClassifier(nn.Module):
    def __init__(self, n_model, n_ffn, n_head, n_query, n_classes):
        super(QueryAttentionClassifier, self).__init__()
        
        self.n_query = n_query

        self.attn = nn.MultiheadAttention(embed_dim=n_model, 
                                          num_heads=n_head, 
                                          batch_first=True,
                                          dropout=0.1)
        self.ffn = nn.Sequential(nn.Linear(n_model, n_ffn),
                                 nn.ReLU(),
                                 nn.Linear(n_ffn, n_model))
        
        self.query = nn.Parameter(torch.randn(n_query, n_model))
        self.q_attn = nn.MultiheadAttention(embed_dim=n_model, 
                                     num_heads=n_head, 
                                     batch_first=True,
                                     dropout=0.1)
        self.q_ffn = nn.Sequential(nn.Linear(n_model, n_ffn),
                                 nn.ReLU(),
                                 nn.Linear(n_ffn, n_model))
        
        self.out = nn.Linear(n_model, n_classes)

    def forward(self, hidden_states):
        # self-attention
        attn_output, _ = self.attn(hidden_states, hidden_states, hidden_states)
        self.ffn_output = self.ffn(attn_output)

        # query the target token
        query = self.query.unsqueeze(0).repeat(hidden_states.size(0), 1, 1)
        q_attn_output, _ = self.q_attn(query, hidden_states, hidden_states)
        q_ffn_output = self.q_ffn(q_attn_output)

        if self.n_query > 1:
            # average the query output
            q_ffn_output = torch.mean(q_ffn_output, dim=1)
            
        out = self.out(q_ffn_output)[:, 0]

        return out
