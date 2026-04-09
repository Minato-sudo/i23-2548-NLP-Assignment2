import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.5, pretrained_embeddings=None):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Initially frozen, can be fine-tuned later
            self.embedding.weight.requires_grad = False
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout if num_layers > 1 else 0, 
                            bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence, lengths):
        embeds = self.embedding(sentence)
        
        # Pack padded sequence
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embeds)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        output = self.dropout(output)
        tag_space = self.hidden2tag(output)
        return tag_space

class CRFLayer(nn.Module):
    def __init__(self, tagset_size):
        super(CRFLayer, self).__init__()
        self.tagset_size = tagset_size
        # transitions[i, j] is the score of transitioning to i from j
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        
        # Helper constants for Viterbi
        self.START_TAG = tagset_size - 2
        self.STOP_TAG = tagset_size - 1
        
        # Enforce transition constraints
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000

    def _forward_alg(self, feats, mask):
        # Forward algorithm to calculate the partition function (log-sum-exp of all paths)
        batch_size, seq_len, tagset_size = feats.size()
        
        # Initial alphas
        init_alphas = torch.full((batch_size, tagset_size), -10000.).to(feats.device)
        init_alphas[:, self.START_TAG] = 0.
        
        forward_var = init_alphas
        
        for i in range(seq_len):
            # feat: (batch, tagset_size)
            feat = feats[:, i, :]
            mask_i = mask[:, i].unsqueeze(1) # (batch, 1)
            
            # emit_score: (batch, tagset_size, 1)
            # trans_score: (1, tagset_size, tagset_size)
            # next_tag_var: (batch, tagset_size, tagset_size)
            
            next_tag_var = forward_var.unsqueeze(1) + self.transitions.unsqueeze(0) + feat.unsqueeze(2)
            
            # log-sum-exp over previous tags
            next_forward_var = torch.logsumexp(next_tag_var, dim=2)
            
            # Apply mask
            forward_var = (next_forward_var * mask_i) + (forward_var * (1 - mask_i))
            
        terminal_var = forward_var + self.transitions[self.STOP_TAG].unsqueeze(0)
        return torch.logsumexp(terminal_var, dim=1)

    def _score_sentence(self, feats, tags, mask):
        # Calculate the score of the gold path
        batch_size, seq_len, tagset_size = feats.size()
        
        score = torch.zeros(batch_size).to(feats.device)
        
        # Add start tag
        tags = torch.cat([torch.full((batch_size, 1), self.START_TAG, dtype=torch.long).to(feats.device), tags], dim=1)
        
        for i in range(seq_len):
            mask_i = mask[:, i]
            feat = feats[:, i, :]
            
            current_tag = tags[:, i+1]
            prev_tag = tags[:, i]
            
            # Emission score
            emit_score = torch.gather(feat, 1, current_tag.unsqueeze(1)).squeeze(1)
            # Transition score
            trans_score = self.transitions[current_tag, prev_tag]
            
            score = score + (emit_score + trans_score) * mask_i
            
        # Add stop tag score (based on the last non-padded tag)
        last_tag_idx = mask.sum(1).long()
        last_tags = torch.gather(tags, 1, last_tag_idx.unsqueeze(1)).squeeze(1)
        score = score + self.transitions[self.STOP_TAG, last_tags]
        
        return score

    def viterbi_decode(self, feats, mask):
        batch_size, seq_len, tagset_size = feats.size()
        
        backpointers = []
        
        # Initial vvars
        init_vvars = torch.full((batch_size, tagset_size), -10000.).to(feats.device)
        init_vvars[:, self.START_TAG] = 0.
        
        forward_var = init_vvars
        
        for i in range(seq_len):
            mask_i = mask[:, i].unsqueeze(1)
            feat = feats[:, i, :]
            
            next_tag_var = forward_var.unsqueeze(1) + self.transitions.unsqueeze(0)
            best_tag_id = torch.argmax(next_tag_var, dim=2) # (batch, tagset_size)
            
            # Update forward_var
            # self.transitions[best_tag_id]? No, we need to gather transitions correctly
            
            # Correct gather for batch max
            # next_tag_var[b, i, j] = forward_var[b, j] + transitions[i, j]
            # best_tag_id[b, i] = argmax_j (next_tag_var[b, i, j])
            
            max_scores = torch.gather(next_tag_var, 2, best_tag_id.unsqueeze(2)).squeeze(2)
            
            next_forward_var = max_scores + feat
            
            # Store backpointers
            backpointers.append(best_tag_id)
            
            forward_var = (next_forward_var * mask_i) + (forward_var * (1 - mask_i))
            
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.STOP_TAG].unsqueeze(0)
        best_tag_id = torch.argmax(terminal_var, dim=1)
        path_score = torch.gather(terminal_var, 1, best_tag_id.unsqueeze(1)).squeeze(1)
        
        # Follow the backpointers to decode the best path
        best_paths = []
        for b in range(batch_size):
            best_path = [best_tag_id[b].item()]
            # Only go back up to the length of this sentence
            length = int(mask[b].sum().item())
            for i in range(length - 1, -1, -1):
                best_tag_id_b = backpointers[i][b][best_path[-1]].item()
                best_path.append(best_tag_id_b)
            
            # Pop off the START_TAG (it's at the end)
            best_path.pop()
            best_path.reverse()
            best_paths.append(best_path)
            
        return path_score, best_paths

    def neg_log_likelihood(self, feats, tags, mask):
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return torch.mean(forward_score - gold_score)

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.5, pretrained_embeddings=None):
        super(BiLSTM_CRF, self).__init__()
        # tagset_size should include START_TAG and STOP_TAG
        self.bilstm = BiLSTMModel(vocab_size, tagset_size, embedding_dim, hidden_dim, num_layers, dropout, pretrained_embeddings)
        self.crf = CRFLayer(tagset_size)

    def forward(self, sentence, lengths, mask):
        # Mask is (batch, seq_len)
        feats = self.bilstm(sentence, lengths)
        return feats

    def neg_log_likelihood(self, sentence, tags, lengths, mask):
        feats = self.forward(sentence, lengths, mask)
        return self.crf.neg_log_likelihood(feats, tags, mask)

    def predict(self, sentence, lengths, mask):
        feats = self.forward(sentence, lengths, mask)
        _, paths = self.crf.viterbi_decode(feats, mask)
        return paths
