import pickle
import torch
import numpy as np
from torch import FloatTensor, LongTensor
from typing import Dict, Callable, List

with open("output.p", "rb") as f:
    vocab, contexts, X = pickle.load(f)# dict, dict, torch tensor
# print(len(vocab))

non_zero_ratio = lambda sparse_matrix: float(torch.sum(torch.where(sparse_matrix > 0, 
											  torch.ones_like(sparse_matrix), 
											  torch.zeros_like(sparse_matrix))))/float(sparse_matrix.shape[0]*sparse_matrix.shape[1])

# print(non_zero_ratio(X))

X = X.to(torch.float) + 0.05
# print(X)

to_probabilities = lambda X: torch.tensor(X.numpy()/(X.shape[0]*np.mean(X.numpy(), axis = 1).reshape(-1, 1)), dtype = float)
P = to_probabilities(X)

query = lambda word_i, word_j, vocab, probability_matrix: probability_matrix[vocab[word_i], vocab[word_j]]
# print(query('harry', 'hagrid', vocab, P))

probe = lambda word_i, word_j, word_k, vocab, probability_matrix: query(word_i,word_k, vocab, probability_matrix)/query(word_j, word_k, vocab, probability_matrix)
# print("ice", "steam", word1, probe("ice", "steam", word1, vocab, P))


weight_fn = lambda X, x_max, alpha: torch.where(X > x_max, torch.ones_like(X), (X/x_max)**alpha)
X_weighted = weight_fn(X, 100, 3/4)
# word_i, word_j = 'harry', 'snape'
# print(X[vocab[word_i], vocab[word_j]], X_weighted[vocab[word_i], vocab[word_j]])

def loss_fn(
    X_weighted: FloatTensor, 
    W: FloatTensor, 
    W_context: FloatTensor, 
    B: FloatTensor, 
    B_context: FloatTensor, 
    X: FloatTensor
) -> FloatTensor:
    # 
    return torch.sum(X_weighted * (W @ W_context.T + B + B_context.T - torch.log(X))**2)


class GloVe(torch.nn.Module):
    def __init__(self, vocab: Dict[str, int], vector_dim: int=30, device: str="cpu") -> None:
        super(GloVe, self).__init__()
        self.device = device
        self.vocab_len = len(vocab)
        self.w = torch.nn.Embedding(self.vocab_len, vector_dim)
        self.wc = torch.nn.Embedding(self.vocab_len, vector_dim)
        self.b = torch.nn.Embedding(self.vocab_len, 1)
        self.bc = torch.nn.Embedding(self.vocab_len, 1)
        # glove init
        self.w.weight.data.uniform_(-1, 1)
        self.wc.weight.data.uniform_(-1, 1)
        self.b.weight.data.zero_()
        self.bc.weight.data.zero_()
        
    def forward(self, X_weighted: FloatTensor, X: FloatTensor) -> FloatTensor:
        embedding_input = torch.arange(self.vocab_len).to(self.device)
        W = self.w(embedding_input)
        WC = self.wc(embedding_input)
        b = torch.squeeze(self.b(embedding_input))
        bc = torch.squeeze(self.bc(embedding_input))
        # loss = GloVe.forward ?
        return loss_fn(X_weighted, W, WC, b, bc, X)
    
    def get_vectors(self) -> FloatTensor:
        embedding_input = torch.arange(self.vocab_len).to(self.device)
        return self.w(embedding_input) + self.wc(embedding_input)


glove = GloVe(vocab, 300)
optimizer = torch.optim.Adagrad(glove.parameters(), lr=0.05)
num_epochs = 300
losses = []

from tqdm import trange
for e in trange(1, num_epochs+1):
    optimizer.zero_grad()
    loss = glove(X_weighted, X)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# print("Saving model...")
# torch.save(glove.get_vectors().detach(), "text8.pt")

def similarity(word_i: str, word_j: str, vocab: Dict[str, int], vectors: FloatTensor) -> float:
    i = vocab[word_i]
    j = vocab[word_j] 
    v_i = vectors[i] / torch.norm(vectors[i], p=2)  # a/|a|
    v_j = vectors[j] / torch.norm(vectors[j], p=2)  # b/|b|
    sim = torch.mm(v_i.view(1, -1), v_j.view(-1, 1)).item()
    return sim

word_vectors = glove.get_vectors().detach()

for pair in [
    ("cruciatus", "imperius"), 
    ("avada", "kedavra"), 
    ("hogwarts", "school"), 
    ("goblin", "hagrid"), 
    ("giant", "hagrid")
]:
    
    print("Similarity between '{}' and '{}' is: {}".
          format(pair[0], pair[1], similarity(pair[0], pair[1], vocab, word_vectors)))


