import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r') as file:
    text = file.read()
chars = sorted(list(set(text)))

### Encoder and decoder 
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}

encode = lambda x: [char2id[c] for c in x]
decode = lambda x: ''.join([id2char[i] for i in x])

vocab_size = len(chars)
n_embd = 32
batch_size = 4
block_size = 8
train_ratio = 0.9

data = encode(text)
train_data = data[:int(len(text)*train_ratio)]
val_data   = data[int(len(text)*train_ratio):]

def get_batch(data):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size],dtype = torch.long ) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype = torch.long) for i in ix])    
    return x,y

class BigramModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        ## Single-head attention
#        self.sa_head = Head(n_embd)

        ## Multi-head attention
        self.sa_heads = MultiHeadAttention(4, int(n_embd / 4))

        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    ## b: batch
    ## l: context length
    ## c: channel of the output
    def forward(self, input_b_l, target_b_1): 
        B,T = input_b_l.shape
        out_b_l_embd = self.embedding_table(input_b_l) # B, T, Embd
        pos_emb_b_embd = self.position_embedding_table(torch.arange(T)) 
        x_b_l_embd = out_b_l_embd + pos_emb_b_embd

        ## For now: the head size is the same as the embedding size
        x_b_l_h = self.sa_heads(x_b_l_embd)
        logits_b_l_embd = self.lm_head(x_b_l_h)

        loss = None

        B,L,C = logits_b_l_embd.shape

        if target_b_1 is not None:
            target_b_1 = target_b_1.view(B * L)
            logits_b_l_embd = logits_b_l_embd.view(B*L,C)
            loss = F.cross_entropy(logits_b_l_embd, target_b_1)

        return logits_b_l_embd, loss
    
    def generate(self, x_b_l, max_new_tokens):
        for _ in range(max_new_tokens):
            x_b_l_con = x_b_l[:, -block_size:]
            logits_bl_c, _ = self.forward(x_b_l_con, None)
            logits_b_c = logits_bl_c[:, -1, :]
            probs_b_c = F.softmax(logits_b_c, dim=1)
            idx_next_b_1 = torch.multinomial(probs_b_c, num_samples=1)
            x_b_l = torch.cat([x_b_l, idx_next_b_1], dim=1)
            
        return x_b_l
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):    
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.key = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape 
        q_b_t_h = self.query(x)
        k_b_t_h = self.key(x)
        
        wei_b_t_t = q_b_t_h @ k_b_t_h.transpose(-2,-1) 
        wei_b_t_t = wei_b_t_t.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei_b_t_t /= C**0.5
        score_b_t_t = F.softmax(wei_b_t_t, dim=-1)

        v_b_t_h = self.value(x)
        return score_b_t_t @ v_b_t_h ## b_t_h
        

def main():    
    train = True
    print("init model")
    model = BigramModel(len(chars),n_embd)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if train:
        print("training started")
        for steps in range(5000):
            xb,yb = get_batch(train_data)

            logits, loss = model(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()    

            if steps % 500 == 0:
                print(loss.item())    

    print("----- Train done -----")

    started_text_1_1 = torch.zeros(1,1, dtype=torch.long)
    g_text = model.generate(started_text_1_1, max_new_tokens=100)[0].tolist()
    print("Generated text: ", decode(g_text))    

    

if __name__ == "__main__":    
    main()