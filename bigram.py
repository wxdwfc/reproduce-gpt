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

batch_size = 32
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
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    ## b: batch
    ## l: context length
    ## c: channel of the output
    def forward(self, input_b_l, target_b_1): 
        out_b_l_c = self.embedding_table(input_b_l)
        loss = None

        B,L,C = out_b_l_c.shape

        if target_b_1 is not None:
            target_b_1 = target_b_1.view(B * L)
            out_b_l_c = out_b_l_c.view(B*L,C)
            loss = F.cross_entropy(out_b_l_c, target_b_1)

        return out_b_l_c, loss
    
    def generate(self, x_b_l, max_new_tokens):
        for _ in range(max_new_tokens):
            logits_bl_c, _ = self.forward(x_b_l, None)
            logits_b_c = logits_bl_c[:, -1, :]
            probs_b_c = F.softmax(logits_b_c, dim=1)
            idx_next_b_1 = torch.multinomial(probs_b_c, num_samples=1)
            x_b_l = torch.cat([x_b_l, idx_next_b_1], dim=1)
            
        return x_b_l
    

def main():    
    print("init model")
    model = BigramModel(len(chars))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("training started")
    for steps in range(10000):
        xb,yb = get_batch(train_data)
    
        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()    

        if steps % 1000 == 0:
            print(loss.item())    

    started_text_1_1 = torch.zeros(1,1, dtype=torch.long)
    g_text = model.generate(started_text_1_1, max_new_tokens=400)[0].tolist()
    print("Generated text: ", decode(g_text))    

    

if __name__ == "__main__":    
    main()