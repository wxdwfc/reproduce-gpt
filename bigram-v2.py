import torch
import torch.nn as nn
import time
from torch.nn import functional as F
import argparse

with open('input.txt', 'r') as file:
    text = file.read()
chars = sorted(list(set(text)))

### Encoder and decoder 
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}

encode = lambda x: [char2id[c] for c in x]
decode = lambda x: ''.join([id2char[i] for i in x])

vocab_size = len(chars)

### Hyper parameters
n_head = 6
n_embd = 384
n_layer = 6
batch_size = 64
block_size = 256
train_ratio = 0.9
dropout = 0.2
max_iter = 5000
learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "shakespere.model"
output_file_name = "out.txt"
#device = 'cpu'

eval_iters = 200
eval_interval = 1000

data = encode(text)
train_data = data[:int(len(text)*train_ratio)]
val_data   = data[int(len(text)*train_ratio):]

def get_batch_wrapper(split):
    if split == "train":
        return get_batch(train_data)
    else:
        return get_batch(val_data)

def get_batch(data):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size],dtype = torch.long ) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype = torch.long) for i in ix])    
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval() ## set the model to the evaluation phase

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        ## use some randomness to make the evaluation more accurate 
        for k in range(eval_iters):
            x,y = get_batch_wrapper(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() ## re-set back the model to the train phase
    return out

class BigramModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layers):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        ## Single-head attention
#        self.sa_head = Head(n_embd)

        """
        ## Multi-head attention        
        self.sa_heads = MultiHeadAttention(4, int(n_embd / 4))
        self.ffwd = FeedforwardNetwork(n_embd)
        """
        """
        self.blocks = nn.Sequential(
            Block(n_embd=n_embd,n_head=4),
            Block(n_embd=n_embd,n_head=4),
            Block(n_embd=n_embd,n_head=4),
            nn.LayerNorm(n_embd)
        )  
        """      
        self.blocks = nn.Sequential(
            *[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    ## b: batch
    ## l: context length
    ## c: channel of the output
    def forward(self, input_b_l, target_b_1): 
        B,T = input_b_l.shape
        out_b_l_embd = self.embedding_table(input_b_l) # B, T, Embd
        pos_emb_b_embd = self.position_embedding_table(torch.arange(T).to(device)) 
        x_b_l_embd = out_b_l_embd + pos_emb_b_embd

        ## For now: the head size is the same as the embedding size
#        x_b_l_h = self.sa_heads(x_b_l_embd)
        x_b_l_h = self.blocks(x_b_l_embd)
        x_b_l_h = self.ln_f(x_b_l_h)
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
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head)    :
        super().__init__()
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedforwardNetwork(n_embd)

        ## ln: layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):    
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) ## kind of weird, should project head_size * num_heads to n_embd?
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.key = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape 
        q_b_t_h = self.query(x)
        k_b_t_h = self.key(x)
        
        wei_b_t_t = q_b_t_h @ k_b_t_h.transpose(-2,-1) 
        wei_b_t_t = wei_b_t_t.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei_b_t_t /= C**0.5
        score_b_t_t = F.softmax(wei_b_t_t, dim=-1)
        score_b_t_t = self.dropout(score_b_t_t)

        v_b_t_h = self.value(x)
        return score_b_t_t @ v_b_t_h ## b_t_h
        
class FeedforwardNetwork(nn.Module):
    def __init__(self, embd_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size, embd_size),
            nn.Dropout(dropout)            
        )

    def forward(self, x):
        return self.network(x)

def main():    
    parser = argparse.ArgumentParser(description='Train or do some inference with a toy model.')
    parser.add_argument('--train', type=bool, default=False, help='Whether train the model')
    args = parser.parse_args()

    train = args.train

    print("init model")
    model = BigramModel(len(chars),n_embd, n_layers=n_layer)
    if not train:
        print("Load model from", model_name )
        model.load_state_dict(torch.load(model_name ))

    model.to(device)    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if train:
        print("training started using device", device)
        start_time = time.time()

        for step in range(max_iter):
            if step % eval_interval == 0:
                losses = estimate_loss(model)
                end_time = time.time()  
                elapsed_time = end_time - start_time

                print(f"Step {step}: train loss: {losses['train']}, val loss: {losses['val']},  \
                      elapsed time {elapsed_time}")                

            xb,yb = get_batch(train_data)

            logits, loss = model(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()    


        print("Save model to ",model_name)
        torch.save(model.state_dict(), model_name)
    

        print("----- Train done -----")

    started_text_1_1 = torch.zeros((1,1), dtype=torch.long, device=device)    
    g_text = model.generate(started_text_1_1, max_new_tokens=10_000)[0].tolist()
    text = decode(g_text)
    print("Generated text: ", text)    

    # Open a file in write mode ('w')
    with open(output_file_name, 'w') as file:
        # Write the string to the file
        file.write(text)        

if __name__ == "__main__":    
    main()  