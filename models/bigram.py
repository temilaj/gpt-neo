import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    """
    we train on all the 8 examples here from context size 1 to 8 not just for computational efficiency,
    but also so the transfomer network is used to seeing context all the way from context of 1 to block_size
    after block_size though, we have to start truncating, because the transformer will never receive more than block_size inputs
    when predicting the next character. This is all the time dimension of tensors we'll be feeding into the transformer, 

    but there's one more dimension to care about, that's the batch dimension. when we're sampling these chunks of text,
    every time we feed them into the transformer we're going to have mini batches of multiple chunks of text all stacked up in a single tensor,
    This is for efficiency since GPUs are good at parrallel processing of data, so we process multiple chunks all at the same time, 
    those chunks are processed completely independelty.

    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # we don't need to call .backward()on this method i.e no back prop 
def estimate_loss():
    out = {}
    model.eval() # set model to evaluation model
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set model back to training
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # match every token Id to an embedding table in a batch by time by channel tensor (B = 4, T = 8, C= vocab_size(65))
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # continue the generation in all the batch dimensions in all the time dimensions
        for _ in range(max_new_tokens):
            # get the predictions
            # we ignore the loss since we have no targets/ ground truths
            logits, loss = self(idx)
            # focus only on the last time step, 
            # i.e last element in time dimension (since this is a bigram), because those are the predicions for the next token
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution, get just one sample
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    # zero out gradients in previous step
    optimizer.zero_grad(set_to_none=True)
    # get gradients of parameters
    loss.backward()
    # use gradients to update perameters
    optimizer.step()

# generate from the model
# create a 1 x 1 tensor containing 0s // 0 is a new line token in our vocab
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generate works on batches, so get the zeroth row
# to get the single batch dimension that exists
# that gives us timesteps (1 dimensional array of all the indices)
generated_token_ids = m.generate(context, max_new_tokens=500)[0]
print(decode(generated_token_ids.tolist()))
