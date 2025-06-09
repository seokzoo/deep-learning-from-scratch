import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

vocab_size = 0
data = ''
with open("input.txt") as f:
    data = f.read()
    vocab = sorted(list(set(data)))
    vocab_size = len(vocab)
    wtoi = {w: i for i, w in enumerate(vocab)}
    itow = {i: w for i, w in enumerate(vocab)}

def encode(s):
    return [wtoi[ch] for ch in s]

def decode(bs):
    return [itow[b] for b in bs]

def get_batches(data, batch_size, context_length):
    idx = np.random.randint(0, data.size(0) - context_length - 1, batch_size)
    x = torch.vstack([data[i:i+context_length] for i in idx])
    y = torch.vstack([data[i+1:i+context_length+1] for i in idx])
    return x, y

def get_rotary_matrix(context_length, latent_size, rope_base=10000):
    rotary_matrix = torch.zeros(context_length, latent_size, latent_size)
    for pos in range(context_length):
        for i in range(latent_size // 2):
            theta = torch.tensor(pos * (rope_base ** (-2 * i / latent_size)))
            rotary_matrix[pos,i*2,i*2] = torch.cos(theta)
            rotary_matrix[pos,i*2,i*2+1] = -torch.sin(theta)
            rotary_matrix[pos,i*2+1,i*2] = torch.sin(theta)
            rotary_matrix[pos,i*2+1,i*2+1] = torch.cos(theta)
    return rotary_matrix

class RMSNorm(torch.nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(latent_size))

    def forward(self, x):
        # shape : (batch_size, context_length, latent_size)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        return norm * self.scale.unsqueeze(0).unsqueeze(0)

class SwiGLU(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.W = torch.nn.Linear(input_size, input_size)
        self.V = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        v = self.V(x)
        w = self.W(x)
        swish = v * self.sigmoid(v)

        return w * swish

class SingleHeadAttention(torch.nn.Module):
    def __init__(self, context_length=16, latent_size=128, rope_base=10000):
        super().__init__()
        self.rope_base = rope_base
        self.context_length = context_length
        self.latent_size = latent_size

        self.query = torch.nn.Linear(latent_size, latent_size)
        self.key = torch.nn.Linear(latent_size, latent_size)
        self.value = torch.nn.Linear(latent_size, latent_size)
        self.rotary_matrix = get_rotary_matrix(context_length, latent_size, rope_base)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = torch.bmm(query.transpose(0, 1), self.rotary_matrix[:x.shape[1]]).transpose(0, 1)
        key = torch.bmm(key.transpose(0, 1), self.rotary_matrix[:x.shape[1]]).transpose(0, 1)

        attention_score = torch.matmul(query, key.transpose(-1,-2)) * (self.latent_size ** -0.5)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool()
        attention_score = attention_score.masked_fill(mask, float("-INF"))
        attention_prob = torch.nn.functional.softmax(attention_score, dim=-1)
        attention = torch.matmul(attention_prob, value)

        return attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_head=8, context_length=16, latent_size=128, rope_base=10000):
        super().__init__()
        self.num_head = num_head
        self.latent_size = latent_size
        self.head_list = torch.nn.ModuleList([SingleHeadAttention(
            context_length,
            latent_size,
            rope_base
        ) for _ in range(num_head)])
        self.fcn = torch.nn.Linear(num_head*latent_size, latent_size)

    def forward(self, x):
        x = [self.head_list[i](x) for i in range(self.num_head)]
        x = torch.cat(x, dim=-1)
        x = self.fcn(x)

        return x

class DecoderBlock(torch.nn.Module):
    def __init__(self, num_head=8, context_length=16, latent_size=128, rope_base=10000):
        super().__init__()
        self.rms_norm = RMSNorm(latent_size)
        self.fcn = torch.nn.Sequential(
                torch.nn.Linear(latent_size, latent_size),
                SwiGLU(latent_size)
        )
        self.multi_head_attention = MultiHeadAttention(
                num_head,
                context_length,
                latent_size,
                rope_base
        )

    def forward(self, x):
        x = self.rms_norm(x)
        x = x + self.multi_head_attention(x)
        x = self.rms_norm(x)
        x = x + self.fcn(x)

        return x

class LlamaModel(torch.nn.Module):
    def __init__(self, num_layers=4, num_head=8, context_length=16, latent_size=128, rope_base=10000):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, latent_size)
        self.fcn = torch.nn.Linear(latent_size, vocab_size)
        self.layers = torch.nn.ModuleList([DecoderBlock(
                num_head,
                context_length,
                latent_size,
                rope_base
            ) for _ in range(self.num_layers)])

    def forward(self, x, labels=None):
        x = self.embedding(x)

        for i in range(self.num_layers):
            x = self.layers[i](x)

        y = self.fcn(x)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(y.view(-1, vocab_size), labels.view(-1))
            return y, loss
        else:
            return y

@torch.no_grad()
def evaluate_loss(model, train, val, num_eval_iters=10, batch_size=8, context_length=16):
    output = {}
    model.eval()
    for dataset in [('train', train), ('val', val)]:
        losses = []
        for _ in range(num_eval_iters):
            x, y = get_batches(dataset[1], batch_size, context_length)
            pred, loss = model(x, y)
            losses.append(loss.item())
        output[dataset[0]] = np.mean(losses)
    model.train()
    return output

def generate(model, output_length=500, num_output=5, context_length=16):
    x = torch.zeros((num_output, 1)).long()

    for _ in range(output_length):
        prob = torch.nn.functional.softmax(model(x[:,-context_length:])[:,-1], dim=-1)
        sample = torch.multinomial(prob, 1)
        x = torch.cat([x, sample], dim=-1)

    return [''.join(decode(bs.tolist())) for bs in x]

if __name__ == "__main__":
    #print(encode('hello world!'))
    #print(decode(encode('hello world!')))
    batch_size = 8
    context_length = 16
    data = encode(data)
    data = torch.tensor(data, dtype=torch.long)
    train = data[:int(len(data)*.8)]
    val = data[int(len(data)*.8):]
    model = LlamaModel()

    optim = torch.optim.Adam(model.parameters())

    #print(generate(model))

    losses = []
    for steps in tqdm(range(10000)):
        optim.zero_grad()

        train_batches = get_batches(train, batch_size, context_length)
        x, y = train_batches
        logits, loss = model(x, y)
        loss.backward()
        optim.step()

        if steps % 20 == 0:
            losses.append(loss.item())
    
    if True:
        plt.plot(losses)
        plt.show()
    #val_batches = get_batches(val, batch_size, context_length)

    print(evaluate_loss(model, train, val))
    print(generate(model)[0])

