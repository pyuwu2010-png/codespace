"""
tiny_chat_from_scratch.py

Tiny GPT-style chatbot in PyTorch with character-level tokenizer.
- Large synthetic dataset covering daily conversation topics
- Transformer with multi-head attention
- Interactive chat only, no demos
- Echo cleanup and lower randomness for coherent replies
"""
import math
import random
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Settings
# ----------------------------
SEP = " § "
RANDOM_SEED = 42
BLOCK_SIZE = 128
BATCH_SIZE = 8
N_LAYERS = 4
N_HEADS = 4
D_MODEL = 128
D_FF = 512
EPOCHS = 80
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYNTHETIC_PAIRS = 2000

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ----------------------------
# Generate expanded synthetic dataset
# ----------------------------
def generate_daily_conversation_pairs(n_pairs=SYNTHETIC_PAIRS, sep=SEP):
    categories = {
        "greeting": (["hi","hello","hey","good morning","good evening","hiya"],
                     ["hi there!","hello!","hey! how can I help?","hello — nice to meet you."]),
        "how": (["how are you","how's it going","how are you doing"],
                ["i'm doing fine, thanks!","i'm just code, but i'm well.","doing well — thanks!"]),
        "name": (["what is your name","what's your name","who are you"],
                 ["i'm a tiny transformer chatbot.","i'm a little AI assistant.","you can call me TinyGPT."]),
        "help": (["can you help me","i need help","can you assist me"],
                 ["sure — tell me what you need help with.","i can try! what's the problem?"]),
        "weather": (["how's the weather","what's the weather like","is it raining"],
                    ["i can't sense the weather, but I hope it's nice where you are.","i don't have sensors, sorry!"]),
        "origin": (["where are you from","are you from here","where do you live"],
                   ["i live in code — no single place.","i exist inside this program."]),
        "time": (["what time is it","do you know the time","what's the time"],
                 ["i don't have a real clock, but I'm always ready to chat.","i can't check the time here."]),
        "ai": (["what is ai","what is artificial intelligence","explain ai"],
               ["ai stands for artificial intelligence, systems that learn from data.",
                "ai refers to algorithms that can perform tasks that normally require human intelligence."]),
        "hobby": (["what do you like to do","what are your hobbies","what do you like"],
                  ["i like chatting with you!","i enjoy processing text and answering questions."]),
        "joke": (["tell me a joke","say a joke","make me laugh"],
                 ["here's a joke: why did the computer get cold? it left its Windows open!",
                  "why did the programmer quit his job? because he didn't get arrays."]),
        "farewell": (["bye","goodbye","see you","talk later"],
                     ["goodbye! have a nice day.","see you later!","take care!"]),
        "thanks": (["thank you","thanks","thanks a lot"],
                   ["you're welcome!","no problem!","happy to help!"]),
        "smalltalk": (["what are you doing","what's up","any plans","tell me something","what's new","how's it going"],
                      ["just here, ready to chat.","thinking about code, as usual.",
                       "i'm available to answer questions.","ready to talk!"]),
        "food": (["what's for dinner","do you like pizza","favorite food"],
                 ["i don't eat, but I like talking about food.","pizza sounds tasty!"]),
        "shopping": (["do you like shopping","can you shop","favorite store"],
                     ["i don't shop, but i can discuss shopping tips.","i like browsing online catalogs in code."]),
        "feelings": (["i'm sad","i'm happy","feeling down","excited"],
                     ["i hope you feel better!","glad to hear that!","let's chat to lift your mood."]),
        "life": (["what is life","what is the meaning of life","why are we here"],
                 ["life is what you make of it.","that's a deep question — I'm just code, but let's chat about it."])
    }

    pairs = []
    for _ in range(n_pairs):
        category = random.choice(list(categories.keys()))
        q_list, a_list = categories[category]
        q = random.choice(q_list)
        a = random.choice(a_list)
        if random.random() < 0.15:
            q += "?"
        if random.random() < 0.10:
            a += " :)"
        pairs.append(q + sep + a)
    # deduplicate
    return list(dict.fromkeys(pairs))

pairs = generate_daily_conversation_pairs(SYNTHETIC_PAIRS)

# ----------------------------
# Tokenizer
# ----------------------------
class CharTokenizer:
    def __init__(self, texts: List[str], min_freq: int = 1):
        chars = {}
        for t in texts:
            for c in t:
                chars[c] = chars.get(c,0)+1
        self.vocab = ['<pad>','<unk>','<bos>','<eos>'] + sorted([c for c,f in chars.items() if f>=min_freq])
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for s,i in self.stoi.items()}

    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        out = []
        if add_bos: out.append(self.stoi['<bos>'])
        out += [self.stoi.get(c,self.stoi['<unk>']) for c in text]
        if add_eos: out.append(self.stoi['<eos>'])
        return out

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.itos.get(i,'<unk>') for i in ids if self.itos.get(i) not in ('<bos>','<eos>','<pad>')])

    @property
    def vocab_size(self): return len(self.vocab)

tokenizer = CharTokenizer(pairs)

# ----------------------------
# Dataset
# ----------------------------
class ChatDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: CharTokenizer, block_size=BLOCK_SIZE):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if x.size(0) > self.block_size:
            x = x[:self.block_size]
        return x[:-1], x[1:]

def collate_batch(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)
    pad_id = tokenizer.stoi['<pad>']
    input_batch = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
    target_batch = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(inputs), max_len), dtype=torch.bool)
    for i,(inp,tgt) in enumerate(zip(inputs,targets)):
        input_batch[i,:len(inp)] = inp
        target_batch[i,:len(tgt)] = tgt
        mask[i,:len(inp)] = 1
    return input_batch, target_batch, mask

# ----------------------------
# Transformer
# ----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim,num_heads,dropout=0.1):
        super().__init__()
        assert dim%num_heads==0
        self.num_heads=num_heads
        self.head_dim=dim//num_heads
        self.qkv=nn.Linear(dim,dim*3)
        self.out=nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        B,T,D=x.size()
        qkv=self.qkv(x).reshape(B,T,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        att=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        causal_mask=torch.tril(torch.ones(T,T,device=x.device)).unsqueeze(0).unsqueeze(0)
        att=att.masked_fill(causal_mask==0,float('-inf'))
        if mask is not None:
            mask2=mask.unsqueeze(1).unsqueeze(2)
            att=att.masked_fill(mask2==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        att=self.dropout(att)
        out=att@v
        return self.out(out.transpose(1,2).contiguous().view(B,T,D))

class FeedForward(nn.Module):
    def __init__(self, dim,hidden_dim,dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,dim),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self,dim,num_heads,ff_hidden,dropout=0.1):
        super().__init__()
        self.ln1=nn.LayerNorm(dim)
        self.attn=MultiHeadSelfAttention(dim,num_heads,dropout)
        self.ln2=nn.LayerNorm(dim)
        self.ff=FeedForward(dim,ff_hidden,dropout)
    def forward(self,x,mask=None):
        x=x+self.attn(self.ln1(x),mask=mask)
        x=x+self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self,vocab_size,block_size,B=N_LAYERS,H=N_HEADS,D=D_MODEL,FF=D_FF,dropout=0.1):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,D)
        self.pos_emb=nn.Embedding(block_size,D)
        self.blocks=nn.ModuleList([TransformerBlock(D,H,FF,dropout) for _ in range(B)])
        self.ln_f=nn.LayerNorm(D)
        self.head=nn.Linear(D,vocab_size,bias=False)
        self.block_size=block_size

    def forward(self,idx,mask=None):
        B,T=idx.size()
        pos=torch.arange(0,T,device=idx.device).unsqueeze(0)
        x=self.tok_emb(idx)+self.pos_emb(pos)
        for blk in self.blocks: x=blk(x,mask=mask)
        x=self.ln_f(x)
        return self.head(x)

# ----------------------------
# Training
# ----------------------------
def train_model(model,dataloader,epochs=EPOCHS,lr=LEARNING_RATE,device=DEVICE):
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss=0
        t0=time.time()
        for inp,tgt,mask in dataloader:
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            optimizer.zero_grad()
            logits=model(inp,mask=mask)
            V=logits.size(-1)
            loss=F.cross_entropy(logits.view(-1,V),tgt.view(-1),ignore_index=tokenizer.stoi['<pad>'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f} time={time.time()-t0:.1f}s")
    return model

# ----------------------------
# Generation
# ----------------------------
def clean_reply(prompt,reply):
    reply=reply.strip()
    if reply.lower().startswith(prompt.lower()): reply=reply[len(prompt):].strip()
    reply=reply.replace(SEP.strip(),"").strip()
    return reply

@torch.no_grad()
def generate_reply(model,prompt,tokenizer,max_new_tokens=200,temperature=0.5,top_k=15,device=DEVICE):
    model.eval()
    ids=tokenizer.encode(prompt,add_bos=True,add_eos=False)
    input_ids=torch.tensor(ids,dtype=torch.long,device=device).unsqueeze(0)
    if input_ids.size(1)>model.block_size: input_ids=input_ids[:,-model.block_size:]
    for _ in range(max_new_tokens):
        logits=model(input_ids,mask=(input_ids!=tokenizer.stoi['<pad>']))
        logits=logits[:,-1,:]/temperature
        vals,idxs=torch.topk(logits,min(top_k,logits.size(-1)),dim=-1)
        probs=F.softmax(vals,dim=-1)
        sampled=torch.multinomial(probs,num_samples=1)
        next_id=idxs.gather(-1,sampled).reshape(1,1)
        input_ids=torch.cat([input_ids,next_id],dim=1)
        if next_id.item()==tokenizer.stoi['<eos>']: break
        if input_ids.size(1)>model.block_size: input_ids=input_ids[:,-model.block_size:]
    return clean_reply(prompt,tokenizer.decode(input_ids[0].tolist()))

# ----------------------------
# Main
# ----------------------------
def main():
    dataset=ChatDataset(pairs,tokenizer,BLOCK_SIZE)
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_batch)
    model=TinyGPT(tokenizer.vocab_size,BLOCK_SIZE)
    print("Training on device:",DEVICE)
    model=train_model(model,dataloader,epochs=EPOCHS,lr=LEARNING_RATE,device=DEVICE)
    print("\nInteractive chat available. Type 'quit' to exit.")
    try:
        while True:
            prompt=input("You: ").strip()
            if prompt.lower() in ("quit","exit"): break
            reply=generate_reply(model,prompt+SEP,tokenizer,temperature=0.5,top_k=15,device=DEVICE)
            print("AI:",reply)
    except (KeyboardInterrupt,EOFError):
        print("\nExiting chat.")

if __name__=="__main__":
    main()
