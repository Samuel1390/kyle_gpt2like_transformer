import os
import json
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# HIPERPARÁMETROS
# =============================================================================
N_ARTICLES = 1000
TOKENIZER_NAME = "DeepESP/gpt2-spanish"
BLOCK_SIZE = 128
D_MODEL, N_LAYERS, N_HEADS = 768, 5, 12
DROPOUT = 0.1

EPOCHS = 4
BATCH_SIZE = 32
LR = 3e-4
MIN_LR_RATIO = 0.1
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

VAL_FRACTION = 0.05
LOG_EVERY = 50
EVAL_EVERY = 250
SAVE_EVERY = 500

RESUME_FROM = None
SEED = 1337

SAMPLE_PROMPTS = [
    "La historia de España comienza",
    "El río Amazonas es",
]

@dataclass
class Config:
    vocab_size: int
    max_seq_length: int = 128
    d_model: int = 768
    num_layers: int = 5
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model debe ser divisible por num_heads"
        self.head_dim = self.d_model // self.num_heads

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        b, t, _ = x.size()
        q = self.W_q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.resid_dropout(self.W_o(out))

class SwiGLUFFN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff * 2, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate, value = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.dropout(F.silu(gate) * value))

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention_layer = MultiHeadAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.attention_layer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.transformers = nn.Sequential(*[Transformer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

        self.linear.weight = self.token_embed.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        _, t = x.size()
        assert t <= self.config.max_seq_length, "secuencia más larga que max_seq_length"
        pos = torch.arange(0, t, device=x.device).unsqueeze(0)
        x = self.dropout(self.token_embed(x) + self.pos_embed(pos))
        x = self.transformers(x)
        return self.linear(self.final_norm(x))

class Wikipedia(Dataset):
    def __init__(self, tokens: torch.Tensor, max_length: int = 128):
        if len(tokens) < max_length + 1:
            raise ValueError("Muy pocos tokens para formar al menos un bloque.")
        self.tokens = tokens
        self.max_length = max_length

    def __len__(self):
        return (len(self.tokens) - 1) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        x = self.tokens[start: start + self.max_length]
        y = self.tokens[start + 1: start + self.max_length + 1]
        return x, y

def load_corpus(tokenizer, n_articles: int, cache_path: str) -> torch.Tensor:
    if os.path.exists(cache_path):
        print(f"--> Cargando tokens desde cache: {cache_path}")
        return torch.load(cache_path)

    from datasets import load_dataset

    print(f"--> Descargando {n_articles} artículos (streaming)...")
    stream = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    stream = stream.shuffle(seed=0, buffer_size=10_000)
    articles = [item["text"] for item in stream.take(n_articles)]

    print("--> Tokenizando...")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    encoded = tokenizer(articles, add_special_tokens=False)["input_ids"]
    pieces = [torch.tensor(ids + [eos_id], dtype=torch.long) for ids in encoded]
    tokens = torch.cat(pieces)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(tokens, cache_path)
    return tokens

def load_params(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", "?")
        loss = ckpt.get("loss", ckpt.get("val_loss", "?"))
        print(f"--> Parámetros cargados desde {path} (step {step}, loss {loss})")
    else:
        model.load_state_dict(ckpt)
        print(f"--> Parámetros cargados desde {path}")
    return model

@torch.no_grad()
def sample(model, tokenizer, prompt, device, max_new_tokens=60, temperature=0.8, top_k=50):
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    block = model.config.max_seq_length
    for _ in range(max_new_tokens):
        logits = model(ids[:, -block:])[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    if was_training:
        model.train()
    return tokenizer.decode(ids[0])

def main():
    in_colab = False
    try:
        from google.colab import drive
        in_colab = True
    except ImportError:
        pass

    if in_colab:
        base_dir = "/content/drive/MyDrive/kyle"
        checkpoint_path = "/tmp/kyle_gpt2_training_export/params/gpt2_step_19311.pt"
    else:
        base_dir = "./kyle"
        checkpoint_path = "./params/gpt2_step_19311.pt"

    save_dir = os.path.join(base_dir, "params")
    metrics_path = os.path.join(base_dir, "metrics.jsonl")
    cache_path = os.path.join(base_dir, f"tokens_{N_ARTICLES}.pt")
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    tokens = load_corpus(tokenizer, N_ARTICLES, cache_path)
    n_val = max(BLOCK_SIZE * 8, int(len(tokens) * VAL_FRACTION))
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]

    train_ds = Wikipedia(train_tokens, BLOCK_SIZE)
    val_ds = Wikipedia(val_tokens, BLOCK_SIZE)

    config = Config(
        vocab_size=len(tokenizer),
        max_seq_length=BLOCK_SIZE,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        d_ff=D_MODEL * 4,
        dropout=DROPOUT,
    )
    model = GPT2(config).to(device)

    try:
        load_params(model, checkpoint_path, device)
    except FileNotFoundError:
        print(f"[Aviso] No se encontró el checkpoint en {checkpoint_path}. Se iniciará con pesos aleatorios si deseas entrenar.")

    eval_prompts = [
        "La historia de España comienza",
        "El río Amazonas es",
    ]

    print("\n--- EVALUACIÓN DE MUESTRAS ---")
    for i, prompt in enumerate(eval_prompts, 1):
        print(f"\n[{i}] Prompt: \"{prompt}\"")
        text = sample(model, tokenizer, prompt, device, max_new_tokens=50)
        print(text)

if __name__ == "__main__":
    main()
