"""
GPT-2 mini en español: script de entrenamiento optimizado para Google Colab (T4)
Uso en Colab: pega este archivo en una celda (o súbelo y ejecuta `%run gpt2_spanish_train.py`).
Requisitos: pip install transformers datasets
si deseas ejecutar este script en tu maquina deberas cambiar algunas rutas
"""

import os
import json
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================================================================================================================================
# HIPERPARÁMETROS (lo que normalmente querrás ajustar, estas son las perillas que puedes mover para un mejor o peor entrenamiento)
# ==========================================================================================================================================================
N_ARTICLES = 1000                     # más artículos = mejor modelo 
TOKENIZER_NAME = "DeepESP/gpt2-spanish"
BLOCK_SIZE = 128                      # longitud de contexto
D_MODEL, N_LAYERS, N_HEADS = 768, 5, 12
DROPOUT = 0.1

EPOCHS = 4                            # empieza con pocas; mira la val loss antes de subir
BATCH_SIZE = 32
LR = 3e-4
MIN_LR_RATIO = 0.1                    # el LR final es LR * MIN_LR_RATIO
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

VAL_FRACTION = 0.05                   # 5% de los tokens para validación
LOG_EVERY = 50                        # steps entre prints
EVAL_EVERY = 250                      # steps entre evaluaciones de validación
SAVE_EVERY = 500                      # steps entre checkpoints (~1 GB c/u en Drive)

RESUME_FROM = None                    # ej: "/content/drive/MyDrive/kyle/params/gpt2_step_500.pt"
SEED = 1337

SAMPLE_PROMPTS = [
    "La historia de España comienza",
    "El río Amazonas es",
]


# =============================================================================
# MODELO
# =============================================================================
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

        # Atención causal fusionada (menos memoria y más rápida que matmul + mask + softmax)
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
        # Pre-LayerNorm (como GPT-2 real): más estable que normalizar la salida del sub-bloque
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

        # Inicialización estilo GPT-2 (IMPORTANTE con pesos atados: el Embedding por defecto
        # es N(0,1) y daría logits enormes al inicio).
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

        # Weight tying: comparte pesos entre el embedding de entrada y la capa de salida
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


# =============================================================================
# DATOS
# =============================================================================
class Wikipedia(Dataset):
    """Bloques NO solapados de `max_length` tokens (x) y su desplazamiento por 1 (y)."""

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
    """Descarga (streaming) n artículos aleatorios de Wikipedia ES y los tokeniza.
    El resultado se cachea en Drive para no repetirlo si se desconecta la sesión."""
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
    if eos_id is None:  # por si el tokenizer no define eos_token
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    encoded = tokenizer(articles, add_special_tokens=False)["input_ids"]
    pieces = [torch.tensor(ids + [eos_id], dtype=torch.long) for ids in encoded]
    tokens = torch.cat(pieces)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(tokens, cache_path)
    return tokens


# =============================================================================
# UTILIDADES: evaluación, muestreo, checkpoints, métricas
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Devuelve (loss media por token, perplejidad, accuracy de token) sobre `loader`."""
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        total_loss += F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        ).item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_tokens += y.numel()
    model.train()
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(min(mean_loss, 20.0)), total_correct / total_tokens


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


def atomic_save(obj, path):
    """Guarda en un archivo temporal y renombra: si la sesión se cae a mitad, no queda un .pt corrupto."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(save_dir, checkpoint, prev_step=None):
    path = os.path.join(save_dir, f"gpt2_step_{checkpoint['step']}.pt")
    atomic_save(checkpoint, path)
    print(f"--> Checkpoint guardado: {path} (loss {checkpoint['loss']:.4f})")
    if prev_step is not None and prev_step != checkpoint["step"]:
        prev_path = os.path.join(save_dir, f"gpt2_step_{prev_step}.pt")
        if os.path.exists(prev_path):
            os.remove(prev_path)


def log_metrics(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def plot_metrics(path):
    """Grafica train loss y val loss desde el archivo JSONL de métricas."""
    import matplotlib.pyplot as plt

    train, val = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            (train if r["type"] == "train" else val).append(r)
    plt.figure(figsize=(9, 5))
    if train:
        plt.plot([r["step"] for r in train], [r["loss"] for r in train], label="train loss", alpha=0.7)
    if val:
        plt.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()


# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def train(model, train_ds, val_ds, device, save_dir, metrics_path, tokenizer=None,
          epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, resume_from=None):
    use_amp = device.type == "cuda"
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Muy pocos datos para un batch completo: baja BATCH_SIZE o sube N_ARTICLES.")
    total_steps = epochs * steps_per_epoch
    warmup = min(WARMUP_STEPS, max(1, total_steps // 10))

    # AdamW: sin weight decay en bias y LayerNorm (parámetros de dimensión < 2)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=use_amp,
    )

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step, start_epoch, best_val = 0, 0, float("inf")
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step, start_epoch = ckpt["step"], ckpt["epoch"]
        best_val = ckpt.get("best_val", best_val)
        print(f"--> Reanudando desde step {global_step} (se repite la época {start_epoch} desde su inicio)")

    print(f"Pasos por época: {steps_per_epoch:,} | pasos totales: {total_steps:,} | warmup: {warmup}")

    model.train()
    start_time = time.time()
    steps_run = 0
    running_loss = torch.zeros((), device=device)
    running_n = 0
    last_loss = float("nan")
    last_saved = None
    best_path = os.path.join(save_dir, "gpt2_best.pt")

    def make_checkpoint(epoch):
        return {
            "step": global_step, "epoch": epoch, "loss": last_loss, "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": asdict(model.config),
            "tokenizer": TOKENIZER_NAME,
        }

    def run_eval(epoch):
        nonlocal best_val
        val_loss, val_ppl, val_acc = evaluate(model, val_loader, device, use_amp)
        print(f"   [VAL] step {global_step} | loss {val_loss:.4f} | ppl {val_ppl:.1f} | acc {val_acc:.3f}")
        log_metrics(metrics_path, {"type": "val", "step": global_step, "epoch": epoch,
                                   "val_loss": val_loss, "val_ppl": val_ppl, "val_acc": val_acc})
        if val_loss < best_val:
            best_val = val_loss
            atomic_save({"model_state_dict": model.state_dict(), "config": asdict(model.config),
                         "step": global_step, "val_loss": val_loss, "tokenizer": TOKENIZER_NAME},
                        best_path)
            print(f"   [VAL] nuevo mejor modelo guardado ({best_path})")

    for epoch in range(start_epoch, epochs):
        for x, y in train_loader:
            if global_step >= total_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            steps_run += 1
            running_loss += loss.detach()
            running_n += 1

            if global_step % LOG_EVERY == 0:
                last_loss = (running_loss / running_n).item()
                running_loss.zero_()
                running_n = 0
                elapsed = time.time() - start_time
                sec_per_step = elapsed / steps_run
                eta_min = sec_per_step * (total_steps - global_step) / 60
                tok_s = steps_run * batch_size * BLOCK_SIZE / elapsed
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch + 1}/{epochs} | Step {global_step}/{total_steps} | "
                      f"Loss {last_loss:.4f} | LR {cur_lr:.2e} | {sec_per_step:.3f}s/step | "
                      f"{tok_s:,.0f} tok/s | ETA {eta_min:.1f} min")
                log_metrics(metrics_path, {"type": "train", "step": global_step, "epoch": epoch,
                                           "loss": last_loss, "lr": cur_lr, "elapsed": elapsed})

            if global_step % EVAL_EVERY == 0:
                run_eval(epoch)

            if global_step % SAVE_EVERY == 0:
                save_checkpoint(save_dir, make_checkpoint(epoch), prev_step=last_saved)
                last_saved = global_step

        if tokenizer is not None:
            print(f"\n--- Muestras al final de la época {epoch + 1} ---")
            for prompt in SAMPLE_PROMPTS:
                print(sample(model, tokenizer, prompt, device), "\n")

    # Evaluación y checkpoint finales
    run_eval(epochs - 1)
    save_checkpoint(save_dir, make_checkpoint(epochs - 1), prev_step=last_saved)
    print(f"\nEntrenamiento terminado en {(time.time() - start_time) / 60:.1f} min.")
    return model

def load_params(model, path, device):
    """Carga solo los pesos del modelo desde un checkpoint .pt."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # El checkpoint puede ser un dict con 'model_state_dict' o directamente el state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", "?")
        loss = ckpt.get("loss", ckpt.get("val_loss", "?"))
        print(f"--> Parámetros cargados desde {path} (step {step}, loss {loss})")
    else:
        model.load_state_dict(ckpt)
        print(f"--> Parámetros cargados desde {path}")
    return model

import zipfile
import os

import os
import zipfile
import requests
# Id del enlace del archivo de de entrenamiento de Samuel Nelo
file_id = '1uxkP8zhDi1HMFxewiouZOtw8o1vS-y4E' 
download_url = f'https://google.com{file_id}'

local_zip_path = 'archivo_descargado.zip'
extract_path = '/tmp/kyle_gpt2_training_export'

print("Descargando el archivo desde Google Drive...")
response = requests.get(download_url, stream=True)

if response.status_code == 200:
    with open(local_zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("¡Descarga completada!")
else:
    print(f"Error al descargar. Código de estado: {response.status_code}")
    # Nota: Si el archivo es muy grande, Drive podría pedir una confirmación de virus
    # y este método simple de descarga podría fallar.

os.makedirs(extract_path, exist_ok=True)
print(f"Descomprimiendo en {extract_path}...")

try:
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("¡Descompresión completada! Archivos extraídos:")
    
    # 4. Listar los archivos extraídos
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            print(os.path.join(root, file))
            
    # Opcional: Eliminar el archivo .zip temporal después de extraerlo
    os.remove(local_zip_path)

except zipfile.BadZipFile:
    print("Error: El archivo descargado no es un ZIP válido. Esto suele pasar si Google Drive bloqueó la descarga directa por el tamaño del archivo.")

# =============================================================================
# MAIN (Colab)
# =============================================================================
def main():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        base_dir = "/content/drive/MyDrive/kyle"
    except ImportError:
        base_dir = "./kyle"  # fuera de Colab: carpeta local

    save_dir = os.path.join(base_dir, "params")
    metrics_path = os.path.join(base_dir, "metrics.jsonl")
    cache_path = os.path.join(base_dir, f"tokens_{N_ARTICLES}.pt")
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "SIN GPU")


    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    tokens = load_corpus(tokenizer, N_ARTICLES, cache_path)
    n_val = max(BLOCK_SIZE * 8, int(len(tokens) * VAL_FRACTION))
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]
    print(f"Tokens totales: {len(tokens):,} | train: {len(train_tokens):,} | val: {len(val_tokens):,}")

    train_ds = Wikipedia(train_tokens, BLOCK_SIZE)
    val_ds = Wikipedia(val_tokens, BLOCK_SIZE)
    print(f"Bloques de {BLOCK_SIZE} tokens -> train: {len(train_ds):,} | val: {len(val_ds):,}")

    config = Config(
        vocab_size=len(tokenizer),        # len() incluye tokens especiales añadidos
        max_seq_length=BLOCK_SIZE,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        d_ff=D_MODEL * 4,
        dropout=DROPOUT,
    )
    model = GPT2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params / 1e6:.1f}M")

    # --- Cargar pesos entrenados ---
    # Puedes elegir entre:
    # - '/tmp/kyle_gpt2_training_export/params/gpt2_step_12800.pt' (El paso más avanzado de 300k artículos)
    # - '/tmp/kyle_gpt2_training_export/params/gpt2_best.pt' (El mejor modelo evaluado)

    """ 
    para Github y otros usuarios de colab, ajusta ek numero de step del archivo segun vayas entrenando:
    """
    checkpoint_path = '/tmp/kyle_gpt2_training_export/params/gpt2_step_19311.pt'
    load_params(model, checkpoint_path, device)

    # # Primer chequeo de sanidad: la loss inicial debe estar cerca de ln(vocab) ~ 10.8
    # with torch.no_grad():
    #     xb, yb = next(iter(DataLoader(train_ds, batch_size=8, shuffle=True)))
    #     init_loss = F.cross_entropy(model(xb.to(device)).view(-1, config.vocab_size), yb.to(device).view(-1))
    # print(f"Loss inicial: {init_loss.item():.3f} (esperado ~ {math.log(config.vocab_size):.2f})")

    # if not RESUME_FROM:
    #     open(metrics_path, "w").close()
    # with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
    #     json.dump({"model": asdict(config), "tokenizer": TOKENIZER_NAME, "epochs": EPOCHS,
    #                "batch_size": BATCH_SIZE, "lr": LR, "n_articles": N_ARTICLES}, f, indent=2)

    # train(model, train_ds, val_ds, device, save_dir, metrics_path,
    #       tokenizer=tokenizer, resume_from=RESUME_FROM)

    # plot_metrics(metrics_path)

    # =================================================================
    # EVALUACIÓN: generar texto con distintos prompts
    # =================================================================
    eval_prompts = [
        "La historia de España comienza",
        "El río Amazonas es",
        "En el año 1492",
        "La inteligencia artificial permite",
        "Los científicos descubrieron que",
        "La música latinoamericana se caracteriza por",
        "El cambio climático afecta",
        "Durante la Revolución Francesa",
        "La economía de México depende",
        "Este fue uno de los mayores",
    ]

    print("\n" + "=" * 70)
    print("  EVALUACIÓN DE MUESTRAS")
    print("=" * 70)

    for i, prompt in enumerate(eval_prompts, 1):
        print(f"\n[{i}/{len(eval_prompts)}] Prompt: \"{prompt}\"")
        print("-" * 50)
        text = sample(model, tokenizer, prompt, device,
                      max_new_tokens=80, temperature=0.8, top_k=50)
        print(text)
        print()


if __name__ == "__main__":
    main()
