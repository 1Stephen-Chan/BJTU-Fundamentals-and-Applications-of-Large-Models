import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import SimpleTokenizer, WikiTextDataset, load_wikitext_texts
from model import Transformer, generate_square_subsequent_mask

os.makedirs("../results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 实验配置清单 ==========
configs = [
    {"tag": "baseline", "use_positional_encoding": True, "use_residual": True, "use_layernorm": True, "n_heads":4, "d_model":128},
    {"tag": "no_pos",   "use_positional_encoding": False, "use_residual": True, "use_layernorm": True, "n_heads":4, "d_model":128},
    {"tag": "no_res",   "use_positional_encoding": True, "use_residual": False, "use_layernorm": True, "n_heads":4, "d_model":128},
    {"tag": "no_ln",    "use_positional_encoding": True, "use_residual": True, "use_layernorm": False, "n_heads":4, "d_model":128},
    {"tag": "1head",    "use_positional_encoding": True, "use_residual": True, "use_layernorm": True, "n_heads":1, "d_model":128},
    {"tag": "small",    "use_positional_encoding": True, "use_residual": True, "use_layernorm": True, "n_heads":4, "d_model":64},
    {"tag": "big",      "use_positional_encoding": True, "use_residual": True, "use_layernorm": True, "n_heads":8, "d_model":256},
]

train_texts = load_wikitext_texts()[:2000]
tokenizer = SimpleTokenizer(train_texts)
dataset = WikiTextDataset(train_texts[:2000], tokenizer, max_len=64)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

def train(model, dataloader, epochs=50, lr=3e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2id[tokenizer.pad_token])
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total += loss.item()
            count += 1
        avg = total / count
        epoch_losses.append(avg)
        print(f"[{model_tag}] epoch {epoch+1}/{epochs} loss={avg:.4f}")
    return epoch_losses

csv_path = "../results/ablation_results.csv"
with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["tag", "final_loss", "epoch_losses_path"])

for cfg in configs:
    tag = cfg["tag"]
    print(f"\n=== RUN {tag} ===")
    model = Transformer(
        src_vocab=len(tokenizer.word2id),
        tgt_vocab=len(tokenizer.word2id),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_layers=2,
        use_positional_encoding=cfg["use_positional_encoding"],
        use_residual=cfg["use_residual"],
        use_layernorm=cfg["use_layernorm"]
    )
    model_tag = tag
    # 训练
    losses = train(model, dataloader, epochs=50)
    # 保存曲线
    plt.figure()
    plt.plot(losses, marker='o')
    plt.title(f"Loss - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ppath = f"results/loss_{tag}.png"
    plt.savefig(ppath)
    plt.close()

    # 保存到csv
    with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([tag, losses[-1], ppath])

print("\n全部实验完成，结果保存在:", csv_path)
