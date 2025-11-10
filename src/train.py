import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset
from model import Transformer, generate_square_subsequent_mask


# ======================
# 1. 自定义简单 tokenizer
# ======================
class SimpleTokenizer:
    """按空格切词并建立词表"""
    def __init__(self, texts, min_freq=2):
        word_freq = {}
        for line in texts:
            for word in line.strip().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        vocab += [w for w, f in word_freq.items() if f >= min_freq]
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text, max_len=64):
        tokens = text.strip().split()
        ids = [self.word2id.get(t, self.word2id[self.unk_token]) for t in tokens]
        ids = [self.word2id[self.bos_token]] + ids + [self.word2id[self.eos_token]]
        ids = ids[:max_len]
        pad_len = max_len - len(ids)
        ids += [self.word2id[self.pad_token]] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.id2word.get(int(i), self.unk_token)
            if w in ["<pad>", "<bos>", "<eos>"]:
                continue
            words.append(w)
        return " ".join(words)


# ======================
# 2. 数据集定义
# ======================
class WikiTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = [line for line in data if len(line.strip().split()) > 3]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        ids = self.tokenizer.encode(text, self.max_len + 1)
        src = ids[:-1]
        tgt = ids[1:]
        return src, tgt


# ======================
# 3. 训练
# ======================
def train(model, dataloader, optimizer, criterion, device, epochs=5):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return losses

def load_wikitext_texts():
    """加载 WikiText-2 训练集文本，过滤空行"""
    print("加载WikiText-2数据集")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]
    return train_texts

# ======================
# 4. 主程序入口
# ======================
if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_texts = load_wikitext_texts()

    # 构建自定义 tokenizer
    tokenizer = SimpleTokenizer(train_texts)
    vocab_size = len(tokenizer.word2id)
    print(f"词表大小: {vocab_size}")

    # 保存词表文件
    vocab_path = "../results/vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in tokenizer.word2id.keys():
            f.write(word + "\n")
    print(f"词表已保存到: {vocab_path}")

    # 构造数据集与加载器
    train_data = WikiTextDataset(train_texts[:2000], tokenizer, max_len=64)
    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    # 定义模型
    model = Transformer(src_vocab=vocab_size, tgt_vocab=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2id[tokenizer.pad_token])

    # 训练
    losses = train(model, dataloader, optimizer, criterion, device, epochs=100)

    # 保存模型和曲线
    torch.save(model.state_dict(), "../results/model.pt")
    print("模型参数已保存到: results/model.pt")

    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Transformer (WikiText-2)")
    plt.savefig("results/loss_curve.png")
    plt.show()
