import json, torch
from torch.utils.data import DataLoader, random_split
from src.dataset import GenreDataset
from src.model import SimpleCNN

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    genres = ["classical","jazz","rock","pop","hiphop"]  # 先 5 类跑通
    ds = GenreDataset("data/raw", genres)

    n = len(ds)
    n_train = int(n*0.8); n_val = int(n*0.1)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = SimpleCNN(n_classes=len(genres)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, 21):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        # val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(1, total)
        print(f"epoch={epoch} val_acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), "models/cnn_melspec.pth")
            with open("models/label_map.json", "w", encoding="utf-8") as f:
                json.dump({i:g for i,g in enumerate(genres)}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
