import torch
from ogb.nodeproppred import Evaluator

from common import DEVICE

_EVALUATOR = Evaluator(name="ogbn-arxiv")


def _accuracy(y_true, logits):
    y_pred = torch.argmax(logits, dim=-1, keepdim=True).detach().cpu()
    return _EVALUATOR.eval({"y_true": y_true, "y_pred": y_pred})["acc"]


def train(gnn, graph, split_idx, optimizer, loss_fn, epochs,
          best_checkpoint_path, last_checkpoint_path=None,
          save_train_acc=False, log_every=50):
    gnn = gnn.to(DEVICE)
    train_history, val_history, acc_history, train_acc_history = [], [], [], []
    best_val_loss = float("inf")
    best_ckpt = None

    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]

    x = graph.x.to(DEVICE)
    y = graph.y.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)
    y_flat = y.squeeze(-1)

    for i in range(epochs):
        gnn.train()
        optimizer.zero_grad()
        logits = gnn(x, edge_index)
        logits_train = logits[train_idx]
        train_loss = loss_fn(logits_train, y_flat[train_idx])
        train_loss.backward()
        optimizer.step()
        train_history.append(train_loss.item())

        if save_train_acc:
            train_acc_history.append(_accuracy(y[train_idx], logits_train))

        gnn.eval()
        with torch.no_grad():
            logits_valid = logits[valid_idx]
            val_loss = loss_fn(logits_valid, y_flat[valid_idx])
            val_history.append(val_loss.item())
            val_acc = _accuracy(y[valid_idx], logits_valid)
            acc_history.append(val_acc)

        if log_every and i % log_every == 0:
            print(f"Epoch {i:<6} train loss {train_loss:<10.4f} "
                  f"val loss {val_loss:<10.4f} val acc {val_acc:<10.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_ckpt = {
                "epoch": i,
                "model_state_dict": gnn.state_dict(),
                "train_history": train_history[:],
                "val_history": val_history[:],
                "acc_history": acc_history[:],
                "best_val_loss": best_val_loss,
                "train_acc_history": train_acc_history[:],
            }
            torch.save(best_ckpt, best_checkpoint_path)

    if last_checkpoint_path is not None:
        torch.save({
            "epoch": epochs,
            "model_state_dict": gnn.state_dict(),
            "train_history": train_history,
            "val_history": val_history,
            "acc_history": acc_history,
            "train_acc_history": train_acc_history,
        }, last_checkpoint_path)

    return best_ckpt


@torch.no_grad()
def evaluate(model, graph, split_idx):
    model = model.to(DEVICE)
    model.eval()
    x = graph.x.to(DEVICE)
    y = graph.y
    edge_index = graph.edge_index.to(DEVICE)

    out = model(x, edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True).detach().cpu()

    accs = []
    for key in ("train", "valid", "test"):
        idx = split_idx[key]
        accs.append(_EVALUATOR.eval({"y_true": y[idx], "y_pred": y_pred[idx]})["acc"])
    return tuple(accs)
