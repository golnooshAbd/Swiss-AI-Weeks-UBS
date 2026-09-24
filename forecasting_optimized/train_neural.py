from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from txembed import (
    PreprocessedTransactions,
    TransactionPreprocessor,
    TransactionSequenceDataset,
    collate_transaction_sequences,
)
from txembed.batching import TransactionSequence

from features import LABELS
from neural_model import TemporalGRUForecastModel


@dataclass
class LabeledBatch:
    transactions: object
    labels: torch.Tensor

    def to(self, device: torch.device) -> LabeledBatch:
        return LabeledBatch(self.transactions.to(device), self.labels.to(device))


class LabeledSequences(Dataset[tuple[TransactionSequence, int]]):
    def __init__(self, sequences: TransactionSequenceDataset, labels: dict[str, int]) -> None:
        self.sequences = sequences
        self.labels = [labels[client_id] for client_id in sequences.client_ids]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[TransactionSequence, int]:
        return self.sequences[index], self.labels[index]


def collate(items: list[tuple[TransactionSequence, int]]) -> LabeledBatch:
    sequences, labels = zip(*items)
    return LabeledBatch(
        collate_transaction_sequences(list(sequences)),
        torch.tensor(labels, dtype=torch.long),
    )


def load_label_indices(path: Path) -> tuple[dict[str, int], np.ndarray]:
    frame = pd.read_csv(path, dtype={"client_id": str})
    mapping = {name: index for index, name in enumerate(LABELS)}
    labels = {
        client_id: mapping[label]
        for client_id, label in zip(
            frame["client_id"], frame["target_next_recurring_merchant"]
        )
    }
    return labels, frame["target_next_recurring_merchant"].map(mapping).to_numpy()


def make_loader(
    artifact_dir: Path,
    split: str,
    labels: dict[str, int],
    batch_size: int,
    max_length: int,
    shuffle: bool,
) -> tuple[DataLoader, TransactionSequenceDataset]:
    rows = PreprocessedTransactions.load_npz(artifact_dir / f"{split}.npz")
    sequences = TransactionSequenceDataset(rows, max_length=max_length)
    dataset = LabeledSequences(sequences, labels)
    return (
        DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate),
        sequences,
    )


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            probabilities.append(torch.softmax(model(batch.transactions), dim=-1).cpu().numpy())
            actual.append(batch.labels.cpu().numpy())
    probability_array = np.concatenate(probabilities)
    actual_array = np.concatenate(actual)
    predicted = probability_array.argmax(axis=1)
    score = f1_score(
        actual_array,
        predicted,
        labels=np.arange(len(LABELS)),
        average="macro",
        zero_division=0,
    )
    return float(score), probability_array, actual_array


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the isolated temporal GRU experiment")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=repository / "transaction_embedding" / "artifacts" / "uncleaned",
    )
    parser.add_argument("--label-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_labels, train_label_array = load_label_indices(args.label_dir / "train_labels.csv")
    valid_labels, _ = load_label_indices(args.label_dir / "valid_labels.csv")
    train_loader, _ = make_loader(
        args.artifact_dir, "train", train_labels, args.batch_size, args.max_length, True
    )
    valid_loader, valid_sequences = make_loader(
        args.artifact_dir, "valid", valid_labels, args.batch_size, args.max_length, False
    )

    preprocessor = TransactionPreprocessor.load(args.artifact_dir / "preprocessor.json")
    model = TemporalGRUForecastModel.from_preprocessor(preprocessor).to(device)
    counts = np.bincount(train_label_array, minlength=len(LABELS)).astype(np.float64)
    weights = np.sqrt(len(train_label_array) / (len(LABELS) * counts))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_score = -1.0
    best_probabilities: np.ndarray | None = None
    epochs_without_improvement = 0
    checkpoint = args.output_dir / "neural_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.transactions)
            loss = loss_fn(logits, batch.labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch.labels)
            total += len(batch.labels)

        valid_score, probabilities, actual = evaluate(model, valid_loader, device)
        scheduler.step(valid_score)
        print(
            f"epoch {epoch:03d} train_loss={total_loss / total:.4f} "
            f"valid_macro_f1={valid_score:.4f} lr={optimizer.param_groups[0]['lr']:.2g}"
        )
        if valid_score > best_score + 1e-5:
            best_score = valid_score
            best_probabilities = probabilities
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    assert best_probabilities is not None
    predicted = best_probabilities.argmax(axis=1)
    print(f"\nbest validation macro_f1={best_score:.4f}")
    print(
        classification_report(
            actual,
            predicted,
            labels=np.arange(len(LABELS)),
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    np.savez_compressed(
        args.output_dir / "neural_valid_probabilities.npz",
        client_ids=np.asarray(valid_sequences.client_ids, dtype=str),
        probabilities=best_probabilities,
        actual=actual,
    )


if __name__ == "__main__":
    main()
