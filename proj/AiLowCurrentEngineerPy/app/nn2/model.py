"""
app/nn2/model.py

NN-2 v2 — BiLSTM + CRF для NER / sequence labeling.

Вход:  последовательность токенов (слов)
Выход: тег для каждого токена (BIO схема)

Почему CRF поверх BiLSTM:
  BiLSTM даёт эмиссионные вероятности для каждого токена,
  CRF добавляет матрицу переходов (нельзя ROOM-I без ROOM-B)
  → более связные и корректные последовательности тегов
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG   = "O"
MAX_SEQ   = 64   # максимум токенов в предложении


# ─── словарь токенов (word-level) ────────────────────────────────────────────

class WordVocab:
    def __init__(self) -> None:
        self.w2i: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.i2w: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, samples: List[Dict]) -> None:
        for s in samples:
            for tok in s["tokens"]:
                w = tok.lower()
                if w not in self.w2i:
                    idx = len(self.w2i)
                    self.w2i[w] = idx
                    self.i2w[idx] = w

    @property
    def size(self) -> int:
        return len(self.w2i)

    def encode(self, tokens: List[str], max_len: int = MAX_SEQ) -> List[int]:
        ids = [self.w2i.get(t.lower(), 1) for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"w2i": self.w2i}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "WordVocab":
        v = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        v.w2i = data["w2i"]
        v.i2w = {int(i): w for w, i in v.w2i.items()}
        return v


# ─── словарь тегов ───────────────────────────────────────────────────────────

class TagVocab:
    def __init__(self) -> None:
        self.t2i: Dict[str, int] = {}
        self.i2t: Dict[int, str] = {}

    def build(self, tag_list: List[str]) -> None:
        for tag in sorted(set(tag_list)):
            idx = len(self.t2i)
            self.t2i[tag] = idx
            self.i2t[idx] = tag

    @property
    def size(self) -> int:
        return len(self.t2i)

    def encode(self, tags: List[str], max_len: int = MAX_SEQ) -> List[int]:
        pad_id = self.t2i.get(PAD_TAG, 0)
        ids = [self.t2i.get(t, pad_id) for t in tags[:max_len]]
        ids += [pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        return [self.i2t.get(i, PAD_TAG) for i in ids]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"t2i": self.t2i}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TagVocab":
        v = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        v.t2i = data["t2i"]
        v.i2t = {int(i): t for t, i in v.t2i.items()}
        return v


# ─── датасет ─────────────────────────────────────────────────────────────────

class NERDataset(Dataset):
    def __init__(self, jsonl_path: str,
                 word_vocab: WordVocab,
                 tag_vocab:  TagVocab) -> None:
        self.word_vocab = word_vocab
        self.tag_vocab  = tag_vocab
        self.samples: List[Dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        s      = self.samples[idx]
        length = min(len(s["tokens"]), MAX_SEQ)
        x      = torch.tensor(self.word_vocab.encode(s["tokens"]), dtype=torch.long)
        y      = torch.tensor(self.tag_vocab.encode(s["tags"]),    dtype=torch.long)
        return x, y, length


# ─── CRF слой ────────────────────────────────────────────────────────────────

class CRF(nn.Module):
    """
    Conditional Random Field слой.
    Обучает матрицу переходов между тегами.
    Использует алгоритм Витерби для декодирования.
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags    = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        # запрещаем переход В начальный тег ИЗ него же (старт/стоп)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions   = nn.Parameter(torch.randn(num_tags) * 0.1)

    def _log_sum_exp(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        max_val, _ = tensor.max(dim=dim, keepdim=True)
        return (tensor - max_val).exp().sum(dim=dim).log() + max_val.squeeze(dim)

    def forward(self, emissions: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет negative log-likelihood (loss).
        emissions: (B, T, num_tags)
        tags:      (B, T)
        mask:      (B, T) bool
        """
        B, T, _ = emissions.shape
        score    = self.start_transitions[tags[:, 0]]
        score   += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, T):
            m = mask[:, t]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit  = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score = score + (trans + emit) * m

        # end transitions
        last_tag_idx = mask.long().sum(1) - 1
        last_tags    = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
        score       += self.end_transitions[last_tags]

        # partition
        partition = self._compute_log_partition(emissions, mask)
        return (partition - score).mean()

    def _compute_log_partition(self, emissions: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = emissions.shape
        score   = self.start_transitions + emissions[:, 0]  # (B, num_tags)

        for t in range(1, T):
            m             = mask[:, t].unsqueeze(1)
            broadcast     = score.unsqueeze(2)               # (B, num_tags, 1)
            emit          = emissions[:, t].unsqueeze(1)     # (B, 1, num_tags)
            next_score    = broadcast + self.transitions + emit  # (B, num_tags, num_tags)
            next_score    = self._log_sum_exp(next_score, dim=1)
            score         = torch.where(m.bool(), next_score, score)

        score += self.end_transitions
        return self._log_sum_exp(score, dim=1)

    def decode(self, emissions: torch.Tensor,
               mask: torch.Tensor) -> List[List[int]]:
        """Алгоритм Витерби — возвращает лучшую последовательность тегов."""
        B, T, _ = emissions.shape
        viterbi  = self.start_transitions + emissions[:, 0]  # (B, num_tags)
        backpointers = []

        for t in range(1, T):
            broadcast  = viterbi.unsqueeze(2)                    # (B, num_tags, 1)
            emit       = emissions[:, t].unsqueeze(1)            # (B, 1, num_tags)
            scores     = broadcast + self.transitions + emit     # (B, num_tags, num_tags)
            best_scores, best_tags = scores.max(dim=1)           # (B, num_tags)
            backpointers.append(best_tags)
            m       = mask[:, t].unsqueeze(1)
            viterbi = torch.where(m.bool(), best_scores, viterbi)

        viterbi += self.end_transitions
        _, best_last = viterbi.max(dim=1)                         # (B,)

        # traceback
        results = []
        for b in range(B):
            length   = int(mask[b].long().sum().item())
            best_seq = [best_last[b].item()]
            for bp in reversed(backpointers[:length - 1]):
                best_seq.append(bp[b][best_seq[-1]].item())
            best_seq.reverse()
            results.append(best_seq)
        return results


# ─── модель ──────────────────────────────────────────────────────────────────

class NERModel(nn.Module):
    """
    BiLSTM + CRF для sequence labeling.

    Архитектура:
      Embedding(vocab, emb_dim)
        ↓
      BiLSTM(emb_dim, hidden_dim, layers=2)
        ↓
      Dropout + Linear(hidden*2 → num_tags)   ← эмиссионные оценки
        ↓
      CRF(num_tags)                            ← матрица переходов
    """

    def __init__(self, vocab_size: int, num_tags: int,
                 emb_dim: int = 64, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout  = nn.Dropout(dropout)
        self.linear   = nn.Linear(hidden_dim * 2, num_tags)
        self.crf      = CRF(num_tags)

    def _emissions(self, x: torch.Tensor) -> torch.Tensor:
        emb          = self.dropout(self.embedding(x))
        out, _       = self.lstm(emb)
        out          = self.dropout(out)
        return self.linear(out)              # (B, T, num_tags)

    def loss(self, x: torch.Tensor, tags: torch.Tensor,
             lengths: torch.Tensor) -> torch.Tensor:
        mask      = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        emissions = self._emissions(x)
        return self.crf(emissions, tags, mask.float())

    def predict(self, x: torch.Tensor,
                lengths: torch.Tensor) -> List[List[int]]:
        mask      = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        emissions = self._emissions(x)
        return self.crf.decode(emissions, mask.float())