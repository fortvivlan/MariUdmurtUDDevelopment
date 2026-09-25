"""Adapted BaseUDParser encoder/biaffine parser with joint UD heads.

BaseUDParser commit 62c70e47766b509c7c97ef53421c0193902366e6 is GPL-3.0.
This implementation keeps its mean-subtoken encoder and biaffine arc/label design,
and adds explicit ROOT handling, long-input windows and transferable tag heads.
"""

from bisect import bisect_right

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from .conllu import Sentence, format_feats, parse_feats
from .lemma import apply_lemma_rule, lemma_rule
from .mst import decode_single_root


def _mlp(input_size: int, hidden_size: int, output_size: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(input_size, hidden_size),
                         nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))


class Biaffine(nn.Module):
    """Biaffine matrix scoring adapted from BaseUDParser's attention layer."""

    def __init__(self, width: int, labels: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(labels, width + 1, width + 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, heads: torch.Tensor, dependents: torch.Tensor) -> torch.Tensor:
        heads = torch.cat((heads, torch.ones_like(heads[..., :1])), dim=-1)
        dependents = torch.cat((dependents, torch.ones_like(dependents[..., :1])), dim=-1)
        # [batch, dependent, head, label]
        return torch.einsum("bdi,rij,bhj->bdhr", dependents, self.weight, heads)


class JointUDParser(nn.Module):
    def __init__(self, backbone: str, vocab: dict, max_length: int = 512,
                 window_overlap: int = 64, dropout: float = .1,
                 gradient_checkpointing: bool = True, revision: str | None = None,
                 tokenizer_path: str | None = None):
        super().__init__()
        self.backbone = backbone
        self.vocab = {key: value for key, value in vocab.items() if key != "lemma_rule_counts"}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or backbone,
                                                       use_fast=True,
                                                       revision=None if tokenizer_path else revision)
        if not self.tokenizer.is_fast:
            raise ValueError("Word IDs require a fast tokenizer")
        self.encoder = AutoModel.from_pretrained(backbone, attn_implementation="sdpa",
                                                 revision=revision)
        if self.encoder.get_input_embeddings().num_embeddings != len(self.tokenizer):
            raise ValueError("Backbone and tokenizer vocabularies differ")
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        if max_length > self.encoder.config.max_position_embeddings - 2:
            raise ValueError("Requested window exceeds encoder position capacity")
        self.max_length = max_length
        self.window_overlap = window_overlap
        width = self.encoder.config.hidden_size
        self.root = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.root, std=.02)
        self.arc_head = _mlp(width, 256, 128, dropout)
        self.arc_dep = _mlp(width, 256, 128, dropout)
        self.rel_head = _mlp(width, 256, 128, dropout)
        self.rel_dep = _mlp(width, 256, 128, dropout)
        self.arc_biaffine = Biaffine(128)
        self.rel_biaffine = Biaffine(128, len(self.vocab["deprel"]))
        self.upos_head = _mlp(width, 256, len(self.vocab["upos"]), dropout)
        self.lemma_head = _mlp(width, 256, len(self.vocab["lemma_rules"]), dropout)
        self.feature_names = list(self.vocab["feats"])
        self.feature_heads = nn.ModuleList(
            [_mlp(width, 128, len(self.vocab["feats"][key]), dropout)
             for key in self.feature_names]
        )

    def _encode_words(self, sentences: list[Sentence]) -> tuple[torch.Tensor, list[int]]:
        device = next(self.parameters()).device
        window_ids, selects, lengths = [], [], []
        global_base = 0
        special_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        capacity = self.max_length - special_count
        if capacity <= self.window_overlap:
            raise ValueError("Window overlap must be smaller than usable input length")
        # Fast tokenizers reject get_special_tokens_mask(..., False). Locate the
        # contiguous payload in their single-sequence template with a token ID
        # that cannot itself be a special token.
        special_ids = set(self.tokenizer.all_special_ids)
        probe_id = next(index for index in range(self.tokenizer.vocab_size)
                        if index not in special_ids)
        probe = self.tokenizer.build_inputs_with_special_tokens([probe_id])
        prefix_length = probe.index(probe_id)
        for sentence in sentences:
            words = [word.form for word in sentence.words]
            lengths.append(len(words))
            encoding = self.tokenizer(words, is_split_into_words=True,
                                      add_special_tokens=False, truncation=False)
            pieces = encoding["input_ids"]
            word_ids = encoding.word_ids()
            if not pieces or any(word_id is None for word_id in word_ids):
                raise ValueError("Tokenizer failed to align syntactic words")
            boundaries = [0] + [index for index in range(1, len(word_ids))
                                if word_ids[index] != word_ids[index - 1]] + [len(pieces)]
            seen = set()
            start = 0
            while start < len(pieces):
                limit = min(start + capacity, len(pieces))
                # Keep complete words together whenever one word fits a window.
                end = boundaries[bisect_right(boundaries, limit) - 1]
                if end <= start:
                    end = limit  # An exceptionally long word spans windows.
                part = pieces[start:end]
                ids = self.tokenizer.build_inputs_with_special_tokens(part)
                if (len(ids) > self.max_length or
                        ids[prefix_length:prefix_length + len(part)] != part or
                        len(ids) - len(part) != special_count):
                    raise ValueError("Invalid tokenizer special-token layout")
                window_index = len(window_ids)
                window_ids.append(torch.tensor(ids, dtype=torch.long))
                for offset, piece_index in enumerate(range(start, end)):
                    if piece_index not in seen:
                        selects.append((window_index, prefix_length + offset,
                                        global_base + word_ids[piece_index]))
                        seen.add(piece_index)
                if end == len(pieces):
                    break
                desired = max(start + 1, end - self.window_overlap)
                next_start = boundaries[bisect_right(boundaries, desired) - 1]
                if next_start <= start:
                    next_boundary = boundaries[bisect_right(boundaries, start)]
                    next_start = next_boundary if next_boundary < end else desired
                start = next_start
            global_base += len(words)
        pad_id = self.tokenizer.pad_token_id
        ids = pad_sequence(window_ids, batch_first=True, padding_value=pad_id).to(device)
        attention = ids.ne(pad_id).long()
        encoded = self.encoder(input_ids=ids, attention_mask=attention).last_hidden_state
        wi, pi, gi = (torch.tensor(column, device=device, dtype=torch.long)
                      for column in zip(*selects, strict=True))
        gathered = encoded[wi, pi]
        pooled_sum = torch.zeros(global_base, encoded.size(-1), device=device,
                                 dtype=encoded.dtype).index_add(0, gi, gathered)
        counts = torch.zeros(global_base, device=device, dtype=encoded.dtype).index_add(
            0, gi, torch.ones_like(gi, dtype=encoded.dtype)
        )
        if torch.any(counts == 0):
            raise ValueError("At least one word received no subtokens")
        pooled = pooled_sum / counts.unsqueeze(-1)
        embeddings = []
        offset = 0
        for length in lengths:
            embeddings.append(torch.cat((self.root.to(pooled.dtype).unsqueeze(0),
                                         pooled[offset:offset + length]), dim=0))
            offset += length
        return pad_sequence(embeddings, batch_first=True), lengths

    def forward(self, sentences: list[Sentence], compute_loss: bool = True) -> dict:
        embeddings, lengths = self._encode_words(sentences)
        arc_logits = self.arc_biaffine(self.arc_head(embeddings),
                                       self.arc_dep(embeddings)).squeeze(-1)
        rel_logits = self.rel_biaffine(self.rel_head(embeddings),
                                       self.rel_dep(embeddings))
        upos_logits = self.upos_head(embeddings)
        lemma_logits = self.lemma_head(embeddings)
        feature_logits = [head(embeddings) for head in self.feature_heads]
        n = arc_logits.size(1)
        device = arc_logits.device
        for batch, length in enumerate(lengths):
            arc_logits[batch, :, length + 1:] = -1e4
        diagonal = torch.arange(n, device=device)
        arc_logits[:, diagonal, diagonal] = -1e4
        output = {"arc": arc_logits, "rel": rel_logits, "upos": upos_logits,
                  "lemma": lemma_logits, "feats": feature_logits, "lengths": lengths}
        if not compute_loss:
            return output
        batch_ids, dep_ids, gold_heads, gold_rels = [], [], [], []
        upos_targets, lemma_targets = [], []
        feat_targets = [[] for _ in self.feature_names]
        rel_index = {label: index for index, label in enumerate(self.vocab["deprel"])}
        upos_index = {label: index for index, label in enumerate(self.vocab["upos"])}
        lemma_index = {label: index for index, label in enumerate(self.vocab["lemma_rules"])}
        feat_indexes = [{label: index for index, label in enumerate(self.vocab["feats"][key])}
                        for key in self.feature_names]
        for batch, sentence in enumerate(sentences):
            for word in sentence.words:
                if word.head is None:
                    raise ValueError("Training sentence is missing HEAD")
                batch_ids.append(batch)
                dep_ids.append(word.id)
                gold_heads.append(word.head)
                gold_rels.append(rel_index[word.deprel.split(":", 1)[0]])
                upos_targets.append(upos_index.get(word.upos, -100))
                rule = lemma_rule(word.form, word.lemma) if word.lemma != "_" else ""
                lemma_targets.append(lemma_index.get(rule, -100))
                observed_feats = parse_feats(word.feats)
                for key_index, key in enumerate(self.feature_names):
                    feat_targets[key_index].append(
                        feat_indexes[key_index].get(observed_feats.get(key, "_"), -100)
                    )
        b = torch.tensor(batch_ids, dtype=torch.long, device=device)
        d = torch.tensor(dep_ids, dtype=torch.long, device=device)
        h = torch.tensor(gold_heads, dtype=torch.long, device=device)
        r = torch.tensor(gold_rels, dtype=torch.long, device=device)
        losses = {
            "arc": F.cross_entropy(arc_logits[b, d].float(), h),
            "rel": F.cross_entropy(rel_logits[b, d, h].float(), r),
        }
        tag_targets = torch.tensor(upos_targets, dtype=torch.long, device=device)
        if torch.any(tag_targets != -100):
            losses["upos"] = F.cross_entropy(upos_logits[b, d].float(), tag_targets,
                                             ignore_index=-100)
        lemma_ids = torch.tensor(lemma_targets, dtype=torch.long, device=device)
        if torch.any(lemma_ids != -100):
            losses["lemma"] = F.cross_entropy(lemma_logits[b, d].float(), lemma_ids,
                                              ignore_index=-100)
        feature_losses = []
        for logits, targets in zip(feature_logits, feat_targets, strict=True):
            ids = torch.tensor(targets, dtype=torch.long, device=device)
            if torch.any(ids != -100):
                feature_losses.append(F.cross_entropy(logits[b, d].float(), ids,
                                                      ignore_index=-100))
        if feature_losses:
            losses["feats"] = torch.stack(feature_losses).mean()
        output["losses"] = losses
        output["loss"] = sum(losses.values())
        return output

    @torch.no_grad()
    def predict(self, sentences: list[Sentence]) -> list[list[dict]]:
        was_training = self.training
        self.eval()
        output = self.forward(sentences, compute_loss=False)
        rel_labels = self.vocab["deprel"]
        predictions = []
        for batch, sentence in enumerate(sentences):
            length = output["lengths"][batch]
            scores = output["arc"][batch, :length + 1, :length + 1].float().T.cpu().tolist()
            heads = decode_single_root(scores)
            rows = []
            for word in sentence.words:
                dep, head = word.id, heads[word.id]
                upos_id = int(output["upos"][batch, dep].argmax())
                lemma_id = int(output["lemma"][batch, dep].argmax())
                rel_scores = output["rel"][batch, dep, head].float().clone()
                if head != 0 and "root" in rel_labels:
                    rel_scores[rel_labels.index("root")] = -1e4
                relation = "root" if head == 0 else rel_labels[int(rel_scores.argmax())]
                features = {}
                for key, logits in zip(self.feature_names, output["feats"], strict=True):
                    value = self.vocab["feats"][key][int(logits[batch, dep].argmax())]
                    if value != "_":
                        features[key] = value
                rows.append({
                    "lemma": apply_lemma_rule(word.form, self.vocab["lemma_rules"][lemma_id]),
                    "upos": self.vocab["upos"][upos_id],
                    "feats": format_feats(features), "head": head, "deprel": relation,
                })
            predictions.append(rows)
        if was_training:
            self.train()
        return predictions
