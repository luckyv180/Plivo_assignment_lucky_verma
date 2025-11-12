# src/ranker_onnx.py
from typing import List, Tuple
import numpy as np
import re

# Optional imports guarded to allow partial environments
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None

class PseudoLikelihoodRanker:
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 onnx_path: str = None,
                 device: str = "cpu",
                 max_length: int = 64,
                 max_mask_positions: int = 16):
        self.max_length = max_length
        self.model_name = model_name
        self.onnx = None
        self.torch_model = None
        self.device = device
        self.tokenizer = None
        self.max_mask_positions = max_mask_positions
        if onnx_path and ort is not None:
            self._init_onnx(onnx_path)
        elif AutoTokenizer is not None and AutoModelForMaskedLM is not None:
            self._init_torch()
        else:
            raise RuntimeError("Neither onnxruntime nor transformers/torch are available. Please install requirements.")

    def _init_onnx(self, onnx_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.onnx = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    def _init_torch(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.torch_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.torch_model.eval()
        self.torch_model.to(self.device)

    def _sample_positions(self, positions: List[int]) -> List[int]:
        # Limit positions to at most max_mask_positions, sample evenly
        if len(positions) <= self.max_mask_positions:
            return positions
        idxs = np.linspace(0, len(positions)-1, self.max_mask_positions).astype(int)
        return [positions[int(i)] for i in idxs]

    def _score_with_onnx(self, text: str) -> float:
        # This uses per-position ONNX run (batch=1) to be compatible with the exported model
        toks = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = toks["input_ids"]           # (1, L)
        attn = toks["attention_mask"]           # (1, L)

        L = int(attn[0].sum())
        positions = list(range(1, L - 1))
        if not positions:
            return 0.0
        positions = self._sample_positions(positions)

        mask_id = self.tokenizer.mask_token_id
        seq = input_ids[0]                      # (L,)
        total = 0.0

        # Run ONNX once per masked position (keeps attention_mask shape (1, L))
        for pos in positions:
            masked = seq.copy()
            orig_token_id = int(masked[pos])
            masked[pos] = mask_id
            ort_inputs = {
                "input_ids": masked[None, :].astype(np.int64),   # (1, L)
                "attention_mask": attn.astype(np.int64),         # (1, L)
            }
            logits = self.onnx.run(None, ort_inputs)[0]  # (1, L, V)
            logits_pos = logits[0, pos, :]  # (V,)
            m = logits_pos.max()
            # stable log-softmax
            log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum())
            total += float(log_probs[orig_token_id])
        return total

    def _score_with_torch(self, text: str) -> float:
        import torch
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        seq = input_ids[0]
        L = int(attn.sum())
        positions = list(range(1, L-1))
        if not positions:
            return 0.0
        # sample positions if too many
        if len(positions) > self.max_mask_positions:
            idxs = np.linspace(0, len(positions)-1, self.max_mask_positions).astype(int)
            positions = [positions[int(i)] for i in idxs]

        batch = seq.unsqueeze(0).repeat(len(positions), 1)
        for i, pos in enumerate(positions):
            batch[i, pos] = self.tokenizer.mask_token_id
        batch_attn = attn.repeat(len(positions), 1)
        with torch.no_grad():
            out = self.torch_model(input_ids=batch, attention_mask=batch_attn).logits  # [B, L, V]
            orig = seq.unsqueeze(0).repeat(len(positions), 1)
            rows = torch.arange(len(positions))
            cols = torch.tensor(positions)
            token_ids = orig[rows, cols]
            logits_pos = out[rows, cols, :]
            log_probs = logits_pos.log_softmax(dim=-1)
            picked = log_probs[torch.arange(len(rows)), token_ids]
        return float(picked.sum().item())

    def score(self, sentences: List[str]) -> List[float]:
        return [self._score_with_onnx(s) if self.onnx is not None else self._score_with_torch(s) for s in sentences]

    # Light validators for short-circuiting (keep these cheap)
    EMAIL_REGEX = re.compile(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$')
    NUMBER_REGEX = re.compile(r'^[\d,₹]+$')

    def _is_valid_email(self, s: str) -> bool:
        return bool(self.EMAIL_REGEX.match(s.strip()))

    def _is_valid_number(self, s: str) -> bool:
        s2 = s.strip().replace(',', '').replace('₹', '')
        return len(s2) >= 2 and s2.isdigit()

    def choose_best(self, candidates: List[str]) -> str:
        # tiny candidate cap (defensive; rules already cap to 3)
        if len(candidates) > 5:
            candidates = candidates[:5]
        # 1) Short-circuit: valid email -> return immediately
        for c in candidates:
            if self._is_valid_email(c):
                return c
        # 2) Short-circuit: valid number/currency -> choose the one with most digits
        num_cands = [c for c in candidates if self._is_valid_number(c)]
        if num_cands:
            best_num = max(num_cands, key=lambda x: len(re.sub(r'[^0-9]', '', x)))
            return best_num
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        # Otherwise score (cap to 3 for speed)
        capped = candidates[:3]
        scores = self.score(capped)
        i = int(np.argmax(scores))
        return capped[i]
