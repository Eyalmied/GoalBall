"""
Appearance matching: how much does this person look like that player?

Two off-the-shelf embedding models turn a crop into a vector whose direction
encodes appearance; cosine similarity then compares crops.

    osnet    OSNet, built specifically for person re-identification. The
             stronger of the two here. Needs the torchreid package.
    dinov2   A general self-supervised vision model. Easier to install, and a
             reasonable fallback.

Two standard tricks are applied:

    FLIP AUGMENTATION   Each crop is embedded twice, once mirrored, and the two
                        vectors averaged. Players face every direction across a
                        throw, and this removes left/right sensitivity for
                        almost no cost.

    MULTI-SHOT MATCHING Every gallery crop stays its own labelled point instead
                        of being averaged into one prototype per player, so a
                        player's front, back and prone appearances are all kept.

HONEST LIMITS
    Both teams wear identical uniforms and blackout eyeshades. On clean gallery
    crops OSNet still only reaches ~61% at telling teammates apart, and ~44% on
    live match frames. That is why identity in this pipeline is decided by
    court position, with appearance available only as a tie-break. Do not raise
    the appearance weight expecting improvement - it was measured to hurt.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import logs

CHOICES = ("osnet", "dinov2")


def _l2(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class OsnetEmbedder:
    name = "osnet"
    label = "OSNet (person re-identification)"

    def __init__(self, model_name: str = "osnet_x1_0"):
        extractor_class = None
        for module in ("torchreid.reid.utils", "torchreid.utils"):
            try:
                extractor_class = __import__(
                    module, fromlist=["FeatureExtractor"]).FeatureExtractor
                break
            except Exception:
                continue
        if extractor_class is None:
            logs.fail("the OSNet appearance model is not installed",
                      "run:  python -m pip install torchreid torch tensorboard\n"
                      "      or switch model with  --appearance dinov2")
        device = _device()
        if device == "mps":
            device = "cpu"      # torchreid is not reliable on Apple's MPS
        with logs.muted():
            self.extractor = extractor_class(model_name=model_name, device=device)
        self.device = device

    def embed(self, crops: list) -> np.ndarray:
        return _l2(self.extractor(crops).cpu().numpy())


class DinoV2Embedder:
    name = "dinov2"
    label = "DINOv2 (general vision features)"

    def __init__(self, model_id: str = "facebook/dinov2-small"):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            logs.fail("the DINOv2 appearance model is not installed",
                      "run:  python -m pip install transformers torch pillow\n"
                      "      or switch model with  --appearance osnet")
        self.torch = torch
        self.device = _device()
        with logs.muted():
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, crops: list) -> np.ndarray:
        from PIL import Image
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                  for c in crops]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model(**inputs)
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        return _l2(features.float().cpu().numpy())


def make(name: str):
    """Build the requested embedder and say which one is actually in use."""
    name = str(name).lower()
    if name not in CHOICES:
        logs.fail(f"--appearance must be one of {', '.join(CHOICES)} (got '{name}')")
    embedder = OsnetEmbedder() if name == "osnet" else DinoV2Embedder()
    logs.ok(f"appearance model: {embedder.label}, running on {embedder.device}")
    return embedder


def embed(embedder, crops: list, flip_augment: bool = True) -> np.ndarray:
    """Embed a list of BGR crops, optionally averaging with their mirrors."""
    if not crops:
        return np.zeros((0, 1), dtype="float32")
    vectors = embedder.embed(crops)
    if flip_augment:
        mirrored = [np.ascontiguousarray(c[:, ::-1]) for c in crops]
        vectors = _l2(vectors + embedder.embed(mirrored))
    return vectors


def reference_bank(embedder, crops: dict, max_per_player: int = 0,
                   flip_augment: bool = True) -> tuple:
    """
    Embed every gallery crop as a labelled point.

    Returns (vectors, player_ids) with one row per reference image.
    """
    vector_blocks, label_blocks = [], []
    for pid in sorted(crops):
        images = crops[pid][:max_per_player] if max_per_player else crops[pid]
        if not images:
            logs.warn(f"player {pid} has no reference images and can never be matched",
                      "re-tag the gallery so every player has crops")
            continue
        vector_blocks.append(embed(embedder, images, flip_augment))
        label_blocks.append(np.full(len(images), pid, dtype=int))
    if not vector_blocks:
        logs.fail("no reference images could be embedded",
                  "rebuild the gallery with scripts/build_gallery.py")
    vectors = np.concatenate(vector_blocks)
    labels = np.concatenate(label_blocks)
    logs.ok(f"appearance reference bank: {len(labels)} images across "
            f"{len(set(labels.tolist()))} players"
            + (" (each also mirrored)" if flip_augment else ""))
    return vectors, labels


def similarity_to_players(query: np.ndarray, vectors: np.ndarray,
                          labels: np.ndarray, player_ids: list,
                          top_k: int = 3) -> dict:
    """
    {player_id -> 0..1 similarity} for one query vector.

    A player's score is the mean of its `top_k` best-matching reference images,
    which is steadier than a single best match and steadier than the average
    over all of them.
    """
    scores = {}
    for pid in player_ids:
        own = vectors[labels == pid]
        if len(own) == 0:
            scores[pid] = 0.0
            continue
        sims = own @ query
        keep = min(top_k, len(sims))
        scores[pid] = float(np.sort(sims)[-keep:].mean())
    return scores
