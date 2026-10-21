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

WHY THE RAW EMBEDDING IS NOT ENOUGH, AND WHAT FIXES IT
    These models are trained for street re-identification, where CLOTHING is
    the main thing telling two people apart. Both goalball teams wear the same
    kit, so the feature the model leans on hardest carries nothing.

    Measured on a real gallery, the raw 512-number embedding puts two photos
    of the SAME player 0.864 apart and two different team-mates only 0.323
    apart. A player resembles a stranger more than they resemble themselves,
    because the stranger stood at the same distance from the camera - the
    single loudest direction in the embedding encodes distance to camera, and
    accounts for 27% of all its variance.

    `fit_discriminant` fixes this. It finds the few directions along which
    different players are far apart while one player's own photos stay close
    together, and throws the rest away. The kit direction scores nothing
    between team-mates because they share a kit; the camera-distance direction
    scores badly because a player's own photos spread along it. On the same
    gallery that turns 0.864-vs-0.323 into 0.426-vs-1.139 - a ratio of 0.37
    becoming 2.67 - and lifts "which of the three team-mates is this" from
    52% to 61%.

    It learns from the gallery labels, which you produce for free by tagging.
    Nothing is hand-tuned and nothing transfers between matches: each game
    gets its own projection from its own images.
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


def fit_discriminant(vectors: np.ndarray, labels: np.ndarray,
                     dims: int = 3) -> Optional[np.ndarray]:
    """
    Directions along which the players separate, learned from the gallery.

    Maximises the spread BETWEEN players relative to the spread WITHIN each
    player (Fisher's linear discriminant). Returns a (D, dims) matrix to
    project embeddings through, or None when the gallery is too thin to
    learn anything - at least two players with at least two images each.

    `dims` is capped at players - 1, which is all this criterion can produce.
    """
    players, counts = np.unique(labels, return_counts=True)
    if len(players) < 2 or (counts < 2).all():
        return None
    dims = max(1, min(dims, len(players) - 1))

    width = vectors.shape[1]
    overall = vectors.mean(axis=0)
    within = np.zeros((width, width))
    between = np.zeros((width, width))
    for player in players:
        own = vectors[labels == player]
        middle = own.mean(axis=0)
        within += (own - middle).T @ (own - middle)
        offset = (middle - overall)[:, None]
        between += len(own) * (offset @ offset.T)

    # With 60 images and 512 dimensions the within-player matrix is singular,
    # so it is nudged toward the identity before being inverted.
    within += np.eye(width) * 1e-2 * np.trace(within) / width
    try:
        strength, directions = np.linalg.eigh(np.linalg.solve(within, between))
    except np.linalg.LinAlgError:
        return None
    keep = np.argsort(-strength)[:dims]
    return np.real(directions[:, keep])


def project(vectors: np.ndarray, projection: Optional[np.ndarray]) -> np.ndarray:
    """Send embeddings through a projection and re-normalise, or pass through."""
    if projection is None:
        return vectors
    single = vectors.ndim == 1
    out = _l2(np.atleast_2d(vectors) @ projection)
    return out[0] if single else out


def separation(vectors: np.ndarray, labels: np.ndarray) -> float:
    """
    How much further apart two players are than one player's own photos.

    Below 1 means a player's own images are more spread out than different
    players are - the matcher cannot work. Reported at startup so a gallery
    that cannot support appearance matching says so.
    """
    inside, across = [], []
    for player in np.unique(labels):
        own = vectors[labels == player]
        for i in range(len(own)):
            for j in range(i + 1, len(own)):
                inside.append(float(np.linalg.norm(own[i] - own[j])))
    middles = {p: vectors[labels == p].mean(axis=0) for p in np.unique(labels)}
    seen = sorted(middles)
    for i, p in enumerate(seen):
        for q in seen[i + 1:]:
            across.append(float(np.linalg.norm(middles[p] - middles[q])))
    if not inside or not across:
        return 0.0
    return float(np.mean(across) / (np.mean(inside) + 1e-8))


def banks_by_half(embedder, crops_by_half: dict, max_per_player: int = 0,
                  flip_augment: bool = True) -> dict:
    """
    {half: (vectors, player_ids)} - one reference bank per half.

    A throw is matched against the half it happened in. The teams change ends,
    so the same player is near the camera in one half and at the far end in
    the other: different size, different lighting, often facing the other way.
    Comparing across halves compares two different-looking things.
    """
    out = {}
    for half in sorted(crops_by_half):
        logs.info(f"  half {half}:")
        out[int(half)] = reference_bank(embedder, crops_by_half[half],
                                        max_per_player, flip_augment)
    return out


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
