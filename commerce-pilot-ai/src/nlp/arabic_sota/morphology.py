"""Arabic stemming/rooting views for the SARF reproduction, matching the exact libraries
verified in the audited SARF repo (PyArabic/Tashaphyne/Qalsadi -- see sarf_protocol_lock.json).
Precomputes and caches views to disk since Qalsadi's morphological analysis is slow (~10-50ms/word).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tashaphyne.stemming import ArabicLightStemmer

_ALS = ArabicLightStemmer()
_analex = None  # lazy -- Qalsadi Analex() init is slow, only build if cache miss


def _get_analex():
    global _analex
    if _analex is None:
        from qalsadi.analex import Analex
        _analex = Analex()
    return _analex


def stem_sentence(text: str) -> str:
    words = str(text).split()
    out = []
    for w in words:
        try:
            _ALS.light_stem(w)
            s = _ALS.get_stem()
            out.append(s if s else w)
        except Exception:
            out.append(w)
    return " ".join(out)


def root_sentence(text: str) -> str:
    an = _get_analex()
    words = str(text).split()
    try:
        results = an.check_text(str(text))
    except Exception:
        return " ".join(words)
    out = []
    for i, w in enumerate(words):
        root = None
        if i < len(results) and results[i]:
            try:
                root = results[i][0].get_root()
            except Exception:
                root = None
        out.append(root if root else w)
    return " ".join(out)


def compute_views_cached(sentences: list[str], cache_path: Path) -> dict:
    """Returns {"stem": [...], "root": [...]} aligned to `sentences`, cached to cache_path
    keyed by a hash of the exact sentence list (so a changed dataset invalidates the cache)."""
    key = hashlib.sha256("\n".join(sentences).encode("utf-8")).hexdigest()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return cached
    stems = [stem_sentence(s) for s in sentences]
    roots = [root_sentence(s) for s in sentences]
    payload = {"key": key, "n": len(sentences), "stem": stems, "root": roots}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload
