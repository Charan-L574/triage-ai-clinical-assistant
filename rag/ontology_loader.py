"""
rag/ontology_loader.py — Downloads and indexes the Human Disease Ontology.

Downloads HumanDO.obo, parses it, generates embeddings via HuggingFace,
and builds a FAISS IndexFlatL2 index saved to disk.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

import config
from utils.logger import get_logger

logger = get_logger(__name__)

METADATA_FILE = "metadata.json"
INDEX_FILE = "faiss.index"
OBO_FILE = "HumanDO.obo"


class OntologyLoader:
    """Downloads and processes the Human Disease Ontology OBO file."""

    def __init__(self):
        self.index_dir = Path(config.FAISS_INDEX_PATH)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def obo_path(self) -> Path:
        return self.index_dir / OBO_FILE

    @property
    def index_path(self) -> Path:
        return self.index_dir / INDEX_FILE

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / METADATA_FILE

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_index(self) -> bool:
        """Full build: download → parse → embed → FAISS save. Returns True on success."""
        try:
            logger.info("Starting Human Disease Ontology index build...")
            t0 = time.monotonic()

            obo_text = self._download_obo()
            if not obo_text:
                return False

            terms = self._parse_obo(obo_text)
            logger.info(f"Parsed {len(terms)} disease terms from OBO")

            if not terms:
                return False

            texts = [t["document"] for t in terms]
            embeddings = self._embed_in_batches(texts, batch_size=32)

            if embeddings is None or len(embeddings) == 0:
                logger.error("Embedding generation failed — index not built")
                return False

            self._build_faiss(embeddings, terms)
            elapsed = int((time.monotonic() - t0) / 60)
            logger.info(
                f"Index build complete: {len(terms)} terms, ~{elapsed} minutes"
            )
            return True

        except Exception as exc:
            logger.error(f"Index build failed: {exc}")
            return False

    # ── Download ───────────────────────────────────────────────────────────────

    def _download_obo(self) -> Optional[str]:
        if self.obo_path.exists():
            logger.info(f"Using cached OBO file: {self.obo_path}")
            return self.obo_path.read_text(encoding="utf-8")

        logger.info(f"Downloading HumanDO.obo from {config.ONTOLOGY_URL}")
        try:
            resp = requests.get(config.ONTOLOGY_URL, timeout=120)
            resp.raise_for_status()
            self.obo_path.write_bytes(resp.content)
            logger.info(f"Downloaded {len(resp.content)} bytes")
            return resp.text
        except Exception as exc:
            logger.error(f"OBO download failed: {exc}")
            return None

    # ── OBO Parser ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_obo(text: str) -> List[Dict]:
        """Parse OBO format and return list of term dicts."""
        terms: List[Dict] = []
        blocks = text.split("\n[Term]")

        for block in blocks[1:]:
            term = OntologyLoader._parse_term_block(block)
            if term:
                terms.append(term)

        return terms

    @staticmethod
    def _parse_term_block(block: str) -> Optional[Dict]:
        lines = block.strip().split("\n")
        data: Dict = {}
        synonyms: List[str] = []
        icd_codes: List[str] = []

        for line in lines:
            line = line.strip()
            if line.startswith("id:"):
                data["id"] = line[3:].strip()
            elif line.startswith("name:"):
                data["name"] = line[5:].strip()
            elif line.startswith("def:"):
                # Extract text between first pair of quotes
                match = re.search(r'"([^"]+)"', line)
                if match:
                    data["def"] = match.group(1)
            elif line.startswith("synonym:"):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    synonyms.append(match.group(1))
            elif line.startswith("xref:"):
                xref = line[5:].strip()
                if xref.startswith("ICD10CM:") or xref.startswith("ICD9CM:"):
                    icd_codes.append(xref)
            elif line.strip() == "is_obsolete: true":
                return None  # Skip obsolete terms

        if not data.get("id") or not data.get("name"):
            return None

        # Build document string
        parts = [data["name"]]
        if data.get("def"):
            parts.append(data["def"])
        if synonyms:
            parts.append("Synonyms: " + "; ".join(synonyms[:3]))

        data["synonyms"] = synonyms
        data["icd_codes"] = icd_codes
        data["document"] = " | ".join(parts)
        return data

    # ── Embeddings via HuggingFace ─────────────────────────────────────────────

    def _embed_in_batches(
        self, texts: List[str], batch_size: int = 32
    ) -> Optional[np.ndarray]:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=config.HF_TOKEN)
            all_embeddings: List[np.ndarray] = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                logger.info(
                    f"Embedding batch {i // batch_size + 1}/"
                    f"{(len(texts) + batch_size - 1) // batch_size}"
                )
                try:
                    result = client.feature_extraction(
                        batch,
                        model=config.EMBEDDING_MODEL_ID,
                    )
                    arr = np.array(result, dtype=np.float32)
                    # Handle 3D output (batch, seq_len, dim) → mean pool
                    if arr.ndim == 3:
                        arr = arr.mean(axis=1)
                    all_embeddings.append(arr)
                    time.sleep(0.5)  # Be polite to free-tier rate limits
                except Exception as exc:
                    logger.warning(f"Batch {i} embedding error: {exc}")
                    continue

            if not all_embeddings:
                return None
            return np.vstack(all_embeddings)

        except Exception as exc:
            logger.error(f"Embedding generation error: {exc}")
            return None

    # ── FAISS index ────────────────────────────────────────────────────────────

    def _build_faiss(self, embeddings: np.ndarray, terms: List[Dict]) -> None:
        import faiss  # type: ignore

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        logger.info(f"FAISS index saved to {self.index_path} (dim={dim})")

        # Metadata: position → disease info
        metadata = {}
        for i, term in enumerate(terms):
            metadata[str(i)] = {
                "name": term.get("name", ""),
                "doid": term.get("id", ""),
                "definition": term.get("def", ""),
                "icd_codes": term.get("icd_codes", []),
                "document": term.get("document", ""),
            }
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=None), encoding="utf-8"
        )
        logger.info(f"Metadata saved to {self.metadata_path}")
