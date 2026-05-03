"""
rag/retriever.py — RAG retriever for the Human Disease Ontology.

Manages FAISS index lifecycle (lazy init in background thread) and
provides semantic disease context retrieval for LLM prompts.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class DiseaseRetriever:
    """
    Manages FAISS index for disease context retrieval.
    Initialization is non-blocking — run in background thread.
    If index is not ready, returns empty context (pipeline continues).
    """

    def __init__(self):
        self._index = None
        self._metadata: Dict[str, Dict] = {}
        self._ready = False
        self._init_lock = threading.Lock()

    # ── Initialization ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Called at startup. Non-blocking — spawns a background thread.
        Builds index if not already cached on disk.
        """
        thread = threading.Thread(target=self._init_worker, daemon=True)
        thread.start()

    def _init_worker(self) -> None:
        with self._init_lock:
            index_dir = Path(config.FAISS_INDEX_PATH)
            index_path = index_dir / "faiss.index"
            metadata_path = index_dir / "metadata.json"

            if not index_path.exists():
                logger.info("FAISS index not found — building from ontology")
                from rag.ontology_loader import OntologyLoader
                loader = OntologyLoader()
                success = loader.build_index()
                if not success:
                    logger.error("Index build failed — RAG disabled")
                    return

            try:
                import faiss  # type: ignore

                self._index = faiss.read_index(str(index_path))
                self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._ready = True
                logger.info(
                    f"FAISS index loaded: {self._index.ntotal} vectors, "
                    f"{len(self._metadata)} metadata entries"
                )
            except Exception as exc:
                logger.error(f"FAISS index load failed: {exc}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = None) -> tuple[str, list[dict]]:
        """
        Embed the query and return top_k disease descriptions and their metadata.
        Returns empty string and list if index not ready or retrieval fails.
        """
        if top_k is None:
            top_k = config.RAG_TOP_K

        if not self._ready or self._index is None:
            logger.warning("RAG retriever not ready — returning empty context")
            return "", []

        try:
            query_vec = self._embed_query(query)
            if query_vec is None:
                return "", []

            distances, indices = self._index.search(
                query_vec.reshape(1, -1), top_k
            )
            results: List[str] = []
            diseases: List[dict] = []
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                entry = self._metadata.get(str(idx))
                if not entry:
                    continue
                
                # Add to raw list for UI
                diseases.append({
                    "name": entry['name'],
                    "icd_codes": entry.get('icd_codes', []),
                    "distance": float(distances[0][i])
                })
                
                parts = [f"• {entry['name']} ({entry['doid']})"]
                if entry.get("icd_codes"):
                    parts.append(f"  ICD: {', '.join(entry['icd_codes'][:3])}")
                if entry.get("definition"):
                    # Truncate long definitions
                    defn = entry["definition"][:300]
                    if len(entry["definition"]) > 300:
                        defn += "..."
                    parts.append(f"  Definition: {defn}")
                results.append("\n".join(parts))

            context = "\n\n".join(results)
            logger.info(f"RAG retrieved {len(results)} disease entries for query")
            return context, diseases

        except Exception as exc:
            logger.warning(f"RAG retrieval error: {exc}")
            return "", []

    @staticmethod
    def _embed_query(text: str) -> Optional[np.ndarray]:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=config.HF_TOKEN)
            result = client.feature_extraction(
                text[:512], model=config.EMBEDDING_MODEL_ID
            )
            arr = np.array(result, dtype=np.float32)
            # Handle 2D/3D output
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            elif arr.ndim == 3:
                arr = arr.mean(axis=(0, 1))
            return arr
        except Exception as exc:
            logger.warning(f"Query embedding error: {exc}")
            return None
