"""
Phase 0 connectivity check for Qdrant Cloud.

Reads QDRANT_URL / QDRANT_API_KEY from config (i.e. from .env), then:
  1. connects and lists existing collections,
  2. creates a throwaway collection,
  3. upserts one point and reads it back,
  4. deletes the throwaway collection.

Prints "QDRANT OK" and exits 0 on success; prints a clear error and exits
non-zero on failure.

Usage (from the repo root):
    python scripts/check_qdrant.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

# Allow `python scripts/check_qdrant.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:
    print("ERROR: qdrant-client is not installed. Run: pip install -r requirements.txt")
    raise SystemExit(1)

from config import settings


def main() -> int:
    if not settings.qdrant_url or not settings.qdrant_api_key:
        print(
            "ERROR: QDRANT_URL and/or QDRANT_API_KEY are empty.\n"
            "Copy .env.example to .env and fill in your Qdrant Cloud cluster URL "
            "and API key."
        )
        return 1

    print(f"Connecting to {settings.qdrant_url} ...")
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
        collections = client.get_collections()
    except Exception as exc:  # noqa: BLE001 - surface the real error to the user
        print(f"ERROR: could not connect to Qdrant: {exc}")
        return 1

    print(f"Connected. Existing collections: {[c.name for c in collections.collections]}")

    probe = f"_healthcheck_{int(time.time())}"
    try:
        client.create_collection(
            collection_name=probe,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        point_id = str(uuid.uuid4())
        client.upsert(
            collection_name=probe,
            points=[
                PointStruct(
                    id=point_id,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"probe": True},
                )
            ],
        )
        fetched = client.retrieve(collection_name=probe, ids=[point_id])
        if not fetched:
            print("ERROR: wrote a probe point but could not read it back.")
            return 1
        print("Write + read round-trip succeeded.")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Qdrant read/write probe failed: {exc}")
        return 1
    finally:
        try:
            client.delete_collection(collection_name=probe)
            print(f"Cleaned up probe collection '{probe}'.")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not delete probe collection '{probe}': {exc}")

    print("\nQDRANT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
