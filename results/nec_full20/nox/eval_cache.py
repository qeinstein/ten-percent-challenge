"""Identity-keyed response-cache directories for lm-evaluation-harness runs.

The harness SQLite cache key depends only on the request (context and
continuation), not on the model.  A cache directory shared by the NEC package,
the dense model or experimental candidates would therefore silently return one
model's log-probabilities for another.  These helpers make the directory name
carry the model kind and a content hash of the evaluated weights, and refuse
to reuse a populated directory unless the caller opts in AND the recorded
identity matches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

IDENTITY_FILE = "cache_identity.json"
SNAPSHOT_IDENTITY_FILE = "response_cache_identity.json"


def hash_directory(path, chunk_bytes=1 << 22):
    """SHA-256 over sorted relative names, sizes and file-content digests."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    outer = hashlib.sha256()
    count = 0
    for file in sorted(p for p in root.rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and not p.name.endswith(".pyc")):
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(chunk_bytes), b""):
                digest.update(block)
        outer.update(f"{file.relative_to(root).as_posix()}\0{file.stat().st_size}\0"
                     f"{digest.hexdigest()}\n".encode())
        count += 1
    if not count:
        raise ValueError(f"{root} contains no files to hash")
    return outer.hexdigest()


def cache_identity(model_kind, content_sha256, **extra):
    return {"model_kind": model_kind, "content_sha256": content_sha256, **extra}


def cache_directory_name(identity):
    return f"{identity['model_kind']}-{identity['content_sha256'][:16]}"


def _same_identity(recorded, identity, keys):
    return all(recorded.get(key) == identity.get(key) for key in keys)


def prepare_cache_dir(base, identity, allow_reuse=False,
                      match_keys=("model_kind", "content_sha256")):
    """Create/return the identity-specific cache directory.

    A directory that already holds cache files is refused unless ``allow_reuse``
    is true and its recorded identity matches ``match_keys``; a populated
    directory without a recorded identity is never reused.
    """
    directory = Path(base) / cache_directory_name(identity)
    identity_path = directory / IDENTITY_FILE
    populated = directory.is_dir() and any(p.name != IDENTITY_FILE for p in directory.iterdir())
    if populated:
        if not allow_reuse:
            raise FileExistsError(
                f"response cache {directory} is not empty; headline runs need a fresh cache. "
                "Pass --allow-cache-reuse to resume with the SAME model content.")
        if not identity_path.is_file():
            raise ValueError(f"{directory} has cache files but no recorded identity; refusing reuse")
        recorded = json.loads(identity_path.read_text())
        if not _same_identity(recorded, identity, match_keys):
            raise ValueError(f"{directory} was created for a different model "
                             f"({recorded.get('model_kind')}, {recorded.get('content_sha256')})")
    directory.mkdir(parents=True, exist_ok=True)
    if not identity_path.is_file():
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return directory


def snapshot_files_to_restore(output_dir, identity, allow_reuse,
                              match_keys=("model_kind", "content_sha256")):
    """Which ``response_cache_rank*.db`` snapshots in ``output_dir`` may be restored.

    Never restores blindly: requires ``allow_reuse`` and a snapshot identity file
    matching the current model.  Returns ``(files, reason)``.
    """
    output_dir = Path(output_dir)
    snapshots = sorted(output_dir.glob("response_cache_rank*.db"))
    if not snapshots:
        return [], "no snapshots"
    if not allow_reuse:
        return [], "snapshots ignored: --allow-cache-reuse not given"
    identity_path = output_dir / SNAPSHOT_IDENTITY_FILE
    if not identity_path.is_file():
        return [], "snapshots ignored: no recorded identity"
    recorded = json.loads(identity_path.read_text())
    if not _same_identity(recorded, identity, match_keys):
        return [], "snapshots ignored: identity differs from the evaluated model"
    return snapshots, "identity matches"


def write_snapshot_identity(output_dir, identity):
    path = Path(output_dir) / SNAPSHOT_IDENTITY_FILE
    path.write_text(json.dumps(identity, indent=2) + "\n")
    return path
