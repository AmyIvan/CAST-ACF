# -*- coding: utf-8 -*-
"""
I/O utilities for reading JSONL files and tar archives.
"""
from __future__ import annotations
import os
import io
import tarfile
import json
from typing import Dict, Any, Iterable, Optional
from dataclasses import dataclass

try:
    import ujson
    _HAS_UJSON = True
except ImportError:
    _HAS_UJSON = False


def _json_loads(s: str) -> Any:
    """Load JSON with ujson fallback to standard json."""
    if _HAS_UJSON:
        try:
            return ujson.loads(s)
        except Exception:
            pass
    return json.loads(s)


def read_jsonl(fp) -> Iterable[Dict[str, Any]]:
    """Read JSONL from a file-like object."""
    for line in fp:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", errors="ignore")
        line = line.strip()
        if not line:
            continue
        try:
            yield _json_loads(line)
        except Exception:
            continue


def read_json_file(path: str) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return _json_loads(txt)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Iterate over a JSONL file."""
    with open(path, "rb") as f:
        for obj in read_jsonl(f):
            yield obj


def write_jsonl(path: str, records: Iterable[Dict[str, Any]], mode: str = "w"):
    """Write records to a JSONL file."""
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            if _HAS_UJSON:
                f.write(ujson.dumps(rec, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dataclass
class TarJsonlIndex:
    """Index for a tar.gz archive containing JSONL files."""
    path: str
    members: Dict[str, tarfile.TarInfo]


def index_cand_tar(tar_path: str) -> TarJsonlIndex:
    """
    Index a tar.gz archive containing JSONL files.
    Each JSONL file should be named as {topic_id}.jsonl.
    """
    tf = tarfile.open(tar_path, "r:gz")
    members: Dict[str, tarfile.TarInfo] = {}
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = os.path.basename(m.name)
        if name.endswith(".jsonl"):
            key = name[:-6]  # "1234.jsonl" -> "1234"
            members[key] = m
    tf.close()
    return TarJsonlIndex(path=tar_path, members=members)


def stream_cand_jsonl_from_tar(
    index: TarJsonlIndex, 
    topic_id: str | int
) -> Iterable[Dict[str, Any]]:
    """Stream JSONL records for a specific topic from a tar archive."""
    topic_id = str(topic_id)
    member = index.members.get(topic_id)
    if not member:
        return []
    with tarfile.open(index.path, "r:gz") as tf:
        fp = tf.extractfile(member)
        if fp is None:
            return []
        for obj in read_jsonl(fp):
            yield obj
