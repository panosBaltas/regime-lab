"""Exclusive-create artifacts and verifiable, locally append-only research records."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    # Git checkout line-ending conventions must not change the model identity.
    return {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(root.glob("*.py"))}


def write_new(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".writer.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"Another writer (or an interrupted writer) holds {path}") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def read_chain(directory):
    rows, previous = [], None
    for number, path in enumerate(sorted(Path(directory).glob("[0-9]*.json"))):
        record = json.loads(path.read_text(encoding="utf-8"))
        payload = {k: v for k, v in record.items() if k != "sha256"}
        if (record.get("sequence") != number or record.get("previous_sha256") != previous
                or record.get("sha256") != digest(payload)
                or path.name != f"{number:08d}.json"):
            raise ValueError(f"Journal chain mismatch at {path}")
        previous = record["sha256"]
        rows.append(record)
    return rows


def append_record(directory, payload):
    """Caller must hold exclusive_lock during any read/compute/write transaction."""
    rows = read_chain(directory)
    record = {"sequence": len(rows), "previous_sha256": rows[-1]["sha256"] if rows else None,
              "recorded_at": now_utc(), "payload": payload}
    record["sha256"] = digest(record)
    write_new(Path(directory) / f"{len(rows):08d}.json", record)
    return record
