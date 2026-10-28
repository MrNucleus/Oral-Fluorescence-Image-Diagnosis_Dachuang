#!/usr/bin/env python3
"""Verify or exactly restore one privately archived dataset version.

The current image package stays intact. Archives contain historical image
objects once per SHA-256 and preserve each version's original metadata bytes.
Restoration creates regular files in a new directory outside existing project
data. It never trains, relabels or overwrites a dataset. Failed restores retain
their uniquely named ``.partial`` staging directory for diagnosis.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CHUNK_SIZE = 1024 * 1024


class RestoreError(Exception):
    """A non-sensitive error code suitable for command-line output."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RestoreError(code)


def _relative(value: object) -> str:
    _require(isinstance(value, str) and bool(value), "invalid_relative_path")
    assert isinstance(value, str)
    parts = value.split("/")
    _require(
        not value.startswith("/")
        and "\\" not in value
        and "\x00" not in value
        and not re.match(r"^[A-Za-z]:", value)
        and all(p not in ("", ".", "..") for p in parts),
        "unsafe_relative_path",
    )
    _require(PurePosixPath(value).as_posix() == value, "noncanonical_relative_path")
    return value


def _path_key(value: str) -> str:
    # Also prevent collisions on common case-insensitive, Unicode-normalizing disks.
    return unicodedata.normalize("NFC", value).casefold()


def _safe_file(base: Path, relative: str) -> Path:
    relative = _relative(relative)
    current = base
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        mode = current.lstat().st_mode
        _require(not stat.S_ISLNK(mode), "symbolic_link_source")
        expected = stat.S_ISREG(mode) if index == len(parts) - 1 else stat.S_ISDIR(mode)
        _require(expected, "nonregular_source")
    return current


@contextlib.contextmanager
def _regular_stream(path: Path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), "nonregular_source")
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with stream:
        yield stream


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _read_json(path: Path):
    with _regular_stream(path) as stream:
        return json.load(stream, object_pairs_hook=_unique_json_pairs)


def _root(path: Path) -> Path:
    _require(not path.is_symlink() and path.is_dir(), "invalid_root_directory")
    return path.resolve(strict=True)


def _find_project_root(store: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return _root(explicit)
    for parent in (store, *store.parents):
        marker = parent / "AGENTS.md"
        if marker.is_file() and not marker.is_symlink():
            return parent
    raise RestoreError("project_root_required")


def _records(manifest, version_id: str) -> list[dict]:
    _require(isinstance(manifest, dict), "invalid_manifest")
    _require(manifest.get("version_id") == version_id, "version_id_mismatch")
    records = manifest.get("files")
    _require(isinstance(records, list), "invalid_file_records")
    paths = set()
    for record in records:
        _require(isinstance(record, dict), "invalid_file_record")
        path = _relative(record.get("path"))
        _require(record.get("kind") in ("image", "metadata"), "invalid_record_kind")
        _require(isinstance(record.get("sha256"), str)
                 and bool(SHA256.fullmatch(record["sha256"])), "invalid_sha256")
        _require(type(record.get("bytes")) is int and record["bytes"] >= 0,
                 "invalid_byte_count")
        key = _path_key(path)
        _require(key not in paths, "duplicate_manifest_path")
        paths.add(key)
    for path in paths:
        for parent in PurePosixPath(path).parents:
            _require(parent.as_posix() not in paths, "manifest_file_directory_collision")
    return records


def _zip_members(archive: zipfile.ZipFile, *, objects: bool) -> dict:
    result = {}
    normalized = set()
    for info in archive.infolist():
        name = _relative(info.filename)
        _require(info.orig_filename == info.filename, "noncanonical_zip_member")
        kind = stat.S_IFMT(info.external_attr >> 16)
        _require(not info.is_dir() and kind in (0, stat.S_IFREG),
                 "nonregular_zip_member")
        _require(not (info.flag_bits & 1), "encrypted_zip_member")
        _require(not objects or bool(SHA256.fullmatch(name)), "invalid_object_name")
        key = _path_key(name)
        _require(key not in normalized, "duplicate_zip_member")
        normalized.add(key)
        result[name] = info
    return result


def _stream_verify(stream, record: dict, destination=None) -> int:
    digest = hashlib.sha256()
    total = 0
    while block := stream.read(CHUNK_SIZE):
        total += len(block)
        _require(total <= record["bytes"], "content_size_mismatch")
        digest.update(block)
        if destination is not None:
            destination.write(block)
    _require(total == record["bytes"], "content_size_mismatch")
    _require(digest.hexdigest() == record["sha256"], "content_sha256_mismatch")
    return total


def _destination(root: Path, relative: str):
    parts = relative.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent / part
        parent.mkdir(mode=0o700, exist_ok=True)
        _require(stat.S_ISDIR(parent.lstat().st_mode), "unsafe_staging_directory")
    target = parent / parts[-1]
    # No existing files are accepted, even in this newly created private staging tree.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    return os.fdopen(fd, "wb")


def _outside(path: Path, protected: Path) -> bool:
    return path != protected and not path.is_relative_to(protected)


def _output_path(output: Path, project: Path, store: Path, versions: dict) -> Path:
    _require(not os.path.lexists(output), "output_already_exists")
    _require(output.parent.is_dir(), "output_parent_missing")
    target = output.parent.resolve(strict=True) / output.name
    _require(not os.path.lexists(target), "output_already_exists")
    protected = [project, store]
    for version in versions.values():
        _require(isinstance(version, dict), "invalid_version_entry")
        source = version.get("source_path")
        if isinstance(source, str) and source:
            p = Path(source)
            protected.append((p if p.is_absolute() else project / p).resolve())
    _require(all(_outside(target, p) for p in protected), "output_inside_existing_data")
    return target


def _publish_exclusive(stage: Path, output: Path) -> None:
    """Atomically publish without replacing even an empty concurrently created directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(stage), os.fsencode(output), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1)
    else:
        raise RestoreError("exclusive_atomic_publish_unavailable")
    if result:
        code = ctypes.get_errno()
        raise RestoreError("output_appeared_during_restore" if code in (errno.EEXIST,
                           errno.ENOTEMPTY) else "atomic_publish_failed")


def restore_version(store: Path, version_id: str, output: Path | None = None,
                    project_root: Path | None = None, *, verify_only: bool = False) -> dict:
    """Verify all selected-version bytes or restore them to a new external directory.

    Returns aggregate counts. Source and manifest paths never appear in errors.
    On failure after staging begins, ``RestoreError.partial_directory`` identifies
    the retained private directory. Do not edit or resume it as a frozen dataset.
    """
    stage = None
    try:
        store = _root(Path(store))
        project = _find_project_root(store, project_root)
        catalog = _read_json(_safe_file(store, "catalog.json"))
        _require(isinstance(catalog, dict) and catalog.get("schema_version") == 1,
                 "unsupported_catalog_schema")
        versions = catalog.get("versions")
        _require(isinstance(versions, dict) and version_id in versions, "unknown_version")
        version = versions[version_id]
        _require(isinstance(version, dict), "invalid_version_entry")
        manifest = _read_json(_safe_file(store, _relative(version.get("manifest"))))
        records = _records(manifest, version_id)
        current = catalog.get("current_images")
        _require(isinstance(current, dict), "invalid_current_image_index")
        for sha, record in current.items():
            _require(bool(SHA256.fullmatch(sha)) and isinstance(record, dict),
                     "invalid_current_image_entry")
            _relative(record.get("path"))
            _require(type(record.get("bytes")) is int and record["bytes"] >= 0,
                     "invalid_current_image_entry")
        if verify_only:
            _require(output is None, "verify_only_has_output")
            target = None
        else:
            _require(output is not None, "output_required")
            target = _output_path(Path(output), project, store, versions)
        metadata_file = _safe_file(store, _relative(version.get("metadata")))
        images = [r for r in records if r["kind"] == "image"]
        metadata = [r for r in records if r["kind"] == "metadata"]
        with contextlib.ExitStack() as stack:
            metadata_stream = stack.enter_context(_regular_stream(metadata_file))
            metadata_zip = stack.enter_context(zipfile.ZipFile(metadata_stream))
            meta_members = _zip_members(metadata_zip, objects=False)
            _require(set(meta_members) == {r["path"] for r in metadata},
                     "metadata_inventory_mismatch")
            historical = [r for r in images if r["sha256"] not in current]
            image_zip = None
            image_members = {}
            if historical:
                image_stream = stack.enter_context(
                    _regular_stream(_safe_file(store, "historical_images.zip")))
                image_zip = stack.enter_context(zipfile.ZipFile(image_stream))
                image_members = _zip_members(image_zip, objects=True)
            for record in records:
                if record["kind"] == "metadata":
                    info = meta_members[record["path"]]
                    _require(info.file_size == record["bytes"], "content_size_mismatch")
                elif record["sha256"] in current:
                    entry = current[record["sha256"]]
                    _require(entry["bytes"] == record["bytes"], "content_size_mismatch")
                    _safe_file(project, entry["path"])
                else:
                    info = image_members.get(record["sha256"])
                    _require(info is not None, "historical_object_missing")
                    _require(info.file_size == record["bytes"], "content_size_mismatch")
            if target is not None:
                stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".partial",
                                              dir=target.parent))
                stage.chmod(0o700)
            total_bytes = 0
            for record in records:
                with contextlib.ExitStack() as item_stack:
                    if record["kind"] == "metadata":
                        source = item_stack.enter_context(metadata_zip.open(record["path"]))
                    elif record["sha256"] in current:
                        source = item_stack.enter_context(_regular_stream(
                            _safe_file(project, current[record["sha256"]]["path"])))
                    else:
                        assert image_zip is not None
                        source = item_stack.enter_context(image_zip.open(record["sha256"]))
                    dest = item_stack.enter_context(_destination(stage, record["path"])) \
                        if stage is not None else None
                    total_bytes += _stream_verify(source, record, dest)
        if stage is not None:
            # Read restored bytes back before publishing; validation is not just of sources.
            for record in records:
                with _regular_stream(_safe_file(stage, record["path"])) as stream:
                    _stream_verify(stream, record)
            assert target is not None
            _publish_exclusive(stage, target)
            stage = None
        return {"status": "verified" if verify_only else "restored", "version_id": version_id,
                "files": len(records), "images": len(images), "metadata_files": len(metadata),
                "bytes": total_bytes, "readback_verified": not verify_only}
    except (RestoreError, OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile,
            NotImplementedError, RuntimeError) as error:
        safe = error if isinstance(error, RestoreError) else RestoreError("archive_or_io_error")
        if stage is not None:
            safe.partial_directory = str(stage)
        raise safe from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = restore_version(args.store, args.version, args.output, args.project_root,
                                 verify_only=args.verify_only)
    except RestoreError as error:
        result = {"status": "failed", "error": str(error)}
        partial = getattr(error, "partial_directory", None)
        if partial is not None:
            result.update(partial_directory=partial, partial_retained=True)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
