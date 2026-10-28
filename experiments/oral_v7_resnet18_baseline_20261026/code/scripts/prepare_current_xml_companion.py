#!/usr/bin/env python3
"""Build/verify one XML per current whole image without changing its frozen package.

XML bytes come from the original complete v3r1 evidence locked during the
whole-image migration. This is an annotation companion for inspection; current
whole-image classification still uses dataset.json, not boxes. Verification can
use a relocated --dataset package and resolves image_package_relpath within it.
JPEG dimensions are checked with a standard-library marker parser, not a new
full pixel/clinical review. Existing outputs and validation reports are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import struct
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "_xml_companion_restore", Path(__file__).with_name("restore_dataset_version.py"))
_safe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_safe)
Error = _safe.RestoreError
DATASET = "data/processed/01_当前工作包/oral_binary_v6_image_classification_2026-10-19_v1"
COMPANION = "data/processed/01_当前工作包/当前图片配套XML_2026-10-21_v1"
SUPPORT = "data/processed/02_私密核对资料/03_历史与过程文件/逐图XML补充_2026-10-21_v1"
ORIGINAL = "data/processed/03_历史版本/oral_binary_v3r1_candidate_2026-10-18"
MIGRATION = "data/processed/02_私密核对资料/03_历史与过程文件/整图分类迁移_2026-10-19_v1"
EXPECTED_DATASET_SHA = "5dfb816e0c1f36c418c7e39b2e279c71128f69b2e70eafba192f0279edda4289"
IMAGE_ID = re.compile(r"P[0-9]+_I[0-9]+\Z")
SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def require(condition, code):
    _safe._require(bool(condition), code)


def signature(path):
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode), "nonregular_or_symlink_source")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def digest(path):
    before = signature(path)
    h = hashlib.sha256()
    with _safe._regular_stream(path) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    require(signature(path) == before, "source_changed_during_read")
    return h.hexdigest()


def read_bytes(path):
    before = signature(path)
    with _safe._regular_stream(path) as stream:
        data = stream.read()
    require(signature(path) == before, "source_changed_during_read")
    return data


def write_bytes(path, payload):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def mkdir_private(path, base):
    current = base
    for part in path.relative_to(base).parts:
        current /= part
        current.mkdir(mode=0o700, exist_ok=True)
        require(stat.S_ISDIR(current.lstat().st_mode), "unsafe_directory")


def jpeg_dimensions(path):
    """Read dimensions from a JPEG SOF marker without cropping or decoding pixels."""
    before = signature(path)
    with _safe._regular_stream(path) as stream:
        require(stream.read(2) == b"\xff\xd8", "not_jpeg")
        while True:
            marker = stream.read(1)
            require(marker == b"\xff", "invalid_jpeg_marker")
            while marker == b"\xff":
                marker = stream.read(1)
            require(bool(marker), "truncated_jpeg")
            kind = marker[0]
            require(kind not in (0xDA, 0xD9, 0x00), "jpeg_dimensions_missing")
            if kind in (0xD8, 0x01) or 0xD0 <= kind <= 0xD7:
                continue
            raw = stream.read(2)
            require(len(raw) == 2, "truncated_jpeg")
            length = struct.unpack(">H", raw)[0]
            require(length >= 2, "invalid_jpeg_segment")
            if kind in SOF:
                raw = stream.read(5)
                require(len(raw) == 5 and length >= 7, "truncated_jpeg")
                height, width = struct.unpack(">HH", raw[1:])
                require(width > 0 and height > 0, "invalid_jpeg_dimensions")
                break
            require(len(stream.read(length - 2)) == length - 2, "truncated_jpeg")
    require(signature(path) == before, "source_changed_during_read")
    return width, height


def xml_info(payload, width, height, label):
    require(b"<!DOCTYPE" not in payload and b"<!ENTITY" not in payload, "unsupported_xml_doctype")
    root = ET.fromstring(payload)
    require((int(root.findtext("size/width")), int(root.findtext("size/height")))
            == (width, height), "xml_image_dimension_mismatch")
    objects = root.findall("object")
    expected = "n" if label == 0 else "l"
    require(objects and {o.findtext("name", "").strip() for o in objects} == {expected},
            "xml_candidate_label_mismatch")
    for obj in objects:
        values = [float(obj.findtext(f"bndbox/{name}"))
                  for name in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = values
        require(all(math.isfinite(v) for v in values)
                and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
                "invalid_historical_xml_coordinates")
    return len(objects)


def dataset_records(dataset):
    path = _safe._safe_file(dataset, "dataset.json")
    require(digest(path) == EXPECTED_DATASET_SHA, "unexpected_frozen_dataset")
    data = _safe._read_json(path)
    require(data.get("task") == "image_classification"
            and data.get("class_names") == ["normal_candidate", "lesion_candidate"],
            "unexpected_dataset_task")
    records = data.get("records")
    require(isinstance(records, list) and len(records) == 1117, "unexpected_image_count")
    seen = set()
    paths = set()
    for record in records:
        image_id = record.get("image_id")
        require(isinstance(image_id, str) and IMAGE_ID.fullmatch(image_id), "invalid_image_id")
        require(image_id not in seen, "duplicate_image_id")
        seen.add(image_id)
        split, mode, label = record.get("split"), record.get("modality"), record.get("label")
        require(split in ("train", "val", "test") and mode in ("standard", "fluorescence")
                and type(label) is int and label in (0, 1), "invalid_image_assignment")
        relative = _safe._relative(record.get("image_relpath"))
        class_name = "0_normal" if label == 0 else "1_lesion"
        require(relative == f"{split}/{mode}/images/{class_name}/{image_id}.jpg",
                "image_path_assignment_mismatch")
        require(relative not in paths, "duplicate_image_path")
        paths.add(relative)
        require(type(record.get("width")) is int and type(record.get("height")) is int
                and record["width"] > 0 and record["height"] > 0, "invalid_image_dimensions")
    return records


def checksum_inventory(companion):
    entries = {}
    for line in read_bytes(_safe._safe_file(companion, "CHECKSUMS.sha256")).decode("utf-8").splitlines():
        sha, relative = line.split(maxsplit=1)
        relative = _safe._relative(relative)
        require(_safe.SHA256.fullmatch(sha) and relative not in entries, "invalid_checksum_entry")
        require(digest(_safe._safe_file(companion, relative)) == sha, "companion_checksum_mismatch")
        entries[relative] = sha
    actual = set()
    for directory, directories, files in os.walk(companion, followlinks=False):
        for name in [*directories, *files]:
            path = Path(directory) / name
            require(not path.is_symlink(), "symbolic_link_in_companion")
        for name in files:
            if name == ".DS_Store" or name.startswith("._"):
                continue
            path = Path(directory) / name
            require(stat.S_ISREG(path.lstat().st_mode), "nonregular_companion_file")
            actual.add(path.relative_to(companion).as_posix())
    require(actual == set(entries) | {"CHECKSUMS.sha256"}, "companion_inventory_mismatch")
    return entries


def verify_companion(dataset, companion):
    dataset = _safe._root(dataset)
    companion = _safe._root(companion)
    entries = checksum_inventory(companion)
    manifest = _safe._read_json(_safe._safe_file(companion, "manifest.json"))
    require(manifest.get("schema_version") == 1
            and manifest.get("dataset_sha256") == EXPECTED_DATASET_SHA
            and manifest.get("training_input") is False, "invalid_companion_manifest")
    source_records = {r["image_id"]: r for r in dataset_records(dataset)}
    records = manifest.get("records")
    require(isinstance(records, list) and len(records) == len(source_records), "xml_membership_mismatch")
    seen = set()
    xml_paths = set()
    boxes = 0
    image_bytes = 0
    counts = {"train": 0, "val": 0, "test": 0, "standard": 0, "fluorescence": 0,
              "normal_candidate": 0, "lesion_candidate": 0}
    for record in records:
        image_id = record.get("image_id")
        require(image_id in source_records and image_id not in seen, "xml_membership_mismatch")
        seen.add(image_id)
        original = source_records[image_id]
        for key in ("split", "modality", "label", "width", "height"):
            require(record.get(key) == original[key], "companion_assignment_mismatch")
        relative = _safe._relative(record.get("image_package_relpath"))
        require(relative == original["image_relpath"], "companion_image_path_mismatch")
        image = _safe._safe_file(dataset, relative)
        require(record.get("image_sha256") == original["sha256"] == digest(image),
                "current_image_sha256_mismatch")
        require(jpeg_dimensions(image) == (original["width"], original["height"]),
                "actual_image_dimension_mismatch")
        image_bytes += image.stat().st_size
        xml_relative = _safe._relative(record.get("xml_relpath"))
        require(xml_relative == f"{original['split']}/{original['modality']}/annotations/{image_id}.xml"
                and xml_relative not in xml_paths, "invalid_xml_assignment")
        xml_paths.add(xml_relative)
        payload = read_bytes(_safe._safe_file(companion, xml_relative))
        require(record.get("source_xml_sha256") == record.get("xml_sha256")
                == hashlib.sha256(payload).hexdigest(), "historical_xml_sha256_mismatch")
        n = xml_info(payload, original["width"], original["height"], original["label"])
        require(n == record.get("xml_box_count"), "xml_box_count_mismatch")
        boxes += n
        counts[original["split"]] += 1
        counts[original["modality"]] += 1
        counts["normal_candidate" if original["label"] == 0 else "lesion_candidate"] += 1
    require(boxes == 1219 and seen == set(source_records), "unexpected_complete_xml_scope")
    require(set(entries) == xml_paths | {"manifest.json", "README.md"},
            "companion_manifest_inventory_mismatch")
    return {"status": "passed", "images": len(records), "xml_files": len(xml_paths),
            "historical_xml_objects": boxes, "xml_bytes": sum(
                (companion / relative).stat().st_size for relative in xml_paths),
            "current_image_bytes_verified": image_bytes, "checksum_entries": len(entries),
            "counts": counts, "dataset_sha256": EXPECTED_DATASET_SHA,
            "current_package_modified": False, "pixel_decode_or_clinical_review": False,
            "verification_scope": "XML bytes, pure historical n/l, coordinates, image hashes, "
                                  "JPEG marker dimensions, unchanged image_id/split/modality/label"}


def build_companion(project, dataset, companion):
    require(not os.path.lexists(companion), "companion_already_exists")
    require(companion.parent.is_dir(), "companion_parent_missing")
    output = companion.parent.resolve(strict=True) / companion.name
    source = _safe._root(project / ORIGINAL)
    require(all(_safe._outside(output, p) for p in (dataset, source)), "output_inside_frozen_data")
    evidence_files = {name: _safe._safe_file(project, f"{MIGRATION}/{name}")
                      for name in ("label_confirmation.json", "source_snapshot_before.json")}
    evidence_hashes = {name: digest(path) for name, path in evidence_files.items()}
    confirmations = _safe._read_json(evidence_files["label_confirmation.json"])
    original_inputs = _safe._read_json(evidence_files["source_snapshot_before.json"])["original_inputs"]
    require(isinstance(confirmations, list), "invalid_migration_evidence")
    by_id = {r["image_id"]: r for r in confirmations}
    require(len(by_id) == len(confirmations) == 1117, "invalid_confirmation_membership")
    records = dataset_records(dataset)
    require(set(by_id) == {r["image_id"] for r in records}, "confirmation_membership_mismatch")
    current_signatures = {}
    source_signatures = {}
    companion_records = []
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent))
    stage.chmod(0o700)
    try:
        for record in records:
            image_id = record["image_id"]
            confirmation = by_id[image_id]
            expected_name = "n" if record["label"] == 0 else "l"
            require(confirmation.get("group_id") == record["group_id"]
                    and confirmation.get("candidate_label") == record["label"]
                    and confirmation.get("frozen_image_label") == str(record["label"])
                    and confirmation.get("original_xml_names") == [expected_name],
                    "migration_candidate_evidence_mismatch")
            image = _safe._safe_file(dataset, record["image_relpath"])
            current_signatures[record["image_relpath"]] = signature(image)
            require(digest(image) == record["sha256"], "current_image_sha256_mismatch")
            require(jpeg_dimensions(image) == (record["width"], record["height"]),
                    "actual_image_dimension_mismatch")
            original_relative = _safe._relative(confirmation["original_xml_relpath"])
            require(Path(original_relative).stem == image_id, "source_xml_id_mismatch")
            expected_sha = original_inputs.get(original_relative)
            require(isinstance(expected_sha, str) and _safe.SHA256.fullmatch(expected_sha),
                    "original_xml_hash_evidence_missing")
            original_xml = _safe._safe_file(source, original_relative)
            source_signatures[original_relative] = signature(original_xml)
            payload = read_bytes(original_xml)
            require(hashlib.sha256(payload).hexdigest() == expected_sha, "source_xml_hash_mismatch")
            n = xml_info(payload, record["width"], record["height"], record["label"])
            relative = f"{record['split']}/{record['modality']}/annotations/{image_id}.xml"
            target = stage / relative
            mkdir_private(target.parent, stage)
            write_bytes(target, payload)
            image_project_relative = image.relative_to(project).as_posix() \
                if image.is_relative_to(project) else None
            companion_records.append({
                "image_id": image_id, "split": record["split"], "modality": record["modality"],
                "label": record["label"], "width": record["width"], "height": record["height"],
                "image_project_relpath": image_project_relative,
                "image_package_relpath": record["image_relpath"],
                "image_sha256": record["sha256"], "xml_relpath": relative,
                "source_xml_package_relpath": original_relative,
                "source_xml_sha256": expected_sha, "xml_sha256": expected_sha,
                "xml_box_count": n,
            })
        manifest = {
            "schema_version": 1, "task": "historical_xml_companion", "training_input": False,
            "dataset_sha256": EXPECTED_DATASET_SHA,
            "default_dataset_project_relpath": DATASET,
            "source_xml_package_project_relpath": ORIGINAL,
            "evidence_sha256": {f"{MIGRATION}/{name}": sha for name, sha in evidence_hashes.items()},
            "records": companion_records,
        }
        write_json(stage / "manifest.json", manifest)
        readme = """# 当前完整图片的逐图历史 XML

这里是 1,117 份实体 XML，一张当前图片对应一份同名 XML；没有合并成一个 XML，也没有另复制照片。
按 train/val/test、fluorescence/standard 分目录，XML 文件名与图片 image_id 完全一致。

图片仍在当前 v6 整图分类包。用 manifest.json 的 image_package_relpath 在指定图片包中找图，xml_relpath 在本目录中找 XML；image_project_relpath 是构建时项目内实际路径的辅助索引。搬到独立交接包时可以把图片包放在 01_图片，验证命令的 --dataset 指向它即可。

XML 来自 v3r1 的完整历史标注，原字节、原框、内部历史 filename/path 均保留，不要按内部旧 path 找当前照片。实际文件位置以 manifest 为准。1,117 份 XML 共 1,219 个历史框；v5 的筛框派生版保留 1,200 框，少 19 框，19 份 XML 字节不同，旧版继续保留。此前已完成的方向归一化及其同步框变换不重复进行。

当前模型仍做整张图片分类，训练入口是图片包的 dataset.json，不读取这些框作为训练目标。XML 的 n/l 只作为历史正常/病变候选证据，不代表医生确诊或具体疾病五分类。

CHECKSUMS.sha256 校验本目录原文件；manifest 锁定图片、XML 和迁移证据 SHA-256。验证使用 scripts/prepare_current_xml_companion.py verify --project-root 项目根 --dataset 图片包 --companion 本目录 --report 新的私密验证报告.json。验证报告必须是新文件；不覆盖旧报告。
"""
        write_bytes(stage / "README.md", readme.encode("utf-8"))
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        checksums = "".join(f"{digest(p)}  {p.relative_to(stage).as_posix()}\n" for p in files)
        write_bytes(stage / "CHECKSUMS.sha256", checksums.encode("utf-8"))
        result = verify_companion(dataset, stage)
        for name, path in evidence_files.items():
            require(digest(path) == evidence_hashes[name], "migration_evidence_changed")
        for relative, before in current_signatures.items():
            require(signature(_safe._safe_file(dataset, relative)) == before,
                    "current_image_changed_during_build")
        for relative, before in source_signatures.items():
            require(signature(_safe._safe_file(source, relative)) == before,
                    "source_xml_changed_during_build")
        require(digest(_safe._safe_file(dataset, "dataset.json")) == EXPECTED_DATASET_SHA,
                "frozen_dataset_changed_during_build")
        _safe._publish_exclusive(stage, output)
        stage = None
        result.update(operation="build", original_source_rechecked=True,
                      migration_evidence_rechecked=True, symbolic_links=0,
                      xml_internal_historical_paths_preserved=True, training_executed=False)
        return result
    except BaseException as error:
        if stage is not None and isinstance(error, Exception):
            error.partial_directory = str(stage)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, help="image package directory or its dataset.json")
    parser.add_argument("--companion", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--report", type=Path, help="new aggregate JSON report; never overwrite")
    args = parser.parse_args(argv)
    try:
        project = _safe._root(args.project_root)
        dataset = args.dataset or project / DATASET
        if dataset.name == "dataset.json":
            dataset = dataset.parent
        dataset = _safe._root(dataset)
        companion = args.companion or project / COMPANION
        support = args.support or project / SUPPORT
        require(support.is_relative_to(project), "support_outside_project")
        mkdir_private(support, project)
        report = args.report or support / f"{args.mode}_validation.json"
        require(not os.path.lexists(report) and report.parent.is_dir(), "report_exists_or_parent_missing")
        require(_safe._outside(report.resolve(), dataset), "report_inside_frozen_dataset")
        if args.mode == "build":
            result = build_companion(project, dataset, companion)
        else:
            result = verify_companion(dataset, companion)
            result.update(operation="verify", original_source_rechecked=False,
                          migration_evidence_rechecked=False, training_executed=False)
        write_json(report, result)
    except (Error, OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
        result = {"status": "failed", "error": str(error) if isinstance(error, Error)
                  else "xml_companion_io_or_structure_error"}
        partial = getattr(error, "partial_directory", None)
        if partial is not None:
            result.update(partial_retained=True, partial_directory=partial)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
