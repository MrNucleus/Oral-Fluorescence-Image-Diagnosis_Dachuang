#!/usr/bin/env python3
"""Build/verify a standalone, physical whole-image training handoff.

Unlike compact historical snapshots, this package contains every JPEG and one
same-ID XML as real regular files. It has no restore step or source-path runtime
dependency. Classification uses frozen image labels; XML is historical evidence.
Existing packages and reports are never overwritten. Failed builds retain a
private .partial staging directory. build does not train, publish pointers or USB.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import os
import stat
import tempfile
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "_complete_handoff_xml", Path(__file__).with_name("prepare_current_xml_companion.py"))
_xml = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_xml)
_safe = _xml._safe
Error = _xml.Error
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "data/processed/01_当前工作包/口腔整图分类_完整交接包_2026-10-21_v7"
SUPPORT = "data/processed/02_私密核对资料/03_历史与过程文件/完整交接包修复_2026-10-21_v1"
PUBLICATION = "data/processed/02_私密核对资料/01_日常核对/部位交接发布清单.json"
SITE_DIR = "03_说明与验收/部位分层与逐图对应"
PAIR_MANIFEST = "03_说明与验收/图片与XML对应.json"
BUILD_MANIFEST = "03_说明与验收/完整包来源与验收.json"
RECOVERY_MARKERS = ("图片恢复入口", "图片对应与恢复")
CODE_FILES = (
    "AGENTS.md", "pyproject.toml",
    "src/oral_fluorescence_detection/__init__.py",
    "src/oral_fluorescence_detection/device.py",
    "src/oral_fluorescence_detection/online_augmentation.py",
    "src/oral_fluorescence_detection/image_classification.py",
    "scripts/train_image_classifier.py", "scripts/check_online_augmentation.py",
    "scripts/evaluate_image_classifier.py", "scripts/prepare_complete_training_handoff.py",
    "scripts/prepare_current_xml_companion.py", "scripts/restore_dataset_version.py",
    "configs/train_image_classifier.yaml", "configs/online_augmentation.yaml",
    "tests/test_online_augmentation.py", "tests/test_whole_image_dataset.py",
    "docs/在线增强代码使用说明.md", "docs/训练与部位分层评价规则.md",
    "docs/AI协作与交接规则.md", "docs/白光与自体荧光在线增强文献建议_2026-10-18.md",
)


def require(condition, code):
    _xml.require(condition, code)


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def tree(root):
    """Strict regular-file inventory, omitting only Finder system metadata."""
    result = {}
    keys = set()
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [d for d in directories if d != ".DS_Store" and not d.startswith("._")]
        files = [f for f in files if f != ".DS_Store" and not f.startswith("._")]
        for name in directories:
            require(stat.S_ISDIR((Path(current) / name).lstat().st_mode), "nonregular_package_directory")
        for name in files:
            path = Path(current) / name
            require(stat.S_ISREG(path.lstat().st_mode), "nonregular_package_file")
            relative = _safe._relative(path.relative_to(root).as_posix())
            key = _safe._path_key(relative)
            require(key not in keys, "package_filename_collision")
            keys.add(key)
            result[relative] = path
    return result


def checksums(root, *, allow_missing_images=False):
    """Verify immutable inventories; compact-source mode only skips absent JPEGs."""
    entries = {}
    inventory = tree(root)
    for line in _xml.read_bytes(_safe._safe_file(root, "CHECKSUMS.sha256")).decode().splitlines():
        sha, relative = line.split(maxsplit=1)
        relative = _safe._relative(relative)
        require(_safe.SHA256.fullmatch(sha) and relative not in entries, "invalid_checksum_manifest")
        entries[relative] = sha
        if allow_missing_images and Path(relative).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        require(relative in inventory, "missing_package_file")
        require(inventory[relative].stat().st_size > 0, "zero_length_package_file")
        require(_xml.digest(inventory[relative]) == sha, "package_checksum_mismatch")
    require(bool(entries), "empty_package_checksum_manifest")
    if not allow_missing_images:
        require(set(inventory) == set(entries) | {"CHECKSUMS.sha256"}, "package_inventory_mismatch")
    return entries


def write_file(root, relative, payload):
    relative = _safe._relative(relative)
    path = root / relative
    _xml.mkdir_private(path.parent, root)
    _xml.write_bytes(path, payload)


def copy_file(source, destination, relative, expected=None):
    before = _xml.signature(source)
    target = destination / _safe._relative(relative)
    _xml.mkdir_private(target.parent, destination)
    record = {"sha256": expected or _xml.digest(source), "bytes": before[2]}
    with _safe._regular_stream(source) as stream, _safe._destination(destination, relative) as output:
        _safe._stream_verify(stream, record, output)
    require(_xml.signature(source) == before, "source_changed_during_copy")
    require(target.stat().st_ino != source.stat().st_ino
            or target.stat().st_dev != source.stat().st_dev, "copy_is_not_independent")
    return record["sha256"]


def checksum_write(root):
    inventory = tree(root)
    require("CHECKSUMS.sha256" not in inventory, "checksum_already_exists")
    payload = "".join(f"{_xml.digest(path)}  {relative}\n"
                      for relative, path in sorted(inventory.items()))
    write_file(root, "CHECKSUMS.sha256", payload.encode())


def _site_validate(package, dataset_records, expected_hashes):
    for name, sha in expected_hashes.items():
        path = _safe._safe_file(package, f"{SITE_DIR}/{_safe._relative(name)}")
        require(_xml.digest(path) == sha, "published_site_file_changed")
    site = _safe._read_json(_safe._safe_file(package, f"{SITE_DIR}/image_sites.json"))
    require(site.get("dataset_json_sha256") == _xml.EXPECTED_DATASET_SHA, "site_dataset_mismatch")
    records = site.get("records")
    require(isinstance(records, list) and len(records) == len(dataset_records), "site_membership_mismatch")
    seen = set()
    counts = Counter()
    for row in records:
        image_id = row.get("image_id")
        require(image_id in dataset_records and image_id not in seen, "site_membership_mismatch")
        seen.add(image_id)
        original = dataset_records[image_id]
        for key in ("split", "modality", "label", "group_id"):
            require(row.get(key) == original[key], "site_assignment_changed")
        require(row.get("image_relpath") == "01_图片/" + original["image_relpath"],
                "site_image_path_changed")
        require(row.get("site_group_main") in site["groups"], "unknown_published_site_group")
        counts[row["site_group_main"]] += 1
    require(seen == set(dataset_records), "site_membership_mismatch")
    with io.StringIO(_xml.read_bytes(_safe._safe_file(package, f"{SITE_DIR}/image_sites.csv"))
                     .decode("utf-8-sig")) as stream:
        csv_records = list(csv.DictReader(stream))
    require(len(csv_records) == len(records), "site_csv_membership_mismatch")
    csv_by_id = {row.get("image_id"): row for row in csv_records}
    require(len(csv_by_id) == len(records) and set(csv_by_id) == seen, "site_csv_membership_mismatch")
    for row in records:
        other = csv_by_id[row["image_id"]]
        for key in ("split", "modality", "group_id", "site_group_main"):
            require(other.get(key) == str(row[key]), "site_csv_assignment_changed")
        require(other.get("label") == str(row["label"]), "site_csv_assignment_changed")
    return dict(counts)


def verify_package(package):
    """Validate solely against contents of this standalone package, without sources."""
    package = _safe._root(package)
    entries = checksums(package)
    inventory = tree(package)
    require(not any(any(mark in Path(p).name for mark in RECOVERY_MARKERS) for p in inventory),
            "recovery_entry_in_complete_handoff")
    for path in inventory.values():
        require(path.stat().st_size > 0, "zero_length_package_file")
    source_manifest = _safe._read_json(_safe._safe_file(package, BUILD_MANIFEST))
    require(source_manifest.get("schema_version") == 1
            and source_manifest.get("standalone_images_and_xml") is True
            and source_manifest.get("dataset_sha256") == _xml.EXPECTED_DATASET_SHA,
            "invalid_complete_package_manifest")
    dataset = package / "01_图片"
    nested = checksums(dataset)
    records = _xml.dataset_records(dataset)
    by_id = {r["image_id"]: r for r in records}
    pair = _safe._read_json(_safe._safe_file(package, PAIR_MANIFEST))
    require(pair.get("dataset_sha256") == _xml.EXPECTED_DATASET_SHA
            and isinstance(pair.get("records"), list), "invalid_pair_manifest")
    paired = pair["records"]
    require(len(paired) == len(records), "pair_membership_mismatch")
    expected_images = {"01_图片/" + r["image_relpath"] for r in records}
    expected_xml = set()
    seen = set()
    xml_bytes = boxes = 0
    for row in paired:
        image_id = row.get("image_id")
        require(image_id in by_id and image_id not in seen, "pair_membership_mismatch")
        seen.add(image_id)
        original = by_id[image_id]
        image_relative = "01_图片/" + original["image_relpath"]
        xml_relative = f"01_图片/{original['split']}/{original['modality']}/annotations/{image_id}.xml"
        require(row.get("image_relpath") == image_relative
                and row.get("xml_relpath") == xml_relative, "pair_path_mismatch")
        for key in ("split", "modality", "label", "group_id", "width", "height"):
            require(row.get(key) == original[key], "pair_assignment_changed")
        expected_xml.add(xml_relative)
        image = _safe._safe_file(package, image_relative)
        require(row.get("image_sha256") == original["sha256"] == _xml.digest(image),
                "image_sha256_mismatch")
        require(_xml.jpeg_dimensions(image) == (original["width"], original["height"]),
                "image_dimension_mismatch")
        payload = _xml.read_bytes(_safe._safe_file(package, xml_relative))
        require(sha_bytes(payload) == row.get("xml_sha256"), "xml_sha256_mismatch")
        count = _xml.xml_info(payload, original["width"], original["height"], original["label"])
        require(count == row.get("xml_box_count"), "xml_box_count_mismatch")
        boxes += count
        xml_bytes += len(payload)
    actual_images = {p for p in inventory if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png"}}
    actual_xml = {p for p in inventory if Path(p).suffix.lower() == ".xml"}
    require(len(records) == 1117 and actual_images == expected_images,
            "complete_handoff_requires_1117_images")
    require(len(expected_xml) == 1117 and actual_xml == expected_xml,
            "complete_handoff_requires_1117_xml")
    require(boxes == 1219, "unexpected_complete_historical_boxes")
    site_hashes = source_manifest.get("site_files_sha256")
    require(isinstance(site_hashes, dict) and len(site_hashes) == 4, "invalid_site_file_manifest")
    site_counts = _site_validate(package, by_id, site_hashes)
    for relative, sha in source_manifest["code_files_sha256"].items():
        require(_xml.digest(_safe._safe_file(package, f"02_训练代码/{relative}")) == sha,
                "code_snapshot_changed")
    rules_sha = _xml.digest(_safe._safe_file(package, "AGENTS.md"))
    require(rules_sha == source_manifest["code_files_sha256"]["AGENTS.md"], "rules_copy_mismatch")
    for name in ("训练与部位分层评价规则.md", "AI协作与交接规则.md"):
        require(_xml.digest(_safe._safe_file(package, f"03_说明与验收/{name}"))
                == source_manifest["code_files_sha256"][f"docs/{name}"], "protocol_copy_mismatch")
    return {"status": "passed", "images": 1117, "xml_files": 1117,
            "historical_xml_objects": boxes, "xml_bytes": xml_bytes,
            "files": len(inventory), "checksums_verified": len(entries),
            "image_package_checksums_verified": len(nested),
            "bytes": sum(p.stat().st_size for p in inventory.values()),
            "dataset_sha256": _xml.EXPECTED_DATASET_SHA,
            "site_files_sha256": site_hashes, "site_group_image_counts": site_counts,
            "source_publication_sha256": source_manifest["source_publication_sha256"],
            "root_checksums_sha256": _xml.digest(package / "CHECKSUMS.sha256"),
            "standalone_images_and_xml": True, "symbolic_links": 0,
            "recovery_required": False, "zero_length_files": 0,
            "training_executed": False, "clinical_review_completed": False}


def build_package(project, package, support):
    require(not os.path.lexists(package), "package_already_exists")
    require(package.parent.is_dir(), "package_parent_missing")
    # The publication entry is itself regular; private compatibility links are not sources.
    pointer = _safe._safe_file(project, PUBLICATION)
    publication_bytes = _xml.read_bytes(pointer)
    publication = json.loads(publication_bytes)
    require(publication.get("dataset_json_sha256") == _xml.EXPECTED_DATASET_SHA,
            "publication_dataset_mismatch")
    published = project / _safe._relative(publication.get("local_handoff_directory"))
    published = _safe._root(published)
    previous_entries = checksums(published, allow_missing_images=True)
    dataset = _safe._root(project / _xml.DATASET)
    companion = _safe._root(project / _xml.COMPANION)
    require(all(_safe._outside(package, p) for p in (published, dataset, companion)),
            "output_inside_source_package")
    _xml.verify_companion(dataset, companion)
    checksums(dataset)
    current_records = _xml.dataset_records(dataset)
    xml_manifest = _safe._read_json(_safe._safe_file(companion, "manifest.json"))
    xml_records = {r["image_id"]: r for r in xml_manifest["records"]}
    source_files = tree(published)
    source_protection = {}
    for relative, path in source_files.items():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        if relative not in previous_entries and relative != "CHECKSUMS.sha256":
            require(any(mark in path.name for mark in RECOVERY_MARKERS),
                    "unrecognized_source_handoff_metadata")
        source_protection[path] = (_xml.signature(path), _xml.digest(path))
    current_files = tree(dataset)
    for path in current_files.values():
        source_protection[path] = (_xml.signature(path), _xml.digest(path))
    for path in tree(companion).values():
        source_protection[path] = (_xml.signature(path), _xml.digest(path))
    code_hashes = {}
    for relative in CODE_FILES:
        path = _safe._safe_file(project, relative)
        code_hashes[relative] = _xml.digest(path)
        source_protection[path] = (_xml.signature(path), code_hashes[relative])
    stage = Path(tempfile.mkdtemp(prefix=f".{package.name}.", suffix=".partial", dir=package.parent))
    stage.chmod(0o700)
    try:
        for relative, path in sorted(source_files.items()):
            if (relative == "CHECKSUMS.sha256" or relative.startswith(("01_图片/", "02_训练代码/"))
                    or any(mark in path.name for mark in RECOVERY_MARKERS)
                    or relative in {"AGENTS.md", "先看这里.md",
                                    "03_说明与验收/AI协作与交接规则.md",
                                    "03_说明与验收/训练与部位分层评价规则.md"}):
                continue
            copy_file(path, stage, relative, source_protection[path][1])
        for relative, path in sorted(current_files.items()):
            if relative == "CHECKSUMS.sha256":
                continue
            copy_file(path, stage, "01_图片/" + relative, source_protection[path][1])
        pairs = []
        for record in current_records:
            image_id = record["image_id"]
            xml = xml_records[image_id]
            target_relative = "01_图片/" + xml["xml_relpath"]
            copy_file(_safe._safe_file(companion, xml["xml_relpath"]), stage,
                      target_relative, xml["xml_sha256"])
            pairs.append({"image_id": image_id, "group_id": record["group_id"],
                          "split": record["split"], "modality": record["modality"],
                          "label": record["label"], "width": record["width"], "height": record["height"],
                          "image_relpath": "01_图片/" + record["image_relpath"],
                          "image_sha256": record["sha256"], "xml_relpath": target_relative,
                          "xml_sha256": xml["xml_sha256"], "xml_box_count": xml["xml_box_count"]})
        for relative, sha in code_hashes.items():
            copy_file(_safe._safe_file(project, relative), stage, "02_训练代码/" + relative, sha)
        copy_file(_safe._safe_file(project, "AGENTS.md"), stage, "AGENTS.md", code_hashes["AGENTS.md"])
        for name in ("训练与部位分层评价规则.md", "AI协作与交接规则.md"):
            copy_file(_safe._safe_file(project, f"docs/{name}"), stage,
                      f"03_说明与验收/{name}", code_hashes[f"docs/{name}"])
        write_file(stage, "03_说明与验收/原始v6图片包_CHECKSUMS.sha256",
                   _xml.read_bytes(dataset / "CHECKSUMS.sha256"))
        write_file(stage, PAIR_MANIFEST, (json.dumps({
            "schema_version": 1, "dataset_sha256": _xml.EXPECTED_DATASET_SHA,
            "xml_training_target": False, "records": pairs,
        }, ensure_ascii=False, indent=2) + "\n").encode())
        source_manifest = {
            "schema_version": 1, "version": 7,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "standalone_images_and_xml": True, "dataset_sha256": _xml.EXPECTED_DATASET_SHA,
            "source_publication_sha256": sha_bytes(publication_bytes),
            "source_handoff_version": publication["handoff_version"],
            "source_handoff_checksums_sha256": _xml.digest(published / "CHECKSUMS.sha256"),
            "source_xml_companion_manifest_sha256": _xml.digest(companion / "manifest.json"),
            "site_files_sha256": publication["files_sha256"], "code_files_sha256": code_hashes,
            "dataset_membership_labels_splits_unchanged": True,
            "source_records_modified": False, "training_executed": False,
        }
        write_file(stage, BUILD_MANIFEST, (json.dumps(source_manifest, ensure_ascii=False,
                                                    indent=2) + "\n").encode())
        readme = """# 完整整图分类交接包 v7

直接把本文件夹完整复制给训练同学即可：这里实际保存 1,117 张完整 JPG 和 1,117 份逐图同名 XML，图片、代码和部位资料齐全。无需恢复、解压、快捷方式或访问原机数据。

- 01_图片：完整图片位于 train/val/test 各自的 fluorescence/standard/images/类别目录；对应 XML 位于同 split、同模态的 annotations/image_id.xml。整图标签仍是原 annotations/image_labels.json 与 dataset.json。
- 02_训练代码：完整的当前训练、预检、评价代码、配置和共同规则。
- 03_说明与验收：逐图图片/XML对应清单、现行部位四份资料、训练协议及验收来源。

dataset.json、图片像素、image_id、来源组、标签和 split 均保持冻结 v6 原值：train/val/test 为893/112/112，普通白光616、荧光501，正常候选418、病变候选699。部位按发布资料保留，只用于同一最终预测的分层评价，不是疾病分类或三套部位模型。

XML 是完整历史标注，共1,219框，原字节及内部历史path保留；实际找图以03_说明与验收/图片与XML对应.json为准。旧v5筛框派生版本1,200框另存历史。当前训练仍是整张图片分类，不把框作为训练目标，不裁ROI，不推定临床确诊。

先保留收到的原件，再复制为训练工作项目。工作副本中进入02_训练代码：

```bash
python -m pip install -e '.[train,dev]'
python scripts/train_image_classifier.py --data ../01_图片/dataset.json --check-only
python scripts/prepare_complete_training_handoff.py verify --package .. --report /absolute/new_private_verify.json
```

以上是安装和检查；正式训练需在已授权工作任务中运行。完整性验证不会训练或计算测试性能。模型同学先读AGENTS、训练与部位分层评价规则、AI协作与交接规则；用整体train训练和整体val选方案，冻结后同一阈值、同一测试预测报告总体/部位/模态。候选标签不是医生确诊，来源组不是已证明独立患者。

根CHECKSUMS.sha256覆盖整个交接包；01_图片/CHECKSUMS.sha256覆盖新增XML后的完整图片资料目录。原v6校验清单保存在03，仅描述原冻结v6文件，不把新增XML说成原包文件。所有恢复源码仅是XML验证器的标准库辅助依赖，本包使用不需要恢复命令或历史归档。
"""
        write_file(stage, "先看这里.md", readme.encode())
        code_readme = """# 训练代码与完整数据

请保持本目录与../01_图片、../03_说明与验收的相对位置。01_图片实际含1,117张JPG及1,117份同名XML，无需恢复。数据入口../01_图片/dataset.json，部位入口../03_说明与验收/部位分层与逐图对应/image_sites.json。

先保留收到的原件，复制完整目录为工作项目，再在工作副本安装依赖、开发和训练。工作副本的图片、标签、split和交付规则仍冻结；输出写工作项目，不回写收到的原件。具体检查命令见../先看这里.md。

所有AI共同遵守AGENTS.md、docs/AI协作与交接规则.md和docs/训练与部位分层评价规则.md。整图分类，不用XML框作目标或裁剪；整体train训练、整体val选方案，冻结后同一测试预测报告总体与部位结果。当前软件预检不代表临床性能、患者独立性或接收方已经训练。
"""
        write_file(stage, "02_训练代码/README.md", code_readme.encode())
        checksum_write(stage / "01_图片")
        checksum_write(stage)
        result = verify_package(stage)
        for source, (before, expected) in source_protection.items():
            require(_xml.signature(source) == before and _xml.digest(source) == expected,
                    "source_changed_during_build")
        require(_xml.read_bytes(pointer) == publication_bytes, "publication_changed_during_build")
        require(not os.path.lexists(support / "source_publication.json"), "source_evidence_already_exists")
        _xml.write_bytes(support / "source_publication.json", publication_bytes)
        _safe._publish_exclusive(stage, package)
        stage = None
        result.update(operation="build", source_snapshots_unchanged=True,
                      source_publication_saved=True, package_version=7)
        return result
    except BaseException as error:
        if stage is not None and isinstance(error, Exception):
            error.partial_directory = str(stage)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        project = _safe._root(args.project_root)
        package = args.package or project / PACKAGE
        support = project / SUPPORT
        report = args.report or support / f"{args.mode}_validation.json"
        if not report.parent.exists():
            require(report.is_relative_to(project), "external_report_parent_missing")
            _xml.mkdir_private(report.parent, project)
        require(not os.path.lexists(report), "report_already_exists")
        require(_safe._outside(report.resolve(), package.resolve()), "report_inside_frozen_package")
        if args.mode == "build":
            _xml.mkdir_private(support, project)
            result = build_package(project, package, support)
        else:
            result = verify_package(package)
            result.update(operation="verify")
        _xml.write_json(report, result)
    except (Error, OSError, ValueError, TypeError, KeyError) as error:
        result = {"status": "failed", "error": str(error) if isinstance(error, Error)
                  else "complete_handoff_io_or_structure_error"}
        partial = getattr(error, "partial_directory", None)
        if partial:
            result.update(partial_retained=True, partial_directory=partial)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
