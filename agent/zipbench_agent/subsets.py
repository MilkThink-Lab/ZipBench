"""Loading of the packaged compressed-subset definitions."""

import json
from dataclasses import dataclass
from importlib import resources

SIZES = ("small", "tiny")


@dataclass(frozen=True)
class Anchor:
    question_id: str
    weight: float


@dataclass(frozen=True)
class Subset:
    dataset: str
    size: str
    anchors: tuple
    manifest: dict

    @property
    def question_ids(self):
        return [a.question_id for a in self.anchors]

    @property
    def n_total_questions(self):
        return self.manifest["n_total_questions"]


def _data_root():
    return resources.files(__package__) / "subsets_data"


def available_datasets():
    return sorted(
        entry.name for entry in _data_root().iterdir()
        if (entry / "manifest.json").is_file())


def load_manifest(dataset):
    path = _data_root() / dataset / "manifest.json"
    if not path.is_file():
        raise KeyError(
            f"unknown dataset {dataset!r}; available: {', '.join(available_datasets())}")
    return json.loads(path.read_text())


def load_subset(dataset, size):
    if size not in SIZES:
        raise KeyError(f"unknown subset size {size!r}; expected one of {SIZES}")
    manifest = load_manifest(dataset)
    jsonl = (_data_root() / dataset / manifest["subsets"][size]["file"]).read_text()
    anchors = tuple(
        Anchor(row["question_id"], row["weight"])
        for row in (json.loads(line) for line in jsonl.splitlines() if line.strip()))
    return Subset(dataset=dataset, size=size, anchors=anchors, manifest=manifest)
