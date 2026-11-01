"""Breakfast Actions video listing, label segments, and person/camera splits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Standard Breakfast person splits (docs/data_provenance.md).
SPLITS: dict[str, range] = {
    "s1": range(3, 16),
    "s2": range(16, 29),
    "s3": range(29, 42),
    "s4": range(42, 55),
}
TEST_SPLIT = "s1"
HELDOUT_CAMERA = "webcam02"  # T2: a camera ID never used in training (docs/label_mapping.md)

Split = Literal["train", "T1", "T2"]

_NAME = re.compile(r"^(P\d+)_(\w+?)(?:_ch(\d))?\.avi$")


@dataclass(frozen=True)
class Video:
    path: Path
    person: str
    camera: str
    recipe: str

    @property
    def key(self) -> str:
        return f"{self.person}_{self.camera}_{self.recipe}"

    def annotation_path(self, annotation_root: Path) -> Path:
        """Official coarse annotation file; the .avi.labels files packed with the videos
        contain extra labels (walk_in, walk_out), so they are not used."""
        name = f"{self.person}_{self.camera}_{self.person}_{self.recipe}.txt"
        return annotation_root / self.recipe / name


@dataclass(frozen=True)
class Segment:
    start: int  # first frame, 1-indexed, inclusive
    end: int  # last frame, inclusive
    label: str


def list_videos(root: Path) -> list[Video]:
    """List usable videos. Stereo channel 1 is skipped: it nearly duplicates channel 0."""
    videos = []
    for path in sorted(root.glob("P*/*/*.avi")):
        m = _NAME.match(path.name)
        if m is None:
            raise ValueError(f"unexpected video name: {path}")
        person, recipe, channel = m.groups()
        folder = path.parent.name
        if folder == "stereo":
            if channel != "0":
                continue
            camera = "stereo01"
        else:
            camera = folder
        videos.append(Video(path=path, person=person, camera=camera, recipe=recipe))
    return videos


def read_segments(labels_path: Path) -> list[Segment]:
    segments = []
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        span, label = line.split()[:2]
        start, end = (int(x) for x in span.split("-"))
        segments.append(Segment(start, end, label))
    return segments


def person_split(person: str) -> str:
    number = int(person[1:])
    for name, persons in SPLITS.items():
        if number in persons:
            return name
    raise ValueError(f"person outside the standard splits: {person}")


def assign_split(video: Video) -> Split | None:
    """train = s2-s4 people on seen cameras; T1 = s1 people on seen cameras;
    T2 = s1 people on the held-out camera. s2-s4 videos from the held-out camera are unused."""
    is_test_person = person_split(video.person) == TEST_SPLIT
    is_heldout_camera = video.camera == HELDOUT_CAMERA
    if is_test_person:
        return "T2" if is_heldout_camera else "T1"
    return None if is_heldout_camera else "train"
