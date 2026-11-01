"""Label mapping, splits, and the Breakfast adapter. Uses synthetic files only (no real data)."""

from pathlib import Path

import pytest

from kcp.adapters.breakfast import action_events, mapping_report
from kcp.perception.data import HELDOUT_CAMERA, Video, assign_split, list_videos
from kcp.perception.labels import BREAKFAST_LABELS, CLASSES, map_label
from tests.conftest import FIXTURES


def test_all_48_breakfast_labels_map_to_a_class() -> None:
    assert len(BREAKFAST_LABELS) == 48
    assert {map_label(lab) for lab in BREAKFAST_LABELS} == set(CLASSES)


def test_unknown_label_is_an_error() -> None:
    with pytest.raises(KeyError):
        map_label("walk_in")


def video(person: str, camera: str) -> Video:
    return Video(path=Path("x.avi"), person=person, camera=camera, recipe="salat")


def test_splits_keep_people_and_heldout_camera_out_of_training() -> None:
    assert assign_split(video("P03", "cam01")) == "T1"
    assert assign_split(video("P15", HELDOUT_CAMERA)) == "T2"
    assert assign_split(video("P16", "cam01")) == "train"
    assert assign_split(video("P42", "cam01")) == "val"
    assert assign_split(video("P54", HELDOUT_CAMERA)) is None  # unused, never trained on
    assert assign_split(video("P20", HELDOUT_CAMERA)) is None


def test_list_videos_skips_stereo_channel_1(tmp_path: Path) -> None:
    for rel in [
        "P03/cam01/P03_salat.avi",
        "P03/stereo/P03_salat_ch0.avi",
        "P03/stereo/P03_salat_ch1.avi",
    ]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    cams = sorted(v.camera for v in list_videos(tmp_path))
    assert cams == ["cam01", "stereo01"]


def test_adapter_maps_segments_to_camera_labels() -> None:
    path = FIXTURES / "breakfast_format" / "salat" / "P99_cam01_P99_salat.txt"
    events = list(action_events(path))
    assert [e.action for e in events] == ["background", "take_object", "cut", "transfer", "other"]
    assert events[2].t_start_s == pytest.approx(30 / 15)
    assert events[2].source_label == "cut_fruit"


def test_mapping_report_lists_every_class(tmp_path: Path) -> None:
    out = tmp_path / "events.jsonl"
    report = mapping_report(FIXTURES / "breakfast_format", out)
    for c in CLASSES:
        assert c in report
    assert len(out.read_text().splitlines()) == 5


def test_centering_uses_only_past_frames_of_the_same_video() -> None:
    import numpy as np

    from kcp.perception.train import SplitData, centered

    emb = np.array([[1.0], [3.0], [5.0], [10.0], [20.0]], dtype=np.float16)
    data = SplitData(
        emb=emb,
        y=np.zeros(5, dtype=np.int64),
        video_start=np.array([0, 0, 0, 3, 3]),
        video_end=np.array([2, 2, 2, 4, 4]),
        keys=["a", "b"],
        frame=np.arange(5),
    )
    out = centered(data).emb.astype(np.float32)[:, 0]
    # video a: running means 1, 2, 3; video b starts fresh: 10, 15
    assert out.tolist() == [0.0, 1.0, 2.0, 0.0, 5.0]


def test_event_scoring_matches_overlapping_runs() -> None:
    import numpy as np

    from kcp.perception.labels import class_index
    from kcp.perception.planner_metrics import event_score, runs

    cut, other = class_index("cut"), class_index("other")
    t = np.arange(0, 120, 1.0)
    mask = (t >= 10) & (t < 40)  # a 30 s run
    assert runs(mask, t, 1.0) == [(10.0, 40.0)]
    assert runs((t >= 10) & (t < 25), t, 1.0) == []  # 15 s is shorter than the 20 s minimum

    y = np.where(mask, cut, other)
    p = np.where((t >= 15) & (t < 45), 0.9, 0.1)  # shifted by 5 s: IoU 25/35 = 0.71
    frame = (t * 15).astype(int)
    score = event_score(y, p, frame, np.zeros(len(t), dtype=int), width_s=1.0, threshold=0.5)
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)
