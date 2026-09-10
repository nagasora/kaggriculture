"""What: 差分同期・ページング・破損修復・失敗時の保存を保証する。"""
from pathlib import Path
import json

import pytest
from kaggriculture_sync.storage import read_json, write_json, read_csv, validate_replay
from kaggriculture_sync.sync import all_submissions, match_row, sync
from kaggriculture_sync.analysis import experiment_id, number, outcome, replay_metrics, analyze


def replay(eid=101):
    """What: day24のリードがday27に消える二人対戦。"""
    frames = []
    for day, money in [(0, [3000, 3000]), (24, [100, 200]), (27, [400, 300]), (29, [600, 400])]:
        frames.append([{"status": "DONE" if day == 29 else "ACTIVE", "reward": money[seat] if day == 29 else 0,
                        "observation": {"day": day, "hour": 0, "player": seat,
                                        "farms": [{"money": m} for m in money]},
                        "action": {"farmer": ["PASS"]}} for seat in (0, 1)])
    return {"steps": frames, "info": {"EpisodeId": eid}}


def episode(eid=101):
    """What: 配列順を反転し、seat=0とreward=0を含める。"""
    return {"id": eid, "state": "COMPLETED", "type": "EPISODE_TYPE_PUBLIC", "agents": [
        {"submissionId": 2, "index": 1, "reward": 400, "teamName": "B"},
        {"submissionId": 1, "index": 0, "reward": 600, "teamName": "A"}]}


class FakeSource:
    """What: 2ページと同一episodeを持つ提出を再現する。"""
    def __init__(self):
        self.downloaded = []
        self.fail = False
        self.episodes_empty = False
        self.pages = {"": ([{"ref": 1, "fileName": "kaggriculture_e001_submission.tar.gz"}], "next"),
                      "next": ([{"ref": 2, "fileName": "02_E003A_submission.tar.gz"}], "")}

    def submission_page(self, competition, token):
        return self.pages[token]

    def episodes(self, sid):
        return [] if self.episodes_empty else [episode()]

    def replay(self, eid, destination):
        self.downloaded.append(eid)
        if self.fail:
            destination.write_text('{"steps":')
            raise OSError("interrupted")
        write_json(destination, replay(int(eid)))


def test_pagination_keeps_both_submissions():
    assert [r["ref"] for r in all_submissions(FakeSource(), "x")] == [1, 2]


def test_repeated_token_is_error():
    source = FakeSource()
    source.pages["next"] = ([{"ref": 2}], "next")
    with pytest.raises(ValueError):
        all_submissions(source, "x")


def test_shared_replay_downloaded_once_and_rerun_downloads_none(tmp_path):
    source = FakeSource()
    first = sync(source, tmp_path, progress=lambda _: None)
    second = sync(source, tmp_path, progress=lambda _: None)
    assert first["matches"] == 2
    assert first["unique_episodes"] == 1
    assert source.downloaded == ["101"]
    assert second["new_replays"] == 0 and second["status"] == "complete"


def test_old_cli_text_is_repaired(tmp_path):
    target = tmp_path / "replays/101.json"
    target.parent.mkdir()
    target.write_text("Replay downloaded to: episode-101-replay.json\n")
    sync(FakeSource(), tmp_path, progress=lambda _: None)
    assert validate_replay(target)["info"]["EpisodeId"] == 101


def test_failed_download_is_not_cached_and_retry_succeeds(tmp_path):
    source = FakeSource()
    source.fail = True
    state = sync(source, tmp_path, progress=lambda _: None)
    assert state["status"] == "partial"
    assert not (tmp_path / "replays/101.json").exists()
    assert not list(tmp_path.rglob("*.partial"))
    assert len(read_csv(tmp_path / "match_index.csv")) == 2
    source.fail = False
    assert sync(source, tmp_path, progress=lambda _: None)["new_replays"] == 1


def test_disappeared_episode_remains_in_archive(tmp_path):
    source = FakeSource()
    sync(source, tmp_path, progress=lambda _: None)
    source.episodes_empty = True
    state = sync(source, tmp_path, progress=lambda _: None)
    assert state["matches"] == 2 and state["cached_replays"] == 1


def test_seat_zero_is_mapped_by_submission_not_array_order():
    row = match_row("1", episode())
    assert row["seat"] == 0 and row["opponent_submission_id"] == 2


def test_unknown_rewards_are_not_zero_or_draws():
    assert outcome(None, 0, True) == "UNKNOWN"
    assert outcome(0, 0, True) == "DRAW"
    assert outcome(1, 0, False) == "UNKNOWN"
    assert number("nan") is None


@pytest.mark.parametrize("filename,expected", [("kaggriculture_e001_submission.tar.gz", "E001"),
    ("02_E003A_submission.tar.gz", "E003A"), ("03_E003B_primary_submission.tar.gz", "E003B")])
def test_real_filename_patterns_keep_variants(filename, expected):
    assert experiment_id({"ref": 1, "fileName": filename}, {}) == expected


def test_reversal_uses_money_and_correct_seat():
    metrics, curve, daily = replay_metrics(replay(), 1)
    assert metrics["coin_margin"] == -200
    assert metrics["lead_day24_then_loss"] is True
    assert metrics["lead_day27_then_loss"] is False
    assert metrics["decisive_reversal_step"] == 2
    assert len(curve) == 4 and len(daily) == 4


def test_end_to_end_analysis_and_reward_disagreement(tmp_path):
    sync(FakeSource(), tmp_path, progress=lambda _: None)
    result = analyze(tmp_path)
    assert result["audit"]["valid_replays"] == 2
    assert result["audit"]["unknown_outcomes"] == 0
    summary = {r["experiment_id"]: r for r in result["experiments"]}
    assert summary["E001"]["WIN"] == 1
    assert summary["E003A"]["reversal_rate_among_leads_day24"] == 1
    broken = replay()
    broken["steps"][-1][0]["reward"] = 999
    write_json(tmp_path / "replays/101.json", broken)
    result = analyze(tmp_path)
    assert result["audit"]["unknown_outcomes"] == 2
    assert len(result["audit"]["errors"]) == 2


def test_sdk_serialization_preserves_absent_reward_and_zero_seat():
    from kagglesdk.competitions.types.competition_api_service import ApiEpisodeAgent
    agent = ApiEpisodeAgent()
    row = agent.to_dict(ignore_defaults=False)
    assert row["index"] == 0
    assert row["reward"] is None
    agent.reward = 0.0
    assert agent.to_dict(ignore_defaults=False)["reward"] == 0.0


def test_submission_failure_preserves_previous_snapshot(tmp_path):
    source = FakeSource()
    sync(source, tmp_path, progress=lambda _: None)
    original = (tmp_path / "submissions.json").read_bytes()
    def fail(*args):
        raise ConnectionError("temporary")
    source.submission_page = fail
    with pytest.raises(ConnectionError):
        sync(source, tmp_path, progress=lambda _: None)
    assert (tmp_path / "submissions.json").read_bytes() == original
    assert read_json(tmp_path / "sync_state.json")["status"] == "failed"


def test_wrong_episode_id_is_repaired(tmp_path):
    target = tmp_path / 'replays/101.json'
    write_json(target, replay(999))
    source = FakeSource()
    sync(source, tmp_path, progress=lambda _: None)
    assert source.downloaded == ['101']
    assert validate_replay(target)['info']['EpisodeId'] == 101


def test_validation_self_play_is_not_counted_in_public_win_rate(tmp_path):
    source = FakeSource()
    def validation(sid):
        ep = episode()
        ep['type'] = 'EPISODE_TYPE_VALIDATION'
        for agent in ep['agents']:
            agent['submissionId'] = int(sid)
        return [ep]
    source.episodes = validation
    sync(source, tmp_path, progress=lambda _: None)
    result = analyze(tmp_path)
    assert result['experiments'] == []
    assert result['audit']['excluded_from_competitive_summary'] == 2
    assert result['audit']['unknown_competitive_outcomes'] == 0


def test_archive_roundtrip_excludes_credentials(tmp_path):
    from kaggriculture_sync.archive import pack, restore
    root = tmp_path / 'data'
    write_json(root / 'submissions.json', [{'ref': 1}])
    write_json(root / 'kaggle.json', {'key': 'synthetic-secret'})
    path = tmp_path / 'data.zip'
    assert pack(root, path)['files'] == 1
    out = tmp_path / 'restored'
    assert restore(path, out) == 1
    assert read_json(out / 'submissions.json') == [{'ref': 1}]
    assert not (out / 'kaggle.json').exists()


def test_archive_rejects_path_traversal_before_writing(tmp_path):
    import zipfile
    from kaggriculture_sync.archive import restore
    path = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('submissions.json', '[]')
        archive.writestr('../outside.json', '{}')
    with pytest.raises(ValueError):
        restore(path, tmp_path / 'out')
    assert not (tmp_path / 'out/submissions.json').exists()
