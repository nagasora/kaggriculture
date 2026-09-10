"""公式 Kaggle SDK の構造化レスポンスを読み取る。"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


class KaggleSource:
    """How: ページトークンと agent 情報を落とさず取得する。"""

    def __init__(self, timeout: float = 60, attempts: int = 3) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi()
        self.api.authenticate()
        self.timeout = timeout
        self.attempts = attempts

    @contextmanager
    def connection(self) -> Iterator:
        """How: SDK の HTTP 通信に接続・読み取り期限を設定する。"""
        from requests.adapters import HTTPAdapter
        timeout = self.timeout

        class TimedAdapter(HTTPAdapter):
            def send(self, request, **kwargs):
                """How: SDK が省略するタイムアウトを補う。"""
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = (15, timeout)
                return super().send(request, **kwargs)

        with self.api.build_kaggle_client() as client:
            # Why not global monkeypatch: 他の HTTP 通信を変更しないため。
            client.http_client()._session.mount("https://", TimedAdapter())
            yield client

    def retry(self, operation: Callable[[], T]) -> T:
        """How: 一時的な通信障害・429・5xx にだけ上限付きで再試行する。"""
        import requests
        for attempt in range(self.attempts):
            try:
                return operation()
            except requests.RequestException as exc:
                response = exc.response
                status = response.status_code if response is not None else None
                if status is not None and status != 429 and status < 500:
                    raise
                if attempt + 1 == self.attempts:
                    raise
                delay = min(2 ** attempt, 30)
                if response is not None:
                    try:
                        delay = min(max(delay, float(response.headers.get("Retry-After", delay))), 60)
                    except ValueError:
                        pass
                time.sleep(delay)
        raise RuntimeError("再試行回数は1以上である必要があります")

    def submission_page(self, competition: str, token: str) -> tuple[list[dict], str]:
        """How: 高水準 API で失われる nextPageToken も返す。"""
        from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
        def fetch():
            request = ApiListSubmissionsRequest()
            request.competition_name = competition
            request.page_size = 100
            request.page_token = token
            with self.connection() as client:
                response = client.competitions.competition_api_client.list_submissions(request)
                return [x.to_dict(ignore_defaults=False) for x in response.submissions or []], response.next_page_token
        return self.retry(fetch)

    def episodes(self, submission_id: str) -> list[dict]:
        """How: reward=0、seat=0、相手 submission ID を保持する。"""
        from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionEpisodesRequest
        def fetch():
            request = ApiListSubmissionEpisodesRequest()
            request.submission_id = int(submission_id)
            with self.connection() as client:
                response = client.competitions.competition_api_client.list_submission_episodes(request)
                return [x.to_dict(ignore_defaults=False) for x in response.episodes or []]
        return self.retry(fetch)

    def replay(self, episode_id: str, destination: Path) -> None:
        """How: ダウンロードされた本文を保存し、CLI の案内文を混ぜない。"""
        from kagglesdk.competitions.types.competition_api_service import ApiGetEpisodeReplayRequest
        def fetch():
            request = ApiGetEpisodeReplayRequest()
            request.episode_id = int(episode_id)
            with self.connection() as client:
                with client.competitions.competition_api_client.get_episode_replay(request) as response:
                    with destination.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
        self.retry(fetch)

