"""ClickHouse behind the engine protocol (`engines.base`): a thin wrapper over the client and the
runner's `ChRound`, unchanged behaviour. Loading stays `loader.load_table`, called by the
orchestrator."""

from __future__ import annotations

from typing import Callable

from chbench.harness.client import CHClient
from chbench.harness.runner import ChRound, Sample


class ClickHouseEngine:
    def __init__(self, client: CHClient):
        self.client = client

    def version(self) -> str:
        return self.client.version()

    def start_round(
        self,
        round_no: int | None,
        *,
        sample_no: int = 0,
        timed: bool = True,
        on_sample: Callable[[str, Sample], None] | None = None,
    ) -> ChRound:
        return ChRound(
            self.client, round_no, sample_no=sample_no, timed=timed, on_sample=on_sample
        )

    def close(self) -> None:
        """Nothing to close: the HTTP client holds no process."""
