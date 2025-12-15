import asyncio
from typing import Iterable, Optional


class DummyContext:
    """Minimal context capturing progress calls for assertions."""

    def __init__(self) -> None:
        self.progress: list[tuple[int | None, int | None, str]] = []

    async def report_progress(self, index: int | None, total: int | None, message: str) -> None:
        self.progress.append((index, total, message))


class FakeProcess:
    """Asyncio-compatible stub for subprocesses used in tests."""

    def __init__(
        self,
        stdout_chunks: Optional[Iterable[bytes]] = None,
        stderr_chunks: Optional[Iterable[bytes]] = None,
        returncode: int = 0,
        wait_delay: float = 0.0,
    ) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()

        for chunk in stdout_chunks or []:
            self.stdout.feed_data(chunk)
        self.stdout.feed_eof()

        for chunk in stderr_chunks or []:
            self.stderr.feed_data(chunk)
        self.stderr.feed_eof()

        self._returncode = returncode
        self._wait_delay = wait_delay

    async def wait(self) -> int:
        if self._wait_delay:
            await asyncio.sleep(self._wait_delay)
        return self._returncode

    def terminate(self) -> None:  # pragma: no cover - invoked during timeout paths
        self._returncode = -1

    def kill(self) -> None:  # pragma: no cover - invoked during timeout paths
        self._returncode = -9
