"""Kill-switch toggle helper used by the dashboard.

The kill switch is a sentinel file on disk plus an environment variable
(see :func:`risk.manager.is_kill_switch_active`). The dashboard talks to
the file path only — flipping ``KILL_SWITCH=1`` in the *running* process
environment would not survive a restart, so we ignore env writes and
operate exclusively on the configured ``kill_switch_file``.

TRADEOFF: We touch / remove the file rather than using something more
ceremonious (lockfile, atomic rename) because the upstream readers only
care about the file's existence. Concurrency is handled by the
:attr:`dashboard.state.AppState.write_lock` at the route layer.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from risk.manager import is_kill_switch_active

logger = structlog.get_logger(__name__)


class KillSwitchService:
    """Read/enable/disable the kill switch via its on-disk sentinel."""

    def __init__(self, kill_file: Path, env_var: str = "KILL_SWITCH") -> None:
        """Construct a service bound to ``kill_file``.

        Args:
            kill_file: Absolute path to the sentinel file. When present,
                the kill switch is considered active.
            env_var: Environment variable that also enables the switch
                when set to ``1`` / ``true``. The dashboard never writes
                this — it is consulted only to surface a truthful "is
                active" reading.
        """
        self._kill_file = kill_file
        self._env_var = env_var

    @property
    def kill_file(self) -> Path:
        """Path to the sentinel file backing the kill switch."""
        return self._kill_file

    def is_active(self) -> bool:
        """Return ``True`` when the kill switch is currently engaged."""
        return is_kill_switch_active(kill_file=self._kill_file, env_var=self._env_var)

    def enable(self) -> bool:
        """Engage the kill switch by touching the sentinel file.

        Returns:
            ``True`` if the file existed already, ``False`` if newly created.
        """
        existed = self._kill_file.is_file()
        self._kill_file.parent.mkdir(parents=True, exist_ok=True)
        self._kill_file.touch(exist_ok=True)
        logger.warning(
            "dashboard_kill_switch_enabled",
            path=str(self._kill_file),
            existed=existed,
        )
        return existed

    def disable(self) -> bool:
        """Disarm the kill switch by removing the sentinel file.

        Returns:
            ``True`` if the file was removed, ``False`` if it didn't exist.
        """
        if not self._kill_file.is_file():
            logger.info("dashboard_kill_switch_disable_noop", path=str(self._kill_file))
            return False
        self._kill_file.unlink()
        logger.warning("dashboard_kill_switch_disabled", path=str(self._kill_file))
        return True

    def set(self, enabled: bool) -> bool:
        """Set the kill switch to a desired state.

        Args:
            enabled: ``True`` to engage, ``False`` to disarm.

        Returns:
            The new active state (matches ``enabled`` on success).
        """
        if enabled:
            self.enable()
        else:
            self.disable()
        return self.is_active()
