# Copyright (c) m8mble 2022.
# SPDX-License-Identifier: BSL-1.0

import coloredlogs
import logging
import sys

_lvls = {
    "critical": {"level": logging.CRITICAL, "color": "red", "bold": True},
    "error": {"level": logging.ERROR, "color": "red"},
    "warning": {"level": logging.WARNING, "color": "yellow"},
    "notice": {"level": logging.INFO + 5, "color": "green"},
    "info": {"level": logging.INFO, "color": "white"},
    "verbose": {"level": logging.DEBUG + 5, "color": "white", "faint": True},
    "debug": {"level": logging.DEBUG, "color": "blue"},
    "trace": {"level": logging.DEBUG - 2, "color": "blue", "faint": True},
    "dump": {"level": logging.DEBUG - 6, "color": "blue", "faint": True},
}


def _level(name: str) -> int:
    """Returns integer level for string name; unknown levels are handled as info."""
    return _lvls.get(name, _lvls["info"])["level"]


def setup(verbosity: int = 0):
    """Configure logging with a user-selected verbosity (default: 0)."""
    selectable_levels = sorted(
        [spec["level"] for spec in _lvls.values() if spec["level"] <= logging.INFO], reverse=True
    )
    level = selectable_levels[min(verbosity, len(selectable_levels) - 1)]
    if sys.stdout.isatty():
        coloredlogs.install(level=level, fmt="{message}", style="{", level_styles=_lvls)
    else:
        logging.basicConfig(level=level, format="{message}", style="{")

    for lvl, spec in _lvls.items():
        logging.addLevelName(spec["level"], lvl.upper())


def log(*args, level: str, **kwargs):
    """Log message on level."""
    return logging.log(_level(level), *args, **kwargs)


def exception(*args, **kwargs):
    """Log an active exception; should only be called from within exception handlers."""
    return logging.exception(*args, **kwargs)


def is_active(level: str) -> bool:
    """Returns whether logging on level is visible. Returns False for unknown levels."""
    level = _lvls.get(level, {}).get("level", -1)
    return level >= logging.root.level


def _level_log(level: str):
    level = _level(level)

    def _log(*args, **kwargs):
        return logging.log(level, *args, **kwargs)

    return _log


# Generate shorthands for every level configured above (e.g. info / warning / etc.
for level in _lvls.keys():
    setattr(sys.modules[__name__], level, _level_log(level))
