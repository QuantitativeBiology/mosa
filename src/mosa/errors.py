"""Error types a user can act on.

Every class here subclasses MosaError, which the CLI boundary catches to print
one actionable line instead of a traceback. Anything not deriving from
MosaError is treated as a bug and keeps its traceback.

Each also subclasses the builtin it replaces, so callers (and tests) that catch
ValueError, FileNotFoundError, RuntimeError, or ImportError keep working.
"""

from __future__ import annotations


class MosaError(Exception):
    """Base for errors a user can fix. Caught at the CLI boundary."""


class ConfigError(MosaError, ValueError):
    """Invalid configuration: bad YAML, unknown key, out-of-range value."""


class DataError(MosaError, ValueError):
    """Data does not satisfy what the config or the model requires."""


class MissingFileError(MosaError, FileNotFoundError):
    """A required input path does not exist."""


class UnsupportedError(MosaError, RuntimeError):
    """The requested operation does not apply to this model or study."""


class MissingDependencyError(MosaError, ImportError):
    """An optional extra is needed for this command."""
