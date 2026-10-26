class AppError(Exception):
    """Runtime failure (exit code 1)."""

    exit_code = 1


class UsageError(AppError):
    """Bad usage or environment (exit code 2)."""

    exit_code = 2
