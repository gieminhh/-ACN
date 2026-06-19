try:
    from importlib.metadata import version as get_version, PackageNotFoundError

    try:
        __version__ = get_version("rl4co")
    except PackageNotFoundError:
        __version__ = "0.0.0"

except Exception:
    __version__ = "0.0.0"
