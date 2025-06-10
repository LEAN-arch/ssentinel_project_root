# sentinel_project_root/config/__init__.py
# This file makes the 'config' directory a Python package.
# It exposes the singleton 'settings' instance for easy, clean importing.

from .settings import settings

__all__ = ["settings"]
