"""Where the station keeps its Claude API key.

Preferred: the ``ANTHROPIC_API_KEY`` environment variable or an ``ant auth login`` profile (the SDK finds both).
A key entered in the station's settings goes to the operating system's credential store (Windows Credential
Manager via ``keyring``) and, only if no store is available, to the local database.
"""

from __future__ import annotations

import contextlib

from ..storage.database import Database

SERVICE = "AvaVision"
ACCOUNT = "anthropic-api-key"
_SETTING = "anthropic_api_key"


def _keyring():
    try:
        import keyring

        keyring.get_keyring()
        return keyring
    except Exception:  # no package or no usable backend
        return None


def load_api_key(db: Database) -> str | None:
    store = _keyring()
    if store is not None:
        try:
            key = store.get_password(SERVICE, ACCOUNT)
            if key:
                return key
        except Exception:
            pass
    return db.setting(_SETTING)


def save_api_key(db: Database, key: str | None) -> str:
    """Stores (or with ``None`` removes) the key; returns where it was stored."""
    store = _keyring()
    if store is not None:
        try:
            if key:
                store.set_password(SERVICE, ACCOUNT, key)
            else:
                with contextlib.suppress(Exception):
                    store.delete_password(SERVICE, ACCOUNT)
            db.save_setting(_SETTING, None)
            return "keyring"
        except Exception:
            pass
    db.save_setting(_SETTING, key)
    return "database"
