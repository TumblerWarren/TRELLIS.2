"""Normalize Hugging Face download settings before importing huggingface_hub.

RunPod or base container images sometimes set HF_HUB_ENABLE_HF_TRANSFER=1,
which makes older huggingface_hub releases require the optional hf_transfer
package even when just fetching a small JSON file. Recent releases ignore
this deprecated setting in favor of the hf-xet backend. Clearing it is safe
across both versions and does not alter authentication, offline mode, or
other HF_* variables.
"""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from typing import Callable


def configure_hf_download_environment(
    environ: MutableMapping[str, str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> bool:
    """Clear legacy hf_transfer flag; return True if it was present.

    MUST be called before importing gradio, transformers or huggingface_hub
    because older releases cache download configuration during import.
    """
    values = os.environ if environ is None else environ
    legacy = values.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if legacy is None:
        return False

    if legacy.strip().lower() in {"1", "on", "true", "yes"}:
        message = (
            "[TRELLIS.2] Removed deprecated HF_HUB_ENABLE_HF_TRANSFER="
            f"{legacy!r}; using the Hugging Face Hub's supported download "
            "backend (hf-xet where available, standard HTTP otherwise)."
        )
        if notify is None:
            print(message, file=sys.stderr, flush=True)
        else:
            notify(message)
    return True


def explain_hub_error(exc: Exception) -> str:
    """Provide human-actionable guidance without masking the real exception."""
    cls = type(exc).__name__
    message = str(exc).lower()
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)

    if "gated" in cls.lower() or "gated repo" in message or "gatedrepo" in message:
        return (
            "The model is gated. Accept its conditions on the model page and "
            "authenticate with a read token from the approved Hugging Face account."
        )
    if status in (401, 403) or "401 client error" in message or "403 client error" in message:
        return (
            "Unauthorized or forbidden. Check model access approval, the "
            "active account (hf auth whoami), and HF_TOKEN permissions."
        )
    if status == 404 or cls in {"RepositoryNotFoundError", "EntryNotFoundError"}:
        return (
            "Repository or requested file not found. Verify the model ID, "
            "filename, and upstream revision."
        )
    if "hf_transfer" in message:
        return (
            "An obsolete HF_HUB_ENABLE_HF_TRANSFER setting is still reaching "
            "huggingface_hub. Unset it before importing Hub libraries."
        )
    if "offline" in message or os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "yes", "true", "on"}:
        return (
            "Offline mode may be enabled or the file is not cached. Check "
            "HF_HUB_OFFLINE and internet access."
        )
    if "no space left" in message or "disk quota" in message:
        return "Insufficient disk space. Free storage or move HF_HOME to a larger volume."
    if any(term in cls.lower() for term in ("timeout", "connection", "proxy", "ssl")):
        return "Network, proxy or TLS failure. Check RunPod internet/DNS/proxy and retry."
    return (
        "Check the error, model access, network connectivity, and model cache; "
        "the preflight did not modify your credentials."
    )
