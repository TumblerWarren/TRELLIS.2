"""RunPod/Gradio public-link settings, independent of heavy model imports.

A temporary gradio.live link is convenient for testing. When sharing or binding
to a non-loopback address, generate random Gradio authentication credentials
unless the operator provided a username/password or explicitly opted out.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from collections.abc import Mapping


TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def parse_flag(name: str, value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be true or false (also accepts 1/0, yes/no, on/off), "
        f"not {value!r}."
    )


@dataclass(frozen=True)
class GradioAccess:
    share: bool
    auth: tuple[str, str] | None
    generated_credentials: bool
    server_name: str
    server_port: int


def resolve_gradio_access(environ: Mapping[str, str] | None = None) -> GradioAccess:
    """Resolve launch settings without network requests or installing Gradio.

    Gradio shares remain temporary, and an outbound tunnel may be blocked.
    We do not silently expose an unprotected paid GPU inference service.
    """
    env = os.environ if environ is None else environ
    share = parse_flag("GRADIO_SHARE", env.get("GRADIO_SHARE", "true"))
    bind = env.get("GRADIO_SERVER_NAME", "0.0.0.0").strip()
    port_raw = env.get("GRADIO_SERVER_PORT", "7860").strip()
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError("GRADIO_SERVER_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("GRADIO_SERVER_PORT must be between 1 and 65535.")
    if not bind:
        raise ValueError("GRADIO_SERVER_NAME cannot be empty.")

    user = env.get("GRADIO_AUTH_USER", "").strip()
    password = env.get("GRADIO_AUTH_PASSWORD", "")
    if bool(user) != bool(password):
        raise ValueError(
            "Set BOTH GRADIO_AUTH_USER and GRADIO_AUTH_PASSWORD, or neither."
        )

    allow_unauthenticated = parse_flag(
        "TRELLIS_ALLOW_UNAUTHENTICATED",
        env.get("TRELLIS_ALLOW_UNAUTHENTICATED", "false"),
    )
    generated = False
    # Binding to an external interface is potentially public via RunPod's HTTP
    # proxy even when the Gradio share tunnel is disabled.
    external_bind = bind not in {"127.0.0.1", "localhost", "::1"}
    needs_auth = share or external_bind
    if needs_auth and not user and not allow_unauthenticated:
        user = "admin"
        password = secrets.token_urlsafe(24)
        generated = True

    return GradioAccess(
        share=share,
        auth=(user, password) if user else None,
        generated_credentials=generated,
        server_name=bind,
        server_port=port,
    )


def explain_gradio_access(config: GradioAccess) -> None:
    """Print accurate access instructions without claiming a tunnel exists."""
    print(f"[TRELLIS.2] Gradio listening at http://{config.server_name}:{config.server_port}", flush=True)
    if config.auth:
        if config.generated_credentials:
            print("[TRELLIS.2] Generated one-time login credentials for this session:", flush=True)
            print(f"[TRELLIS.2]   Username: {config.auth[0]}", flush=True)
            print(f"[TRELLIS.2]   Password: {config.auth[1]}", flush=True)
            print("[TRELLIS.2] Keep these console credentials private; they change on restart.", flush=True)
        else:
            print(f"[TRELLIS.2] Gradio login enabled for user {config.auth[0]!r}.", flush=True)
    else:
        print("[TRELLIS.2] WARNING: Gradio authentication is DISABLED by explicit opt-out.", flush=True)

    if config.share:
        print(
            "[TRELLIS.2] Requesting a temporary https://*.gradio.live public URL. "
            "The URL is generated only if Gradio's outbound share tunnel succeeds.",
            flush=True,
        )
        print(
            "[TRELLIS.2] If no public URL appears, check network/firewall/share-server "
            "availability or expose HTTP port 7860 using RunPod's Connect menu. "
            "The RunPod proxy does not require Gradio sharing.",
            flush=True,
        )
    else:
        print(
            "[TRELLIS.2] GRADIO_SHARE=false: no gradio.live link requested. "
            "Use RunPod's exposed HTTP port / proxy or an SSH port-forward.",
            flush=True,
        )
