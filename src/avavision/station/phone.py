"""The iPhone as the station camera, over the pharmacy's Wi-Fi (no app to install, no watermark).

Safari gives a web page the camera only when the page comes from a trusted HTTPS address, so the station:

- keeps its own small certificate authority, limited by name constraints to private network addresses, which the
  iPhone trusts once (downloaded from the setup page);
- serves on the local network, separately from its own interface (which stays on 127.0.0.1), only a setup page
  over HTTP (steps and the authority's certificate) and the camera page over HTTPS, whose script streams JPEG
  frames to the station over a WebSocket;
- accepts frames only with the pairing key from the QR code shown in the station's settings, and only from
  private network addresses.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from ..vision.camera import FrameSource

#: The only addresses the station's certificate authority may ever vouch for (RFC 1918, link-local, loopback).
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "127.0.0.0/8")
)
#: Host names are not used; this reserved name keeps every real domain outside the authority's reach.
NO_DOMAINS = "avavision.invalid"
CA_FILE = "avavision-ca.crt"
_PAGES = Path(__file__).with_name("phone_pages")


def new_pairing_key() -> str:
    return secrets.token_urlsafe(18)


# --------------------------------------------------------------------------- certificates


@dataclass
class Authority:
    certificate: x509.Certificate
    key: ec.EllipticCurvePrivateKey

    @property
    def der(self) -> bytes:
        return self.certificate.public_bytes(serialization.Encoding.DER)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the certificate, as iOS shows it under the profile's details."""
        return self.certificate.fingerprint(hashes.SHA256()).hex(" ").upper()


def _write_private(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


def load_or_create_authority(folder: Path) -> Authority:
    """The station's certificate authority, created on first use. Its key never leaves ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = folder / "ca.pem", folder / "ca.key"
    if cert_path.exists() and key_path.exists():
        certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if isinstance(key, ec.EllipticCurvePrivateKey) and certificate.not_valid_after_utc > datetime.now(UTC):
            return Authority(certificate, key)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "AvaVision station (local network only)"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AvaVision"),
        ]
    )
    now = datetime.now(UTC)
    constraints = x509.NameConstraints(
        permitted_subtrees=[x509.IPAddress(n) for n in PRIVATE_NETWORKS] + [x509.DNSName(NO_DOMAINS)],
        excluded_subtrees=None,
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(constraints, critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    _write_private(key_path, _key_pem(key))
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return Authority(certificate, key)


def issue_server_certificate(
    authority: Authority, addresses: list[str], days: int = 397
) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """A TLS certificate for the station's own addresses (397 days, within Apple's limits)."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AvaVision station")]))
        .issuer_name(authority.certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(a)) for a in addresses]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(authority.key.public_key()), critical=False)
        .sign(authority.key, hashes.SHA256())
    )
    return certificate, key


def _covers(certificate: x509.Certificate, authority: Authority, addresses: list[str]) -> bool:
    try:
        certificate.verify_directly_issued_by(authority.certificate)
        names = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except Exception:
        return False
    have = {str(a) for a in names.get_values_for_type(x509.IPAddress)}
    fresh = certificate.not_valid_after_utc > datetime.now(UTC) + timedelta(days=30)
    return fresh and set(addresses) <= have


def ensure_server_certificate(folder: Path, authority: Authority, addresses: list[str]) -> tuple[Path, Path]:
    """Certificate and key files for the HTTPS server, reissued when the addresses change or expiry nears."""
    cert_path, key_path = folder / "server.pem", folder / "server.key"
    if cert_path.exists() and key_path.exists():
        current = x509.load_pem_x509_certificates(cert_path.read_bytes())[0]
        if _covers(current, authority, addresses):
            return cert_path, key_path
    certificate, key = issue_server_certificate(authority, addresses)
    _write_private(key_path, _key_pem(key))
    chain = certificate.public_bytes(serialization.Encoding.PEM) + authority.certificate.public_bytes(
        serialization.Encoding.PEM
    )
    cert_path.write_bytes(chain)
    return cert_path, key_path


def lan_addresses() -> list[str]:
    """This computer's private IPv4 addresses, the one on the default route first."""
    found: list[str] = []
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # a UDP "connect" sends nothing; it only picks the outgoing interface
        found.append(probe.getsockname()[0])
    with contextlib.suppress(OSError):
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    result: list[str] = []
    for text in found:
        address = ipaddress.ip_address(text)
        if text not in result and not address.is_loopback and any(address in n for n in PRIVATE_NETWORKS):
            result.append(text)
    return result


def is_private_client(host: str | None) -> bool:
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_private or address.is_link_local


# --------------------------------------------------------------------------- frames


class PhoneCamera(FrameSource):
    """Frames pushed by the iPhone's camera page. ``read`` waits briefly for the next one."""

    def __init__(self, stale_after: float = 3.0, wait: float = 0.5):
        self.stale_after = stale_after
        self.wait = wait
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._delivered = 0
        self._session = 0
        self._current: int | None = None
        self._closed = False
        self.device: str | None = None
        self.resolution: tuple[int, int] | None = None
        self.last_frame_at = 0.0
        self.frames = 0
        self.fps = 0.0

    def connect(self, device: str) -> int:
        """A newly paired page; it replaces any earlier one."""
        with self._condition:
            self._session += 1
            self._current = self._session
            self.device = device[:120]
            return self._session

    def revoke(self) -> None:
        """Drops the connected page (for example after a new pairing key)."""
        with self._condition:
            self._current = None

    def push(self, jpeg: bytes, session: int) -> bool | None:
        """Decodes one frame. ``None`` when ``session`` is no longer the connected page, ``False`` when the data is
        not a picture."""
        if session != self._current:
            return None
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False
        now = time.time()
        with self._condition:
            if session != self._current:
                return None
            if self.last_frame_at and now - self.last_frame_at < 5:
                self.fps = 0.8 * self.fps + 0.2 / max(now - self.last_frame_at, 1e-3)
            self._frame = image
            self._sequence += 1
            self.frames += 1
            self.last_frame_at = now
            self.resolution = (image.shape[1], image.shape[0])
            self._condition.notify_all()
        return True

    @property
    def connected(self) -> bool:
        return self._current is not None and time.time() - self.last_frame_at < self.stale_after

    def read(self) -> np.ndarray | None:
        with self._condition:
            self._condition.wait_for(lambda: self._sequence != self._delivered or self._closed, timeout=self.wait)
            if self._sequence == self._delivered:
                return None
            self._delivered = self._sequence
            return self._frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def view(self) -> dict:
        return {
            "connected": self.connected,
            "device": self.device,
            "resolution": list(self.resolution) if self.resolution else None,
            "fps": round(self.fps, 1),
            "frames": self.frames,
            "seconds_since_frame": round(time.time() - self.last_frame_at, 1) if self.last_frame_at else None,
        }


# --------------------------------------------------------------------------- the phone's web server


def _page(name: str, **values: str) -> HTMLResponse:
    text = (_PAGES / name).read_text(encoding="utf-8")
    for field, value in values.items():
        text = text.replace("{{" + field + "}}", value)
    return HTMLResponse(text, headers={"Cache-Control": "no-store"})


def _local_network_only(app: FastAPI, allowed) -> None:
    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if not allowed(request.client.host if request.client else None):
            return PlainTextResponse("AvaVision accepts the iPhone only from the local network.", status_code=403)
        return await call_next(request)


def create_setup_app(authority: Authority, allowed=is_private_client) -> FastAPI:
    """Plain HTTP: the steps and the authority's certificate (public, nothing secret)."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    _local_network_only(app, allowed)

    @app.get("/")
    def setup():
        return _page("setup.html", fingerprint=authority.fingerprint)

    @app.get(f"/{CA_FILE}")
    def certificate():
        return Response(
            authority.der,
            media_type="application/x-x509-ca-cert",
            headers={"Content-Disposition": f'attachment; filename="{CA_FILE}"'},
        )

    return app


def create_camera_app(camera: PhoneCamera, link: PhoneLink, allowed=is_private_client) -> FastAPI:
    """HTTPS: the camera page and the WebSocket its frames arrive on."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    _local_network_only(app, allowed)

    @app.get("/")
    @app.get("/camera")
    def page():
        return _page("camera.html")

    @app.websocket("/camera/stream")
    async def stream(socket_: WebSocket):
        if not allowed(socket_.client.host if socket_.client else None):
            await socket_.close(code=1008)
            return
        await socket_.accept()
        try:
            hello = json.loads(await asyncio.wait_for(socket_.receive_text(), timeout=10))
            key = str(hello.get("key", ""))
        except Exception:
            await socket_.close(code=1008)
            return
        if not secrets.compare_digest(key.encode(), link.key.encode()):
            await socket_.close(code=4401)  # the page tells the pharmacist to scan the QR code again
            return
        session = camera.connect(str(hello.get("device") or "iPhone"))
        try:
            while True:
                message = await socket_.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    accepted = await run_in_threadpool(camera.push, message["bytes"], session)
                    if accepted is None:
                        await socket_.close(code=4409 if link.key == key else 4401)
                        break
                    await socket_.send_text("ok" if accepted else "bad")
        except WebSocketDisconnect:
            pass

    return app


class PhoneLink:
    """The two local-network servers for the iPhone camera, with their certificates and pairing key."""

    def __init__(
        self,
        folder: Path,
        camera: PhoneCamera,
        key: str,
        port: int = 8766,
        setup_port: int = 8767,
        addresses: list[str] | None = None,
    ):
        self.camera = camera
        self.key = key
        self.port = port
        self.setup_port = setup_port
        self.addresses = lan_addresses() if addresses is None else addresses
        self.authority = load_or_create_authority(folder)
        self.cert_file, self.key_file = ensure_server_certificate(
            folder, self.authority, [*self.addresses, "127.0.0.1"]
        )
        self.errors: dict[str, str] = {}
        self._servers: list = []
        self._threads: list[threading.Thread] = []

    @property
    def address(self) -> str | None:
        return self.addresses[0] if self.addresses else None

    @property
    def setup_url(self) -> str | None:
        return f"http://{self.address}:{self.setup_port}/" if self.address else None

    @property
    def camera_url(self) -> str | None:
        """Includes the pairing key: shown only as a QR code on the station's own screen."""
        return f"https://{self.address}:{self.port}/camera#k={self.key}" if self.address else None

    def pair_again(self) -> None:
        self.key = new_pairing_key()
        self.camera.revoke()

    def start(self) -> None:
        import uvicorn

        apps = [
            ("camera", create_camera_app(self.camera, self), self.port, True),
            ("setup", create_setup_app(self.authority), self.setup_port, False),
        ]
        for name, app, port, tls in apps:
            tls_files = {"ssl_certfile": str(self.cert_file), "ssl_keyfile": str(self.key_file)} if tls else {}
            config = uvicorn.Config(
                app, host="0.0.0.0", port=port, log_level="warning", ws="wsproto", ws_max_size=32 << 20, **tls_files
            )
            server = uvicorn.Server(config)
            thread = threading.Thread(
                target=self._run, args=(name, server), name=f"avavision-phone-{name}", daemon=True
            )
            self._servers.append(server)
            self._threads.append(thread)
            thread.start()

    def _run(self, name: str, server) -> None:
        try:
            server.run()
        except BaseException as error:  # uvicorn exits (SystemExit) when the port is taken
            self.errors[name] = f"could not start on port {server.config.port}: {type(error).__name__} {error}"
        if not server.started and name not in self.errors:
            self.errors[name] = f"could not start on port {server.config.port} (is it in use?)"

    def wait_started(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(s.started for s in self._servers):
                return True
            if self.errors:
                return False
            time.sleep(0.02)
        return False

    def stop(self) -> None:
        for server in self._servers:
            server.should_exit = True
        for thread in self._threads:
            thread.join(timeout=3)
        self.camera.close()

    def view(self) -> dict:
        problem = None
        if not self.addresses:
            problem = "this computer has no local network address: connect it to the same Wi-Fi as the iPhone"
        elif self.errors:
            problem = "; ".join(self.errors.values())
        return {
            **self.camera.view(),
            "address": self.address,
            "setup_url": self.setup_url,
            "ports": {"camera": self.port, "setup": self.setup_port},
            "fingerprint": self.authority.fingerprint,
            "problem": problem,
        }
