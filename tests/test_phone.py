from __future__ import annotations

import ipaddress
import json
import socket
import ssl
import time
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.verification import PolicyBuilder, Store, VerificationError
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from avavision.station import phone as phone_module
from avavision.station.api import create_app
from avavision.station.phone import (
    PhoneCamera,
    create_camera_app,
    create_setup_app,
    ensure_server_certificate,
    is_private_client,
    issue_server_certificate,
    load_or_create_authority,
)
from avavision.station.service import Station
from avavision.storage.database import Database
from avavision.vision.codes import read_codes
from avavision.vision.synthetic import Station as SyntheticStation
from avavision.vision.synthetic import full_scene

ADDRESS = "192.168.50.10"


def jpeg(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def verifier(authority, name):
    return PolicyBuilder().store(Store([authority.certificate])).build_server_verifier(name)


def test_the_station_authority_can_only_vouch_for_private_addresses(tmp_path):
    authority = load_or_create_authority(tmp_path)
    assert load_or_create_authority(tmp_path).fingerprint == authority.fingerprint  # kept, not remade
    cert_path, key_path = ensure_server_certificate(tmp_path, authority, [ADDRESS, "127.0.0.1"])
    leaf = x509.load_pem_x509_certificates(cert_path.read_bytes())[0]
    assert len(verifier(authority, x509.IPAddress(ipaddress.ip_address(ADDRESS))).verify(leaf, [])) == 2

    public, _ = issue_server_certificate(authority, ["8.8.8.8"])
    with pytest.raises(VerificationError):
        verifier(authority, x509.IPAddress(ipaddress.ip_address("8.8.8.8"))).verify(public, [])

    # The same certificate is reused until the computer's address changes.
    assert ensure_server_certificate(tmp_path, authority, [ADDRESS])[0].read_bytes() == cert_path.read_bytes()
    moved = ensure_server_certificate(tmp_path, authority, ["10.0.0.7"])[0]
    names = x509.load_pem_x509_certificates(moved.read_bytes())[0].extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    )
    assert "10.0.0.7" in {str(a) for a in names.value.get_values_for_type(x509.IPAddress)}
    assert all(is_private_client(h) for h in ("192.168.1.4", "10.1.2.3", "172.20.0.9", "::ffff:192.168.0.2"))
    assert not any(is_private_client(h) for h in ("8.8.8.8", "testclient", None, "1.1.1.1"))


def test_phone_camera_keeps_only_the_connected_page():
    camera = PhoneCamera(wait=0.05)
    image = np.full((90, 160, 3), 120, np.uint8)
    assert camera.push(jpeg(image), 1) is None and camera.read() is None  # nothing paired yet
    first = camera.connect("Back Camera")
    assert camera.push(b"not a picture", first) is False
    assert camera.push(jpeg(image), first) is True and camera.connected
    frame = camera.read()
    assert frame.shape == (90, 160, 3) and camera.read() is None  # each frame is delivered once
    second = camera.connect("another iPhone")
    assert camera.push(jpeg(image), first) is None and camera.push(jpeg(image), second) is True
    camera.revoke()
    assert camera.push(jpeg(image), second) is None and not camera.connected
    assert camera.view()["resolution"] == [160, 90]


def test_pages_refuse_other_networks_and_the_stream_needs_the_pairing_key(tmp_path):
    authority = load_or_create_authority(tmp_path)
    camera = PhoneCamera(wait=0.05)
    link = SimpleNamespace(key="right-key")

    outside = TestClient(create_camera_app(camera, link))  # TestClient's address is not a private one
    assert outside.get("/camera").status_code == 403
    with pytest.raises(WebSocketDisconnect), outside.websocket_connect("/camera/stream") as ws:
        ws.receive_text()

    setup = TestClient(create_setup_app(authority, allowed=lambda host: True))
    assert authority.fingerprint in setup.get("/").text
    certificate = setup.get("/avavision-ca.crt")
    assert certificate.content == authority.der and certificate.headers["content-type"] == "application/x-x509-ca-cert"

    client = TestClient(create_camera_app(camera, link, allowed=lambda host: True))
    page = client.get("/camera")
    assert page.status_code == 200 and "wss://" in page.text and "right-key" not in page.text
    with client.websocket_connect("/camera/stream") as ws:
        ws.send_text(json.dumps({"key": "wrong-key"}))
        with pytest.raises(WebSocketDisconnect) as refused:
            ws.receive_text()
    assert refused.value.code == 4401 and camera.device is None
    with client.websocket_connect("/camera/stream") as ws:
        ws.send_text(json.dumps({"key": "right-key", "device": "Back Camera · 3840×2160"}))
        ws.send_bytes(jpeg(np.zeros((40, 60, 3), np.uint8)))
        assert ws.receive_text() == "ok"
        ws.send_bytes(b"\xff\xd8 broken")
        assert ws.receive_text() == "bad"
        link.key = "new-key"  # paired again on the station
        camera.revoke()
        ws.send_bytes(jpeg(np.zeros((40, 60, 3), np.uint8)))
        with pytest.raises(WebSocketDisconnect) as dropped:
            ws.receive_text()
    assert dropped.value.code == 4401 and camera.read().shape == (40, 60, 3)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_station_uses_the_iphone_over_https(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_module, "lan_addresses", lambda: [ADDRESS])
    port, setup_port = free_port(), free_port()
    db = Database(tmp_path)
    db.save_setting("station", {"camera_source": "iphone", "phone_port": port, "phone_setup_port": setup_port})
    db.close()
    station = Station.open(tmp_path)
    with TestClient(create_app(station)) as client:
        assert station.phone.wait_started(), station.phone.errors
        status = client.get("/api/status").json()["camera"]
        assert status["source"] == "iphone" and "iPhone not connected" in status["error"]

        # Real TLS, checked against the station's own authority only.
        authority_pem = station.phone.authority.certificate.public_bytes(serialization.Encoding.PEM).decode()
        trust = ssl.create_default_context(cadata=authority_pem)
        with httpx.Client(verify=trust, trust_env=False) as https:
            assert "AvaVision camera" in https.get(f"https://127.0.0.1:{port}/camera").text
        with httpx.Client(trust_env=False) as http:
            assert station.phone.authority.fingerprint in http.get(f"http://127.0.0.1:{setup_port}/").text

        codes = read_codes(cv2.imdecode(np.frombuffer(client.get("/api/phone/camera.png").content, np.uint8), 0))
        key = station.db.setting("phone_key")
        assert codes == [f"https://{ADDRESS}:{port}/camera#k={key}"]

        scene = SyntheticStation()
        frame = jpeg(scene.render(full_scene(scene.layout(), ["metformin-500"])))
        session = station.phone.camera.connect("Back Camera")
        deadline = time.time() + 5
        while station.feed.latest() is None and time.time() < deadline:
            station.phone.camera.push(frame, session)
            time.sleep(0.05)
        view = client.get("/api/phone").json()
        assert view["connected"] and view["resolution"] == [1920, 1080] and view["address"] == ADDRESS
        assert client.get("/api/status").json()["camera"]["error"] is None

        paired = client.post("/api/phone/pair").json()
        assert not paired["connected"] and station.db.setting("phone_key") != key
        assert station.phone.camera.push(frame, session) is None


def test_phone_routes_explain_when_the_iphone_is_not_in_use(tmp_path):
    station = Station.open(tmp_path, demo=True)
    with TestClient(create_app(station)) as client:
        refused = client.get("/api/phone")
        assert refused.status_code == 409 and "iPhone" in refused.json()["error"]
        assert client.get("/api/status").json()["camera"]["source"] == "demo"
