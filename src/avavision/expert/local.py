"""A second opinion from a vision-language model running on this computer (Qwen3-VL through Ollama).

Same job, prompt and answer structure as :meth:`ClaudeExpert.review_compartment`, but offline: the model runs in
Ollama (MIT licence) on the station or another computer in the pharmacy, and nothing leaves that network. The
default model, Qwen3-VL 8B Instruct, is Apache-2.0 (D-137).

Like every advisor it is never in the acceptance path: a disagreement sends the compartment to review, agreement
or uncertainty changes nothing (``core.advisory``).
"""

from __future__ import annotations

import base64
import ipaddress
import json
import time
from urllib.parse import urlsplit

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

from ..core.advisory import AdvisoryOpinion
from ..core.models import Catalog, CompartmentIndex, PackLayout, PackProfile
from .claude import (
    REVIEW_SYSTEM,
    CompartmentReading,
    ExpertAnswer,
    ExpertCall,
    ExpertIncomplete,
    ExpertUnavailable,
    encode_jpeg,
    opinion_from_reading,
    review_question,
)

DEFAULT_MODEL = "qwen3-vl:8b-instruct"


class LocalAdvisorSettings(BaseModel):
    enabled: bool = False
    #: Ollama on this computer, or on another computer of the pharmacy network (never the internet).
    url: str = "http://127.0.0.1:11434"
    model: str = DEFAULT_MODEL
    #: Review every accepted compartment right after capture. Needs a graphics card: on a processor alone each
    #: compartment takes tens of seconds.
    automatic: bool = False
    timeout_seconds: float = 120.0
    #: Longest image side sent to the model; Qwen3-VL works at any resolution, smaller is faster.
    image_side: int = 768

    @field_validator("url")
    @classmethod
    def _local_only(cls, url: str) -> str:
        return check_local_url(url)


def check_local_url(url: str) -> str:
    """The URL without a trailing slash, if it points to this computer or a private network address."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("the local model address must look like http://127.0.0.1:11434")
    host = parts.hostname
    if host != "localhost":
        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError("use the IP address of the computer running the model, not a name") from error
        if not (address.is_private or address.is_loopback or address.is_link_local):
            raise ValueError("the local model must run on this computer or the pharmacy network")
    return url.strip().rstrip("/")


class LocalStatus(BaseModel):
    running: bool
    model_installed: bool
    models: list[str] = []
    problem: str | None = None
    #: Share of the loaded model in graphics-card memory (0 = processor only); ``None`` until it has been loaded.
    graphics_share: float | None = None


class LocalVisionAdvisor:
    def __init__(self, settings: LocalAdvisorSettings | None = None, transport: httpx.BaseTransport | None = None):
        self.settings = settings or LocalAdvisorSettings()
        self.base = check_local_url(self.settings.url)
        self._client = httpx.Client(transport=transport, timeout=self.settings.timeout_seconds, trust_env=False)

    @property
    def name(self) -> str:
        return self.settings.model

    def status(self) -> LocalStatus:
        """Whether Ollama answers and the configured model is installed."""
        try:
            response = self._client.get(f"{self.base}/api/tags", timeout=3.0)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as error:
            return LocalStatus(running=False, model_installed=False, problem=f"Ollama is not answering ({error})")
        wanted = self.settings.model if ":" in self.settings.model else f"{self.settings.model}:latest"
        installed = wanted in models
        problem = None if installed else f"the model is not installed: run  ollama pull {self.settings.model}"
        return LocalStatus(
            running=True,
            model_installed=installed,
            models=models,
            problem=problem,
            graphics_share=self._graphics_share(wanted),
        )

    def _graphics_share(self, model: str) -> float | None:
        """How much of the loaded model sits in graphics-card memory, from Ollama's list of running models."""
        try:
            response = self._client.get(f"{self.base}/api/ps", timeout=3.0)
            response.raise_for_status()
            loaded = [m for m in response.json().get("models", []) if m.get("name") == model and m.get("size")]
        except (httpx.HTTPError, ValueError):
            return None
        if not loaded:
            return None
        return round(max(min(1.0, (m.get("size_vram") or 0) / m["size"]) for m in loaded), 2)

    def review_compartment(
        self,
        image: np.ndarray,
        index: CompartmentIndex,
        layout: PackLayout,
        profile: PackProfile,
        catalog: Catalog,
    ) -> ExpertAnswer[AdvisoryOpinion]:
        question, total = review_question(index, layout, profile, catalog)
        picture = base64.standard_b64encode(encode_jpeg(image, max_side=self.settings.image_side)).decode()
        request = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": REVIEW_SYSTEM},
                {"role": "user", "content": question, "images": [picture]},
            ],
            "format": CompartmentReading.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0},
            "keep_alive": "30m",
        }
        start = time.perf_counter()
        try:
            response = self._client.post(f"{self.base}/api/chat", json=request)
        except httpx.TimeoutException as error:
            raise ExpertUnavailable(
                f"the local model took longer than {self.settings.timeout_seconds:.0f} s for one compartment: "
                "it needs a graphics card, or a smaller model"
            ) from error
        except httpx.HTTPError as error:
            raise ExpertUnavailable(f"the local model is not answering at {self.base}: is Ollama running?") from error
        if response.status_code == 404:
            raise ExpertUnavailable(f"the model is not installed: run  ollama pull {self.settings.model}")
        if response.status_code >= 400:
            raise ExpertUnavailable(f"the local model failed ({response.status_code}): {response.text[:200]}")
        try:
            data = response.json()
            if data.get("done_reason") == "length":
                raise ExpertIncomplete("the answer was cut off")
            reading = CompartmentReading.model_validate_json(data["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as error:
            raise ExpertIncomplete("the answer did not match the expected structure") from error
        call = ExpertCall(
            requested_model=self.settings.model,
            served_model=f"{data.get('model') or self.settings.model} (local)",
            fallback_used=False,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            seconds=round(time.perf_counter() - start, 2),
        )
        return ExpertAnswer(value=opinion_from_reading(reading, index, total, call), call=call)

    def close(self) -> None:
        self._client.close()
