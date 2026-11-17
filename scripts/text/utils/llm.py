"""Compatible Chat Completions client with opt-in, secret-free per-task telemetry."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import http.client
import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

_CALL_LOG = ContextVar("llm_call_log", default=None)


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Invalid LLM endpoint URL")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid LLM endpoint URL") from None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, parsed.hostname.lower(), port


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the LLM authorization header on its original origin."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        try:
            allowed = _origin(request.full_url) == _origin(newurl)
        except ValueError:
            allowed = False
        if not allowed:
            raise RuntimeError("LLM request blocked a cross-origin redirect") from None
        return super().redirect_request(request, fp, code, msg, headers, newurl)


@contextmanager
def capture_calls():
    """One independent collector per thread/task; never retains prompts, URLs or keys."""
    records=[]
    token=_CALL_LOG.set(records)
    try: yield records
    finally: _CALL_LOG.reset(token)


def _config() -> tuple[str,str,str]:
    base=os.environ.get("LLM_BASE_URL","https://api.deepseek.com/v1")
    key=os.environ.get("LLM_API_KEY","")
    model=os.environ.get("LLM_MODEL","deepseek-chat")
    if not key: raise RuntimeError("LLM_API_KEY is not configured")
    scheme, host, _ = _origin(base)
    parsed = urllib.parse.urlsplit(base)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LLM endpoint URL must not contain credentials, a query or a fragment")
    if scheme == "http":
        try:
            loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("Remote LLM endpoints require HTTPS")
    return base,key,model


def chat(system: str,user: str,temperature: float=1.0,max_tokens: int=512, *, json_output: bool=False) -> str:
    started=time.perf_counter()
    meta={"success":False,"max_tokens":max_tokens}
    try:
        base,key,model=_config()
        meta["model"]=model
        payload = {"model":model,"messages":[{"role":"system","content":system},
                   {"role":"user","content":user}],"temperature":temperature,"max_tokens":max_tokens}
        if json_output and urllib.parse.urlsplit(base).hostname == "api.deepseek.com":
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        req=urllib.request.Request(f"{base.rstrip('/')}/chat/completions",data=body,
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},method="POST")
        opener = urllib.request.build_opener(_SameOriginRedirectHandler())
        for attempt in range(2):
            try:
                with opener.open(req, timeout=120) as response:
                    try:
                        data = json.loads(response.read().decode("utf-8"))
                    except http.client.IncompleteRead as exc:
                        # Some servers close early despite sending a complete JSON body.
                        # Accept only a parseable, explicitly finished response; never retry
                        # an already-paid request merely because the connection closed.
                        meta["incomplete_bytes"] = len(exc.partial)
                        try:
                            candidate = json.loads(exc.partial.decode("utf-8"))
                            choice = candidate["choices"][0]
                            content = choice["message"]["content"]
                            if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
                                raise ValueError("response not complete")
                        except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                            raise RuntimeError("LLM response was incomplete") from None
                        data = candidate
                        meta["recovered_complete_json"] = True
                break
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError as exc:
                # DNS and connection refusal occur before the service receives a request.
                if not attempt and isinstance(exc.reason, (socket.gaierror, ConnectionRefusedError)):
                    time.sleep(.5)
                    continue
                raise RuntimeError("LLM network request failed") from None
            except http.client.IncompleteRead:
                # The provider may have processed this paid request; avoid sending it twice.
                raise RuntimeError("LLM response was incomplete") from None
        choice=data["choices"][0]
        content=choice["message"]["content"]
        if not isinstance(content,str) or not content.strip(): raise ValueError("Empty model response")
        usage=data.get("usage",{})
        meta.update(success=True,finish_reason=choice.get("finish_reason"),
                    usage={k:v for k,v in usage.items() if k in (
                        "prompt_tokens","completion_tokens","total_tokens",
                        "prompt_cache_hit_tokens","prompt_cache_miss_tokens") and isinstance(v,int)})
        return content.strip()
    except urllib.error.HTTPError as exc:
        meta.update(error_type="HTTPError",http_status=exc.code)
        # Remote bodies can echo sensitive request data. Keep only the HTTP status.
        raise RuntimeError(f"LLM API HTTP {exc.code}") from None
    except urllib.error.URLError:
        meta["error_type"]="URLError"
        raise RuntimeError("LLM network request failed") from None
    except Exception as exc:
        meta["error_type"]=type(exc).__name__
        raise
    finally:
        meta["elapsed_s"]=round(time.perf_counter()-started,4)
        records=_CALL_LOG.get()
        if records is not None: records.append(meta)
