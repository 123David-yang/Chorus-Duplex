"""Compatible Chat Completions client with opt-in, secret-free per-task telemetry."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

_CALL_LOG = ContextVar("llm_call_log", default=None)

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
        with urllib.request.urlopen(req,timeout=120) as response:
            data=json.loads(response.read().decode("utf-8"))
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
