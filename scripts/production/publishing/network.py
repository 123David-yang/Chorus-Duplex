"""Bounded HF-only HTTP transport. Reuse the existing loopback proxy without changing Windows or other clients."""
import os
from urllib.request import getproxies
from urllib.parse import urlparse
_configured=False

def loopback_proxy():
    if any(os.environ.get(k) for k in ("HTTPS_PROXY","https_proxy","HTTP_PROXY","http_proxy","ALL_PROXY","all_proxy")):return None
    proxies=getproxies();url=proxies.get("https") or proxies.get("http")
    if url and urlparse(url).hostname in ("127.0.0.1","localhost","::1"):return url
    return None

def configure_hf_transport():
    global _configured
    if _configured:return
    os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    import httpx
    from huggingface_hub import constants,set_client_factory
    from huggingface_hub.utils._http import default_client_factory
    # is_xet_available() reads this constant, including when another library imported the SDK earlier.
    constants.HF_HUB_DISABLE_XET=os.environ["HF_HUB_DISABLE_XET"].upper() in ("1","ON","YES","TRUE")
    proxy=loopback_proxy()
    def factory():
        template=default_client_factory();hooks=template.event_hooks;template.close()
        return httpx.Client(proxy=proxy,follow_redirects=True,event_hooks=hooks,timeout=httpx.Timeout(connect=15.,read=60.,write=60.,pool=15.))
    set_client_factory(factory);_configured=True
