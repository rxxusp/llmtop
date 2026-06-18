"""Discovery layer: port scan, cmdline classification, router correlation."""

from __future__ import annotations

import socket

from llmtop.discovery.port_scan import scan_ports
from llmtop.discovery.process_scan import classify_cmdline
from llmtop.discovery.discover import correlate_router
from llmtop.models import EngineInfo, EngineType, ModelInfo


def test_scan_ports_finds_open_and_skips_closed():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    open_port = srv.getsockname()[1]
    try:
        # An almost-certainly-closed port (1 is privileged/unused for us).
        found = scan_ports([open_port, 1], host="127.0.0.1", timeout=0.3)
        assert open_port in found
        assert 1 not in found
    finally:
        srv.close()


def test_classify_cmdline():
    assert classify_cmdline(["python", "-m", "vllm", "serve", "m"]) == EngineType.VLLM
    assert classify_cmdline(["/usr/local/bin/ollama", "serve"]) == EngineType.OLLAMA
    assert classify_cmdline(["llama-server", "-m", "x.gguf"]) == EngineType.LLAMACPP
    assert classify_cmdline(["text-generation-launcher"]) == EngineType.TGI
    assert classify_cmdline(["python", "-m", "sglang.launch_server"]) == EngineType.SGLANG
    assert classify_cmdline(["bash", "-c", "echo hi"]) is None
    assert classify_cmdline([]) is None


def _engine(port, etype, models):
    return EngineInfo(
        engine_type=etype, name=etype.value,
        base_url=f"http://127.0.0.1:{port}", host="127.0.0.1", port=port,
        models=[ModelInfo(id=m) for m in models],
    )


def test_correlate_router_links_backends():
    router = _engine(8077, EngineType.OPENAI, ["A", "B", "C", "D"])
    ollama = _engine(11434, EngineType.OLLAMA, ["A", "B"])
    vllm = _engine(8088, EngineType.VLLM, ["C"])

    out = correlate_router([router, ollama, vllm])
    routers = [e for e in out if e.is_router]
    assert len(routers) == 1
    r = routers[0]
    assert r.engine_type == EngineType.ROUTER
    assert r.port == 8077
    backend_ports = {b.port for b in r.backends}
    assert backend_ports == {11434, 8088}
    assert ollama.routed_by == r.base_url
    assert vllm.routed_by == r.base_url


def test_correlate_router_no_false_positive_on_unrelated_engines():
    # Two unrelated single-model engines must not become a router/backend pair.
    a = _engine(8088, EngineType.VLLM, ["qwen36-coder"])
    b = _engine(11434, EngineType.OLLAMA, ["llama3", "phi3", "mistral"])
    out = correlate_router([a, b])
    assert not any(e.is_router for e in out)
