#!/usr/bin/env python3
"""
Wakanda Voice Pipeline — Single-command launcher.

Starts all model servers, the orchestrator, the token server,
and optionally LiveKit Server — all from one command.

Usage:
    python launch.py --config configs/default.yaml

    # With LiveKit (if installed):
    python launch.py --config configs/default.yaml --livekit

    # CPU only (no GPU):
    python launch.py --config configs/default.yaml --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Colors for terminal output
C_RESET = "\033[0m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_DIM = "\033[90m"


def colored(text, color):
    return f"{color}{text}{C_RESET}"


class ServiceProcess:
    """Wraps a subprocess with health checking."""

    def __init__(self, name: str, cmd: list[str], port: int, env: dict | None = None):
        self.name = name
        self.cmd = cmd
        self.port = port
        self.process: subprocess.Popen | None = None
        self.env = {**os.environ, **(env or {})}

    def start(self) -> None:
        logger.info(f"Starting {colored(self.name, C_CYAN)} on port {self.port}")
        self.process = subprocess.Popen(
            self.cmd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.process and self.is_alive():
            logger.info(f"Stopping {self.name}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    async def wait_ready(self, timeout: float = 120) -> bool:
        """Wait for the service to be ready by checking if port is open."""
        import socket
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if not self.is_alive():
                # Process died — read output for error
                if self.process and self.process.stdout:
                    output = self.process.stdout.read()
                    logger.error(f"{self.name} died:\n{output[-500:]}")
                return False
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(("localhost", self.port))
                sock.close()
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                await asyncio.sleep(1)
        return False


async def log_output(service: ServiceProcess):
    """Stream a service's stdout to the terminal with a prefix."""
    if not service.process or not service.process.stdout:
        return
    loop = asyncio.get_event_loop()
    while service.is_alive():
        line = await loop.run_in_executor(None, service.process.stdout.readline)
        if line:
            tag = colored(f"[{service.name}]", C_DIM)
            print(f"  {tag} {line.rstrip()}")
        else:
            break


async def main():
    parser = argparse.ArgumentParser(description="Wakanda Voice — Launch all services")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda", help="Device for model servers (cuda or cpu)")
    parser.add_argument("--livekit", action="store_true", help="Also start LiveKit Server")
    parser.add_argument("--frontend-port", type=int, default=3000)
    parser.add_argument("--no-orchestrator", action="store_true", help="Skip LiveKit orchestrator (model servers only)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    python = sys.executable
    project_root = str(Path(__file__).parent)

    services: list[ServiceProcess] = []

    # ── Parse ports from config URLs ──
    def port_from_url(url: str) -> int:
        return int(url.split(":")[-1].split("/")[0])

    stt_port = port_from_url(config["stt"]["url"])
    mt_port = port_from_url(config["mt"]["url"])
    tts_port = port_from_url(config["tts"]["url"])

    # ── LiveKit Server (optional) ──
    if args.livekit:
        services.append(ServiceProcess(
            name="LiveKit",
            cmd=["livekit-server", "--dev", "--bind", "0.0.0.0"],
            port=7880,
        ))

    # ── STT Server ──
    stt_model = config["stt"]["model"]
    default_lang = config.get("default_lang", "eng")
    services.append(ServiceProcess(
        name="STT",
        cmd=[python, f"{project_root}/servers/stt_server.py",
             "--model", stt_model,
             "--language", default_lang,
             "--port", str(stt_port),
             "--device", args.device,
             "--no-preload"],
        port=stt_port,
    ))

    # ── MT Server ──
    mt_model = config["mt"]["model"]
    services.append(ServiceProcess(
        name="MT",
        cmd=[python, f"{project_root}/servers/mt_server.py",
             "--model", mt_model,
             "--port", str(mt_port),
             "--device", args.device,
             "--no-preload"],
        port=mt_port,
    ))

    # ── TTS Server ──
    tts_model = config["tts"]["model"]
    services.append(ServiceProcess(
        name="TTS",
        cmd=[python, f"{project_root}/servers/tts_server.py",
             "--model", tts_model,
             "--port", str(tts_port),
             "--device", args.device,
             "--no-preload"],
        port=tts_port,
    ))

    # ── Token + Frontend Server ──
    services.append(ServiceProcess(
        name="Frontend",
        cmd=[python, f"{project_root}/scripts/token_server.py",
             "--port", str(args.frontend_port)],
        port=args.frontend_port,
    ))

    # ── Handle Ctrl+C ──
    def shutdown_handler(sig, frame):
        print(f"\n{colored('Shutting down all services...', C_YELLOW)}")
        for svc in reversed(services):
            svc.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # ── Start all services ──
    print(f"\n{colored('🚀 Wakanda Voice Pipeline', C_GREEN)}")
    print(f"{colored('=' * 50, C_DIM)}")
    print(f"  Config:  {args.config}")
    print(f"  Device:  {args.device}")
    print(f"  Mode:    {config.get('mode', 'unknown')}")
    print(f"  STT:     {stt_model}")
    print(f"  MT:      {mt_model}")
    print(f"  TTS:     {tts_model}")
    print(f"{colored('=' * 50, C_DIM)}\n")

    for svc in services:
        svc.start()
        # Small delay between starts to avoid GPU contention
        await asyncio.sleep(1)

    # ── Wait for all services to be ready ──
    print(f"\n{colored('Waiting for services to be ready...', C_YELLOW)}\n")

    log_tasks = []
    for svc in services:
        log_tasks.append(asyncio.create_task(log_output(svc)))

    all_ready = True
    for svc in services:
        ready = await svc.wait_ready(timeout=180)
        if ready:
            print(f"  {colored('✓', C_GREEN)} {svc.name} ready on port {svc.port}")
        else:
            print(f"  {colored('✗', C_RED)} {svc.name} failed to start")
            all_ready = False

    if not all_ready:
        print(f"\n{colored('Some services failed to start. Check logs above.', C_RED)}")
        for svc in reversed(services):
            svc.stop()
        return

    # ── Start orchestrator (if LiveKit mode) ──
    if args.livekit and not args.no_orchestrator:
        orch_svc = ServiceProcess(
            name="Orchestrator",
            cmd=[python, f"{project_root}/orchestrator/main.py",
                 "--config", args.config],
            port=0,  # orchestrator doesn't listen on its own port
        )
        orch_svc.start()
        services.append(orch_svc)
        asyncio.create_task(log_output(orch_svc))
        print(f"  {colored('✓', C_GREEN)} Orchestrator started")

    # ── Ready! ──
    print(f"\n{colored('=' * 50, C_DIM)}")
    print(f"{colored('✅ All services running!', C_GREEN)}\n")
    print(f"  🌐 Open {colored(f'http://localhost:{args.frontend_port}', C_CYAN)} in your browser")
    if args.livekit:
        print(f"  📡 LiveKit Server at {colored('ws://localhost:7880', C_CYAN)}")
    print(f"  🛑 Press {colored('Ctrl+C', C_YELLOW)} to stop all services")
    print(f"{colored('=' * 50, C_DIM)}\n")

    # Keep running, streaming logs
    try:
        while True:
            # Check if any service died
            for svc in services:
                if svc.process and not svc.is_alive() and svc.port > 0:
                    print(f"\n{colored(f'⚠ {svc.name} died!', C_RED)}")
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass
    finally:
        for svc in reversed(services):
            svc.stop()


if __name__ == "__main__":
    asyncio.run(main())
