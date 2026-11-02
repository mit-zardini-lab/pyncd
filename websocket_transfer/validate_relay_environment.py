'''Check the relay's HOST and PORT environment configuration.

    python websocket_transfer/validate_relay_environment.py

Each configuration is read by a fresh Python process so that import-time constants
cannot retain a previous case's environment.
'''
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from collections.abc import Callable


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
READ_CONFIGURATION = (
    'import websocket_transfer.websockets_transfer as relay; '
    'print(repr((relay.SERVER_HOST, relay.SERVER_PORT, relay.SERVER_URI)))'
)


def run_relay(script: str, **variables: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop('HOST', None)
    environment.pop('PORT', None)
    environment.update(variables)
    return subprocess.run(
        [sys.executable, '-c', script], cwd=PROJECT_ROOT, env=environment,
        capture_output=True, text=True, timeout=15, check=False)


def check_defaults_and_blank_variables() -> None:
    for variables in ({}, {'HOST': '', 'PORT': ''}):
        process = run_relay(READ_CONFIGURATION, **variables)
        expected = "('localhost', 8765, 'ws://localhost:8765')\n"
        if process.returncode or process.stdout != expected:
            raise AssertionError(f'{variables}: {process.stdout!r} {process.stderr!r}')


def check_ipv6_address_and_configured_port() -> None:
    process = run_relay(READ_CONFIGURATION, HOST='::1', PORT='12345')
    expected = "('::1', 12345, 'ws://[::1]:12345')\n"
    if process.returncode or process.stdout != expected:
        raise AssertionError(f'{process.stdout!r} {process.stderr!r}')


def check_invalid_ports_are_rejected() -> None:
    for port in ('abc', '1.5', '0', '-1', '+1', '65536', '１２３', ' '):
        process = run_relay(READ_CONFIGURATION, PORT=port)
        expected = f'PORT must be an integer from 1 to 65535: {port!r}'
        if process.returncode == 0 or expected not in process.stderr:
            raise AssertionError(
                f'PORT={port!r}: {process.stdout!r} {process.stderr!r}')


def check_configured_address_and_port_connect_the_server_and_data_client() -> None:
    process = run_relay('''
import asyncio
import contextlib
import json
import os
import socket

import websockets

with socket.socket() as reservation:
    reservation.bind(('127.0.0.1', 0))
    os.environ['PORT'] = str(reservation.getsockname()[1])
os.environ['HOST'] = '127.0.0.1'

import websocket_transfer.websockets_transfer as relay

async def check_roundtrip():
    server = relay.DataServer()
    server_task = asyncio.create_task(server.main())
    try:
        for _ in range(100):
            if server_task.done():
                await server_task
            try:
                async with websockets.connect(relay.SERVER_URI):
                    break
            except OSError:
                await asyncio.sleep(0.01)
        else:
            raise TimeoutError(f'Server did not bind {relay.SERVER_URI}')
        handshake = json.dumps({
            'msgType': 'identify', 'clientType': 'DataClient',
            'clientVersion': '0.1.0', 'clientID': 'environment-validator'})
        client = relay.DataClient(handshake=handshake, data='test-data')
        await asyncio.wait_for(client.main(), timeout=5)
        if server.data_structure != 'test-data':
            raise AssertionError(f'No data received on {relay.SERVER_URI}')
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task

asyncio.run(check_roundtrip())
''')
    if process.returncode:
        raise AssertionError(f'{process.stdout!r} {process.stderr!r}')


def main() -> None:
    checks: tuple[Callable[[], None], ...] = (
        check_defaults_and_blank_variables,
        check_ipv6_address_and_configured_port,
        check_invalid_ports_are_rejected,
        check_configured_address_and_port_connect_the_server_and_data_client,
    )
    failed = False
    for check in checks:
        try:
            check()
        except Exception as error:
            failed = True
            print(f'FAIL {check.__name__}: {error}')
        else:
            print(f'PASS {check.__name__}')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
