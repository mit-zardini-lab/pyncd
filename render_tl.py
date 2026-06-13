"""
Render a TL DSL morphism in the tsncd frontend.

Usage:
  1. In the tsncd directory, run:  npm run dev   (starts frontend on port 3000)
  2. Paste your DSL code into the section marked below.
  3. Assign the morphism to `morph` at the end of that section.
  4. Run:  uv run render_tl.py
"""

import asyncio
import socket
import subprocess
import sys
import webbrowser

import display as dpl
import websocket_transfer.websockets_transfer as wst
from data_structure.TensorDSL import TL, real_axis, axes, norm_axis, relu, softmax


# ── PASTE YOUR DSL CODE HERE ─────────────────────────────────────────────────

H, W, C, P, S = 6, 3, 2, 2, 2
H_pad = H + W - 1
H_out = (H - P) // S + 1

x,  y  = real_axis('x',  H),     real_axis('y',  H)
dx, dy = real_axis('dx', W),     real_axis('dy', W)
ch     = real_axis('ch', C)
xi, yi = real_axis('xi', H_pad), real_axis('yi', H_pad)
px, py = real_axis('px', P),     real_axis('py', P)
xo, yo = real_axis('xo', H_out), real_axis('yo', H_out)

tl = TL()
tl.Filter.tensor(dx, dy, ch)
tl.Image.tensor(xi, yi, ch)
tl.Features.tensor(x, y)
tl.Features[x, y] = relu(tl.Filter[dx, dy, ch] * tl.Image[x+dx, y+dy, ch])
tl.Pooled[xo, yo] = tl.Features[S*xo+px, S*yo+py]

morph = tl.to_morphism()

# ─────────────────────────────────────────────────────────────────────────────


def server_is_running(host: str = '127.0.0.1', port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


async def main() -> None:
    try:
        dpl.print_category(morph)  # type: ignore
    except Exception as e:
        print(f'ASCII display skipped ({type(e).__name__}: {e})')

    server = None
    started_here = False
    if server_is_running():
        print('Using existing server at ws://localhost:8765')
    else:
        server = subprocess.Popen([sys.executable, 'run_server.py'])
        started_here = True
        print('Server starting...', end='', flush=True)
        for _ in range(20):
            await asyncio.sleep(0.25)
            if server_is_running():
                break
            print('.', end='', flush=True)
        else:
            print('\nServer did not start in time.')
            return
        print(' ready.')

    webbrowser.open('http://localhost:3000')
    await wst.send_term(morph)
    if started_here:
        print('Server running — press Ctrl+C to stop.')
        try:
            await asyncio.Event().wait()  # wait forever
        except (KeyboardInterrupt, asyncio.CancelledError):
            if server is not None:
                server.kill()


if __name__ == '__main__':
    asyncio.run(main())
