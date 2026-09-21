---
tags: [layer/practice, reference]
code: environment_versions.txt
status: stable
---

# Debugging the Jupyter Restart Button

On the machine ALYOSHA, in July 2026, VS Code failed to restart a notebook kernel. Three
symptoms appeared together. A kernel took an extremely long time to connect, or never
finished connecting. A restart request hung, and the only recovery was restarting VS Code.
The Restart button intermittently disappeared from the notebook toolbar while the kernel
was still alive.

## The working versions

`environment_versions.txt` records the combination that works, committed on 2026-07-22
under the message "environment version (working)":

| component | version |
|---|---|
| Python | 3.13.3 |
| VS Code | 1.129.0 |
| `ms-toolsai.jupyter` | 2025.9.1 |
| ipykernel | 6.29.5 |

The pin still holds. The interpreter on PATH is now 3.14.0 and ipykernel is 6.29.5.

## The failing environment

A diagnostic of the affected machine was generated on 2026-07-22 to diff it
against a second machine where the same notebooks behaved correctly. It records ipykernel
7.1.0, Python 3.14.0, VS Code 1.129.1, `ms-toolsai.jupyter` 2025.9.1, `ms-python.python`
2026.4.0, and `ms-python.vscode-python-envs` not installed. Two orphaned `python.exe`
processes were left behind, and the Jupyter log typed the interpreter as `Unknown` and
reported `Failed to get activated env vars` five times.

The two VS Code extensions are still at those versions, and `vscode-python-envs` is still
absent. Between the failing report and the working environment today, ipykernel is the one
component that changed.

## What the diagnostic ruled out

The report names a version skew between `ms-python.python` and `ms-toolsai.jupyter` as its
leading hypothesis, and rules ipykernel out. A start, execute, restart, execute cycle was
driven through `jupyter_client` in clean virtualenvs on ipykernel 6.31.0, 7.1.0 and 7.3.0,
and all three passed. The test drove the kernel directly rather than through the Jupyter
extension, so it does not cover the path the extension takes.

Python 3.14 was ruled out by the same report, and the eight `ms-toolsai.jupyter` folders on
disk were ruled out because VS Code loads only the highest version.

## The upstream issues

- [microsoft/vscode-python #25804](https://github.com/microsoft/vscode-python/issues/25804)
  reports the `Failed to get activated env vars` timeout blocking kernel startup, as a
  regression introduced in `ms-python.python` 2026.2.0 alongside `vscode-python-envs`
  1.17.x. The reported workaround is to downgrade `ms-python.python` below 2026.2.0 and to
  uninstall `vscode-python-envs`.
- [microsoft/vscode-jupyter #17228](https://github.com/microsoft/vscode-jupyter/issues/17228)
  reports that ipykernel 7 hangs notebook execution. It points at PR #17411, which was a
  draft and has not merged, so no shipped extension carries the fix.
- [microsoft/vscode-jupyter #15280](https://github.com/microsoft/vscode-jupyter/issues/15280)
  reports the Restart button disappearing while the kernel is active. The report ties it to
  switching notebook tabs and treats it as a separate fault from the connect and restart
  hang. The workaround is to switch between notebook tabs until the button reappears.
- [ipython/ipykernel #1469](https://github.com/ipython/ipykernel/issues/1469) reports the
  Windows `ProactorEventLoop` freezing tornado's selector helper thread under the debugger.
  The fix merged after the 7.3.0 release and is in no released version.

## See also

- [[Notebooks]] — the notebooks these kernels run, and how to execute one headlessly
- [[Diagram Display]] — the renderer thread, which runs a proactor loop of its own because
  ipykernel runs a selector loop on Windows
