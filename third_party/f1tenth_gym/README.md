# Vendored F1TENTH Simulator

Imported from the user-provided `f1tenth-atlas.zip` on 2026-09-30. The bundled
MIT license is retained in `LICENSE`. Only `f110_gym` simulator sources are
vendored; no Atlas-DeePC algorithm from that archive lives here.

Local compatibility edits:

- `envs/__init__.py` lazily loads `F110Env` instead of importing rendering at
  package import time.
- `envs/f110_env.py` no longer imports pyglet/OpenGL at module scope; its existing
  render method still imports the renderer when rendering is requested.

These changes allow headless dynamics/collision replay. The numerical dynamics,
laser and collision routines are unchanged. The experiment adapter overrides
the agent update method during replay to apply the saved raw steering-rate and
acceleration inputs with the same RK4 plant used during data collection.

Map assets live in `data/f1tenth/maps/`. Diagnostic videos use the project's
Matplotlib renderer rather than the simulator's interactive OpenGL window.
