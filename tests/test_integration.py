"""
DE:
Pytest-Tests für die Zeitintegration (Integrator).

Wir testen hier die Funktion:
`step_semi_implicit_euler(particles, ax, ay, dt)`

Wichtig:
- Das sind absichtlich einfache, kleine Tests.
- Wir prüfen nur: "Rechnet der Integrator genau das, was er soll?"
- Wir verändern keine Physik. Wir testen nur den Integrationsschritt.

EN:
Pytest tests for time integration (integrator).

We test the function:
`step_semi_implicit_euler(particles, ax, ay, dt)`

Important:
- These are intentionally small, simple tests.
- We only check: "Does the integrator do exactly what it should do?"
- We do not change physics. We test only the integration step.
"""

# --- Deutsch ---
# -----------------------------------------------------------------------------
# src/-Layout: Warum müssen wir `src/` in `sys.path` einfügen?
# -----------------------------------------------------------------------------
# Dieses Projekt nutzt eine "src-Struktur": der Python-Code liegt in `src/`.
# Wenn wir Tests starten, kennt Python diesen Ordner nicht automatisch als Import-Pfad.
#
# Darum fügen wir `src/` hier einmalig zum Suchpfad hinzu, damit `import sph_sim...` klappt.
# Später kann man das professionell über Packaging lösen (z.B. `pip install -e .`),
# dann braucht man diesen sys.path-Block nicht mehr.
#
# --- English ---
# -----------------------------------------------------------------------------
# src layout: why do we need to add `src/` to `sys.path`?
# -----------------------------------------------------------------------------
# This project uses a "src structure": the Python code lives in `src/`.
# When we start tests, Python does not automatically know this folder as an import path.
#
# Therefore we add `src/` once to the search path here so that `import sph_sim...` works.
# Later, you can solve this professionally via packaging (e.g., `pip install -e .`),
# then you no longer need this sys.path block.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest

from sph_sim.core.particles import initialize_particles_cube
from sph_sim.core.integration import step_semi_implicit_euler


def test_step_semi_implicit_euler_zero_acceleration_position_unchanged() -> None:
    """
    DE:
    Test A: Nullbeschleunigung → Position bleibt gleich (wenn v am Anfang 0 ist).

    Idee:
    - Wenn ax = 0 und ay = 0 gilt:
      v bleibt gleich.
    - Wenn v am Anfang 0 ist, dann bleibt auch die Position gleich.

    Wichtig:
    - Wenn v am Anfang NICHT 0 wäre, würde sich x trotzdem ändern (x += v*dt),
      auch ohne Beschleunigung. Darum wählen wir bewusst einen Startzustand mit v=0.

    EN:
    Test A: zero acceleration → position stays the same (if v is 0 initially).

    Idea:
    - If ax = 0 and ay = 0:
      v stays the same.
    - If v is 0 initially, then position also stays the same.

    Important:
    - If v was NOT 0 initially, x would still change (x += v*dt),
      even with zero acceleration. That is why we deliberately start with v=0.
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n

    # --- Deutsch ---
    # Wir merken uns die Position VOR dem Integrationsschritt.
    #
    # --- English ---
    # We store the position BEFORE the integration step.
    x_before = particles.x.copy()
    y_before = particles.y.copy()

    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    dt = 0.1

    step_semi_implicit_euler(particles=particles, ax=ax, ay=ay, dt=dt)

    # --- Deutsch ---
    # Position soll unverändert bleiben (weil v=0 und a=0).
    #
    # --- English ---
    # Position should remain unchanged (because v=0 and a=0).
    assert np.allclose(particles.x, x_before)
    assert np.allclose(particles.y, y_before)


def test_step_semi_implicit_euler_constant_ax_updates_v_and_x() -> None:
    """
    DE:
    Test B: Konstante ax → v und x ändern sich korrekt (Semi-Implicit Euler).

    Wir testen eine einfache Rechnung, die man von Hand nachvollziehen kann:
    - Start: v_x = 0
    - Beschleunigung: a_x = 2.0 (konstant)
    - dt = 0.1

    Schritt 1 (Geschwindigkeit):
    - v_x_new = v_x_old + a_x * dt = 0 + 2.0 * 0.1 = 0.2

    Schritt 2 (Position, semi-implicit nutzt v_new):
    - x_new = x_old + v_x_new * dt = x_old + 0.2 * 0.1 = x_old + 0.02

    EN:
    Test B: constant ax → v and x update correctly (semi-implicit Euler).

    We test a simple computation that can be verified by hand:
    - Start: v_x = 0
    - Acceleration: a_x = 2.0 (constant)
    - dt = 0.1

    Step 1 (velocity):
    - v_x_new = v_x_old + a_x * dt = 0 + 2.0 * 0.1 = 0.2

    Step 2 (position, semi-implicit uses v_new):
    - x_new = x_old + v_x_new * dt = x_old + 0.2 * 0.1 = x_old + 0.02
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n

    x_before = particles.x.copy()
    y_before = particles.y.copy()

    ax = np.full(N, 2.0, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    dt = 0.1

    step_semi_implicit_euler(particles=particles, ax=ax, ay=ay, dt=dt)

    # --- Deutsch ---
    # Erwartete Ergebnisse (siehe Rechnung im Docstring):
    #
    # --- English ---
    # Expected results (see the hand calculation in the docstring):
    vx_expected = np.full(N, 0.2, dtype=np.float64)
    vy_expected = np.zeros(N, dtype=np.float64)
    x_expected = x_before + 0.02
    y_expected = y_before + 0.0

    # --- Deutsch ---
    # Wir testen v und x getrennt, damit klar ist, welcher Schritt was macht.
    #
    # --- English ---
    # We test v and x separately so it is clear which step does what.
    assert np.allclose(particles.vx, vx_expected)
    assert np.allclose(particles.vy, vy_expected)
    assert np.allclose(particles.x, x_expected)
    assert np.allclose(particles.y, y_expected)


def test_step_semi_implicit_euler_invalid_dt_raises() -> None:
    """
    DE:
    Test C: dt <= 0 → ValueError.

    Warum testen wir das?
    - Ein Zeitschritt dt muss positiv sein, sonst ist es kein sinnvoller Schritt "nach vorne".
    - Gute Validierung spart später sehr viel Debugging.

    EN:
    Test C: dt <= 0 → ValueError.

    Why do we test this?
    - A time step dt must be positive, otherwise it is not a meaningful step "forward".
    - Good validation saves a lot of debugging later.
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)

    with pytest.raises(ValueError):
        step_semi_implicit_euler(particles=particles, ax=ax, ay=ay, dt=0.0)

    with pytest.raises(ValueError):
        step_semi_implicit_euler(particles=particles, ax=ax, ay=ay, dt=-1.0)


