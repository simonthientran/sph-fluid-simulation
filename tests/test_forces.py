"""
DE:
Unit-Tests für `sph_sim.core.forces`.

Wir testen hier speziell:
- `compute_pressure_acceleration(...)`

Worum geht es?
-------------
Diese Funktion berechnet die Beschleunigung durch Druckkräfte in SPH.
Wir testen NICHT die komplette Physik einer Simulation, sondern nur:
- Form (Shapes) der Ausgabe
- "Triviale" Situation: konstanter Druck → keine Druckgradienten → keine Beschleunigung
- Input-Validierung (frühe, klare Fehler)

EN:
Unit tests for `sph_sim.core.forces`.

We specifically test:
- `compute_pressure_acceleration(...)`

What is this about?
-------------------
This function computes acceleration due to pressure forces in SPH.
We are NOT testing a full simulation, only:
- output shapes
- a "trivial" situation: constant pressure → no pressure gradients → no acceleration
- input validation (early, clear errors)
"""

from pathlib import Path
import sys

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest

from sph_sim.core.density_grid import compute_density_grid
from sph_sim.core.density import compute_density_naive
from sph_sim.core.forces import compute_pressure_acceleration
from sph_sim.core.particles import initialize_particles_cube


def test_pressure_acceleration_shape() -> None:
    """
    DE:
    Test A: Shape-Test.

    Wir erwarten:
    - ax und ay sind 1D-Arrays der Länge N (Anzahl Partikel).

    EN:
    Test A: shape test.

    We expect:
    - ax and ay are 1D arrays of length N (number of particles).
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n

    # --- Deutsch ---
    # Wir brauchen rho und p als Inputs.
    # rho nehmen wir aus einer SPH-Dichteberechnung (Grid oder Naiv ist egal für diesen Shape-Test).
    #
    # --- English ---
    # We need rho and p as inputs.
    # We take rho from an SPH density computation (grid or naive does not matter for this shape test).
    h = 0.6
    rho = compute_density_grid(particles=particles, h=h)

    # --- Deutsch ---
    # Druckwerte können erst mal irgendwas sein; hier nehmen wir 0 (konstant).
    #
    # --- English ---
    # Pressure values can be anything for shape; here we use 0 (constant).
    p = np.zeros(N, dtype=np.float64)

    ax, ay = compute_pressure_acceleration(particles=particles, rho=rho, p=p, h=h)

    assert isinstance(ax, np.ndarray)
    assert isinstance(ay, np.ndarray)
    assert ax.shape == (N,)
    assert ay.shape == (N,)


def test_pressure_acceleration_constant_pressure_is_zero() -> None:
    """
    DE:
    Test B: Konstanter Druck → keine Beschleunigung.

    Physikalische Idee:
    - Druckkräfte entstehen durch Druck-UNTERSCHIEDE im Raum (Gradient).
    - Wenn der Druck überall gleich ist, gibt es keinen Druckgradienten.
    - Dann sollte die Druckbeschleunigung 0 sein.

    Didaktische Vereinfachung:
    - Wir wählen hier einen konstanten Druck von p = 0.
    - Damit wird der Faktor in der Summe exakt 0 und wir erwarten ax=ay=0 exakt (bis auf Rundung).

    EN:
    Test B: constant pressure → no acceleration.

    Physical idea:
    - Pressure forces come from pressure differences in space (a gradient).
    - If pressure is the same everywhere, there is no pressure gradient.
    - Then pressure acceleration should be 0.

    Didactic simplification:
    - We choose a constant pressure p = 0.
    - Then the factor in the sum is exactly 0 and we expect ax=ay=0 (up to rounding).
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n

    h = 0.6

    # --- Deutsch ---
    # Dichte berechnen (wir nutzen hier die Grid-Version).
    # Falls du später nur die naive Version nutzen willst, geht das auch:
    # rho = compute_density_naive(particles=particles, h=h)
    #
    # --- English ---
    # Compute density (we use the grid version here).
    # If later you only want the naive version, that also works:
    # rho = compute_density_naive(particles=particles, h=h)
    rho = compute_density_grid(particles=particles, h=h)

    # --- Deutsch ---
    # Konstanter Druck: überall 0.0.
    #
    # --- English ---
    # Constant pressure: 0.0 everywhere.
    p = np.zeros(N, dtype=np.float64)

    ax, ay = compute_pressure_acceleration(particles=particles, rho=rho, p=p, h=h)

    # --- Deutsch ---
    # "allclose" ist hier sehr konservativ; praktisch sollten es exakt Nullen sein.
    #
    # --- English ---
    # "allclose" is very conservative here; in practice these should be exact zeros.
    assert np.allclose(ax, 0.0)
    assert np.allclose(ay, 0.0)


def test_pressure_acceleration_invalid_inputs_raise() -> None:
    """
    DE:
    Test C: Ungültige Inputs → ValueError.

    Warum ist das wichtig?
    - Wenn Inputs falsch sind (z.B. h <= 0 oder Shapes passen nicht),
      ist das Ergebnis nicht sinnvoll.
    - Besser: früh abbrechen mit einer klaren Fehlermeldung.

    Wir testen:
    - h <= 0
    - rho hat falsche Shape
    - p hat falsche Shape

    EN:
    Test C: invalid inputs → ValueError.

    Why is this important?
    - If inputs are wrong (e.g., h <= 0 or shapes do not match),
      the result is not meaningful.
    - Better: fail early with a clear error message.

    We test:
    - h <= 0
    - rho has the wrong shape
    - p has the wrong shape
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n

    # --- Deutsch ---
    # Erstmal gültige Inputs vorbereiten.
    #
    # --- English ---
    # Prepare valid inputs first.
    h_valid = 0.6
    rho_valid = compute_density_naive(particles=particles, h=h_valid)
    p_valid = np.zeros(N, dtype=np.float64)

    # --- Deutsch ---
    # h <= 0 muss Fehler werfen.
    #
    # --- English ---
    # h <= 0 must raise an error.
    with pytest.raises(ValueError):
        compute_pressure_acceleration(particles=particles, rho=rho_valid, p=p_valid, h=0.0)
    with pytest.raises(ValueError):
        compute_pressure_acceleration(particles=particles, rho=rho_valid, p=p_valid, h=-1.0)

    # --- Deutsch ---
    # Falsche Shape für rho: Länge passt nicht.
    #
    # --- English ---
    # Wrong shape for rho: length does not match.
    rho_wrong_len = np.zeros(N + 1, dtype=np.float64)
    with pytest.raises(ValueError):
        compute_pressure_acceleration(particles=particles, rho=rho_wrong_len, p=p_valid, h=h_valid)

    # --- Deutsch ---
    # Falsche Shape für rho: nicht 1D.
    #
    # --- English ---
    # Wrong shape for rho: not 1D.
    rho_wrong_ndim = np.zeros((N, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        compute_pressure_acceleration(particles=particles, rho=rho_wrong_ndim, p=p_valid, h=h_valid)

    # --- Deutsch ---
    # Falsche Shape für p: Länge passt nicht.
    #
    # --- English ---
    # Wrong shape for p: length does not match.
    p_wrong_len = np.zeros(N + 1, dtype=np.float64)
    with pytest.raises(ValueError):
        compute_pressure_acceleration(particles=particles, rho=rho_valid, p=p_wrong_len, h=h_valid)

    # --- Deutsch ---
    # Falsche Shape für p: nicht 1D.
    #
    # --- English ---
    # Wrong shape for p: not 1D.
    p_wrong_ndim = np.zeros((N, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        compute_pressure_acceleration(particles=particles, rho=rho_valid, p=p_wrong_ndim, h=h_valid)


