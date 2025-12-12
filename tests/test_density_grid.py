"""
DE:
Pytest-Tests für die grid-basierte SPH-Dichteberechnung.

Worum geht es?
-------------
Wir testen `compute_density_grid` aus `src/sph_sim/core/density_grid.py`.

Wichtig (Didaktik):
-------------------
Die Physik ist identisch zur naiven Methode (`compute_density_naive`):

    rho[i] = sum_j m[j] * W(r_ij, h)

Der Unterschied ist nur die Nachbarsuche:
- Naiv: prüft alle Paare (i, j).
- Grid: prüft nur Kandidaten aus dem Uniform Grid und überspringt r > h.

EN:
Pytest tests for the grid-based SPH density computation.

What is this about?
-------------------
We test `compute_density_grid` from `src/sph_sim/core/density_grid.py`.

Important (teaching):
---------------------
The physics is identical to the naive method (`compute_density_naive`):

    rho[i] = sum_j m[j] * W(r_ij, h)

The difference is only the neighbor search:
- Naive: checks all (i, j) pairs.
- Grid: checks only candidates from the uniform grid and skips r > h.
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
from sph_sim.core.density import compute_density_naive
from sph_sim.core.density_grid import compute_density_grid


def test_density_grid_shape() -> None:
    """
    DE:
    Test A: Ausgabe-Shape ist (N,).

    Warum ist das wichtig?
    - Dichte ist "pro Partikel" ein Skalarwert.
    - Daher erwarten wir ein 1D-Array mit genau N Einträgen.

    EN:
    Test A: output shape is (N,).

    Why is this important?
    - Density is one scalar value "per particle".
    - Therefore we expect a 1D array with exactly N entries.
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n

    rho = compute_density_grid(particles=particles, h=0.6)

    assert isinstance(rho, np.ndarray)
    assert rho.shape == (N,)


def test_density_grid_positive() -> None:
    """
    DE:
    Test B: Alle Dichten sind > 0.

    Warum erwarten wir das?
    - Die Masse m[j] ist positiv.
    - Der Poly6-Kernel ist für r=0 positiv.
    - Jedes Partikel zählt sich selbst als Nachbar mit r_ii = 0.
    - Daher bekommt jedes Partikel mindestens einen positiven Beitrag.

    EN:
    Test B: all densities are > 0.

    Why do we expect this?
    - Mass m[j] is positive.
    - The Poly6 kernel is positive at r=0.
    - Each particle counts itself as a neighbor with r_ii = 0.
    - Therefore each particle gets at least one positive contribution.
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)

    rho = compute_density_grid(particles=particles, h=0.6)

    assert np.all(rho > 0.0)


def test_density_grid_matches_naive() -> None:
    """
    DE:
    Test C: Grid-Dichte entspricht der naiven Dichte (gleiche Physik).

    Idee:
    - Wir berechnen einmal `compute_density_naive` (Referenz, aber langsam).
    - Wir berechnen einmal `compute_density_grid` (schneller durch Kandidaten).
    - Wir erwarten numerisch gleiche Ergebnisse (bis auf winzige Rundungsfehler).

    Wichtiger Didaktikpunkt:
    - Das Grid ist keine Physik.
    - Es reduziert nur, welche Paare wir überhaupt prüfen.
    - Weil der Kernel für r > h sowieso 0 ist, darf man diese Paare überspringen
      und bekommt trotzdem das gleiche Ergebnis.

    EN:
    Test C: grid density matches naive density (same physics).

    Idea:
    - Compute `compute_density_naive` once (reference, but slow).
    - Compute `compute_density_grid` once (faster due to candidates).
    - We expect numerically equal results (up to tiny rounding differences).

    Key teaching point:
    - The grid is not physics.
    - It only reduces which pairs we even check.
    - Because the kernel is 0 for r > h anyway, we may skip those pairs
      and still get the same result.
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)

    h = 0.6

    rho_naive = compute_density_naive(particles=particles, h=h)
    rho_grid = compute_density_grid(particles=particles, h=h)

    assert np.allclose(rho_naive, rho_grid, rtol=1e-12)


def test_density_grid_invalid_h() -> None:
    """
    DE:
    Test D: Ungültige Parameter → ValueError.

    Warum?
    - h ist eine Länge (Einflussradius) und muss > 0 sein.
    - h <= 0 ist physikalisch unsinnig und kann zu mathematischen Problemen führen.

    EN:
    Test D: invalid parameters → ValueError.

    Why?
    - h is a length (influence radius) and must be > 0.
    - h <= 0 is physically meaningless and can cause mathematical issues.
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)

    with pytest.raises(ValueError):
        compute_density_grid(particles=particles, h=0.0)

    with pytest.raises(ValueError):
        compute_density_grid(particles=particles, h=-1.0)


