"""
DE:
Unit-Tests für `sph_sim.core.density.compute_density_naive`.

Wichtig für Anfänger:
Diese Tests prüfen "Plausibilität" und grundlegendes Verhalten – nicht die perfekte Physik.
Warum?
- SPH-Formeln haben oft Parameter/Varianten, und kleine Details können Zahlen verändern.
- Für Unit-Tests ist es aber sehr wertvoll, einfache, robuste Eigenschaften zu testen:
  - richtige Form (Shape)
  - Dichte ist positiv (bei positivem Kernel und positiver Masse)
  - Symmetrie in einem symmetrischen Aufbau
  - saubere Fehler bei ungültigen Eingaben

EN:
Unit tests for `sph_sim.core.density.compute_density_naive`.

Important for beginners:
These tests check "plausibility" and basic behavior — not perfect physics.
Why?
- SPH formulas often have parameters/variants, and small details can change numbers.
- For unit tests, it is very valuable to test simple, robust properties:
  - correct shape
  - density is positive (with positive kernel and positive mass)
  - symmetry in a symmetric setup
  - clean errors for invalid inputs
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

from sph_sim.core.particles import initialize_particles_cube
from sph_sim.core.density import compute_density_naive


def test_density_naive_shape() -> None:
    """
    DE:
    Test A: Rückgabeform (Shape).

    Erwartung:
    - `compute_density_naive` soll für N Partikel genau N Dichtewerte zurückgeben.
    - Das Ergebnis ist also ein 1D-Array der Länge N: shape == (N,)

    EN:
    Test A: return shape.

    Expectation:
    - `compute_density_naive` should return exactly N density values for N particles.
    - The result is therefore a 1D array of length N: shape == (N,)
    """

    # --- Deutsch ---
    # Kleines Partikelset, damit der Test schnell bleibt.
    #
    # --- English ---
    # Small particle set so the test stays fast.
    particles = initialize_particles_cube(
        L=1.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    # --- Deutsch ---
    # smoothing length (Einflussradius) muss > 0 sein.
    #
    # --- English ---
    # smoothing length (influence radius) must be > 0.
    h = 1.0

    rho = compute_density_naive(particles=particles, h=h)

    # --- Deutsch ---
    # Das Ergebnis muss ein NumPy-Array sein.
    #
    # --- English ---
    # The result must be a NumPy array.
    assert isinstance(rho, np.ndarray)

    # --- Deutsch ---
    # Es muss 1D sein.
    #
    # --- English ---
    # It must be 1D.
    assert rho.ndim == 1

    # --- Deutsch ---
    # Und genau Länge N haben.
    #
    # --- English ---
    # And it must have exactly length N.
    N = particles.n
    assert rho.shape == (N,)


def test_density_naive_positive() -> None:
    """
    DE:
    Test B: Dichte ist positiv.

    Warum erwarten wir rho > 0?
    - Wir addieren Beiträge: rho[i] += m[j] * W(r_ij, h)
    - m[j] ist positiv (Masse)
    - W(...) ist beim Poly6-Kernel innerhalb des Supports nicht-negativ und bei r=0 sogar > 0
    - Selbst wenn alle anderen Partikel weit weg wären, gibt es immer j=i mit r=0
      (das "Selbst"-Partikel), und dadurch einen positiven Beitrag.

    EN:
    Test B: density is positive.

    Why do we expect rho > 0?
    - We add contributions: rho[i] += m[j] * W(r_ij, h)
    - m[j] is positive (mass)
    - For the Poly6 kernel, W(...) is non-negative inside the support and at r=0 even > 0
    - Even if all other particles were far away, there is always j=i with r=0
      (the "self" particle), and thus a positive contribution.
    """

    particles = initialize_particles_cube(
        L=1.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    h = 1.0
    rho = compute_density_naive(particles=particles, h=h)

    # --- Deutsch ---
    # Alle Dichtewerte müssen > 0 sein.
    #
    # --- English ---
    # All density values must be > 0.
    assert np.all(rho > 0.0)


def test_density_naive_symmetry() -> None:
    """
    DE:
    Test C: Symmetrie (innere Partikel).

    Idee:
    - Wir bauen ein symmetrisches, gleichmäßiges Gitter.
    - Bei einem solchen Gitter haben "innere" Partikel (nicht am Rand) die gleiche
      Nachbarschaftsstruktur: gleich viele Nachbarn in gleichen Abständen.
    - Deshalb sollten ihre Dichten sehr ähnlich (im Idealfall identisch) sein.

    Wichtig:
    - Wir testen hier bewusst "grob" mit `np.allclose`, weil Fließkomma-Rechnungen
      kleine Rundungsunterschiede haben können.

    EN:
    Test C: symmetry (inner particles).

    Idea:
    - We build a symmetric, uniform grid.
    - In such a grid, "inner" particles (not on the boundary) have the same
      neighborhood structure: the same number of neighbors at the same distances.
    - Therefore, their densities should be very similar (ideally identical).

    Important:
    - We intentionally test "roughly" with `np.allclose` because floating-point computations
      can have small rounding differences.
    """

    # --- Deutsch ---
    # L=2.0, dx=0.5 → Koordinaten: 0.0, 0.5, 1.0, 1.5 → 4x4 = 16 Partikel
    # In diesem 4x4-Gitter gibt es 4 innere Partikel:
    # (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)
    #
    # --- English ---
    # L=2.0, dx=0.5 → coordinates: 0.0, 0.5, 1.0, 1.5 → 4x4 = 16 particles
    # In this 4x4 grid there are 4 inner particles:
    # (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)
    particles = initialize_particles_cube(
        L=2.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    # --- Deutsch ---
    # h so wählen, dass Nachbarn in der Nähe sicher beitragen (inkl. diagonale Nachbarn).
    # (Das ist eine plausible Wahl, keine Optimierung.)
    #
    # --- English ---
    # Choose h so that nearby neighbors contribute reliably (including diagonal neighbors).
    # (This is a plausible choice, not an optimization.)
    h = 1.0

    rho = compute_density_naive(particles=particles, h=h)

    # --- Deutsch ---
    # Hilfsfunktion: Index eines Partikels mit genau dieser Position finden.
    # Bei diesem Gitter sind die Positionen exakte Vielfache von 0.5 (float64),
    # daher können wir hier mit == arbeiten.
    #
    # --- English ---
    # Helper function: find the index of a particle at exactly this position.
    # In this grid, the positions are exact multiples of 0.5 (float64),
    # therefore we can use == here.
    def idx_for_position(x_target: float, y_target: float) -> int:
        mask = (particles.x == x_target) & (particles.y == y_target)
        indices = np.where(mask)[0]

        # --- Deutsch ---
        # Wir erwarten genau ein Partikel pro Position.
        #
        # --- English ---
        # We expect exactly one particle per position.
        assert indices.size == 1
        return int(indices[0])

    inner_indices = [
        idx_for_position(0.5, 0.5),
        idx_for_position(0.5, 1.0),
        idx_for_position(1.0, 0.5),
        idx_for_position(1.0, 1.0),
    ]

    inner_rho = rho[inner_indices]

    # --- Deutsch ---
    # Die Dichten der inneren Partikel sollten sehr ähnlich sein.
    # Toleranz bewusst "grob", damit der Test robust bleibt.
    #
    # --- English ---
    # The densities of the inner particles should be very similar.
    # The tolerance is intentionally "coarse" so the test remains robust.
    assert np.allclose(inner_rho, inner_rho.mean(), atol=1e-10, rtol=0.0)


def test_density_naive_invalid_h() -> None:
    """
    DE:
    Test D: h-Validierung.

    Warum?
    - h ist eine Länge (Einflussradius) und muss > 0 sein.
    - Für h <= 0 kann der Kernel nicht sinnvoll berechnet werden.

    EN:
    Test D: h validation.

    Why?
    - h is a length (influence radius) and must be > 0.
    - For h <= 0, the kernel cannot be computed meaningfully.
    """

    particles = initialize_particles_cube(
        L=1.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    # --- Deutsch ---
    # h = 0 → muss ValueError werfen
    #
    # --- English ---
    # h = 0 → must raise ValueError
    with pytest.raises(ValueError):
        compute_density_naive(particles=particles, h=0.0)

    # --- Deutsch ---
    # h < 0 → muss auch ValueError werfen
    #
    # --- English ---
    # h < 0 → must also raise ValueError
    with pytest.raises(ValueError):
        compute_density_naive(particles=particles, h=-1.0)


