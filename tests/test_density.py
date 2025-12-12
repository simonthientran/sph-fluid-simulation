"""
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
"""

from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# src/-Layout: Warum müssen wir `src/` in `sys.path` einfügen?
# -----------------------------------------------------------------------------
# Dieses Projekt nutzt eine "src-Struktur": der Python-Code liegt in `src/`.
# Wenn wir Tests starten, kennt Python diesen Ordner nicht automatisch als Import-Pfad.
#
# Darum fügen wir `src/` hier einmalig zum Suchpfad hinzu, damit `import sph_sim...` klappt.
# Später kann man das professionell über Packaging lösen (z.B. `pip install -e .`),
# dann braucht man diesen sys.path-Block nicht mehr.
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
    Test A: Rückgabeform (Shape).

    Erwartung:
    - `compute_density_naive` soll für N Partikel genau N Dichtewerte zurückgeben.
    - Das Ergebnis ist also ein 1D-Array der Länge N: shape == (N,)
    """

    # Kleines Partikelset, damit der Test schnell bleibt.
    particles = initialize_particles_cube(
        L=1.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    # smoothing length (Einflussradius) muss > 0 sein.
    h = 1.0

    rho = compute_density_naive(particles=particles, h=h)

    # Das Ergebnis muss ein NumPy-Array sein.
    assert isinstance(rho, np.ndarray)

    # Es muss 1D sein.
    assert rho.ndim == 1

    # Und genau Länge N haben.
    N = particles.n
    assert rho.shape == (N,)


def test_density_naive_positive() -> None:
    """
    Test B: Dichte ist positiv.

    Warum erwarten wir rho > 0?
    - Wir addieren Beiträge: rho[i] += m[j] * W(r_ij, h)
    - m[j] ist positiv (Masse)
    - W(...) ist beim Poly6-Kernel innerhalb des Supports nicht-negativ und bei r=0 sogar > 0
    - Selbst wenn alle anderen Partikel weit weg wären, gibt es immer j=i mit r=0
      (das "Selbst"-Partikel), und dadurch einen positiven Beitrag.
    """

    particles = initialize_particles_cube(
        L=1.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    h = 1.0
    rho = compute_density_naive(particles=particles, h=h)

    # Alle Dichtewerte müssen > 0 sein.
    assert np.all(rho > 0.0)


def test_density_naive_symmetry() -> None:
    """
    Test C: Symmetrie (innere Partikel).

    Idee:
    - Wir bauen ein symmetrisches, gleichmäßiges Gitter.
    - Bei einem solchen Gitter haben "innere" Partikel (nicht am Rand) die gleiche
      Nachbarschaftsstruktur: gleich viele Nachbarn in gleichen Abständen.
    - Deshalb sollten ihre Dichten sehr ähnlich (im Idealfall identisch) sein.

    Wichtig:
    - Wir testen hier bewusst "grob" mit `np.allclose`, weil Fließkomma-Rechnungen
      kleine Rundungsunterschiede haben können.
    """

    # L=2.0, dx=0.5 → Koordinaten: 0.0, 0.5, 1.0, 1.5 → 4x4 = 16 Partikel
    # In diesem 4x4-Gitter gibt es 4 innere Partikel:
    # (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)
    particles = initialize_particles_cube(
        L=2.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    # h so wählen, dass Nachbarn in der Nähe sicher beitragen (inkl. diagonale Nachbarn).
    # (Das ist eine plausible Wahl, keine Optimierung.)
    h = 1.0

    rho = compute_density_naive(particles=particles, h=h)

    # Hilfsfunktion: Index eines Partikels mit genau dieser Position finden.
    # Bei diesem Gitter sind die Positionen exakte Vielfache von 0.5 (float64),
    # daher können wir hier mit == arbeiten.
    def idx_for_position(x_target: float, y_target: float) -> int:
        mask = (particles.x == x_target) & (particles.y == y_target)
        indices = np.where(mask)[0]

        # Wir erwarten genau ein Partikel pro Position.
        assert indices.size == 1
        return int(indices[0])

    inner_indices = [
        idx_for_position(0.5, 0.5),
        idx_for_position(0.5, 1.0),
        idx_for_position(1.0, 0.5),
        idx_for_position(1.0, 1.0),
    ]

    inner_rho = rho[inner_indices]

    # Die Dichten der inneren Partikel sollten sehr ähnlich sein.
    # Toleranz bewusst "grob", damit der Test robust bleibt.
    assert np.allclose(inner_rho, inner_rho.mean(), atol=1e-10, rtol=0.0)


def test_density_naive_invalid_h() -> None:
    """
    Test D: h-Validierung.

    Warum?
    - h ist eine Länge (Einflussradius) und muss > 0 sein.
    - Für h <= 0 kann der Kernel nicht sinnvoll berechnet werden.
    """

    particles = initialize_particles_cube(
        L=1.0,
        dx=0.5,
        rho0=1000.0,
        mass_per_particle=0.01,
    )

    # h = 0 → muss ValueError werfen
    with pytest.raises(ValueError):
        compute_density_naive(particles=particles, h=0.0)

    # h < 0 → muss auch ValueError werfen
    with pytest.raises(ValueError):
        compute_density_naive(particles=particles, h=-1.0)


