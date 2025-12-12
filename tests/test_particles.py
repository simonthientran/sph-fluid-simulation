"""
Unit-Tests für `sph_sim.core.particles`.

Wichtig für Anfänger:
Unit-Tests sind kleine, automatische Prüfungen, die sicherstellen sollen, dass Code so
funktioniert, wie wir es erwarten. Wenn wir später etwas ändern, helfen Tests dabei,
Fehler früh zu entdecken.
"""

from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# src/-Layout: Warum müssen wir `src/` in `sys.path` einfügen?
# -----------------------------------------------------------------------------
# In diesem Projekt liegen die importierbaren Python-Module unter `src/`.
# Python kennt diesen Ordner aber nicht automatisch als "Suchpfad für Imports".
#
# Ohne diesen Block würde z.B. `import sph_sim...` fehlschlagen, weil Python das Paket
# `sph_sim` nicht findet.
#
# Das ist für kleine Lernprojekte okay. Professionell löst man das später sauber durch
# Packaging (z.B. `pyproject.toml` + `pip install -e .`), sodass keine sys.path-Hacks
# nötig sind.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    # `insert(0, ...)` setzt den Pfad ganz vorne, damit Python zuerst dort sucht.
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest

from sph_sim.core.particles import ParticleSet2D, initialize_particles_cube


def test_initialize_particles_cube_shapes_and_count() -> None:
    """
    Test A: Anzahl Partikel & Shapes.

    Wir testen hier zwei Dinge:
    1) Die Funktion gibt wirklich ein `ParticleSet2D` zurück.
    2) Alle Felder sind 1D-Arrays und haben die erwartete Länge N.
    """

    # Test-Parameter (bewusst kleine Zahlen, damit man es im Kopf nachvollziehen kann)
    L = 1.0
    dx = 0.5
    rho0 = 1000.0
    mass_per_particle = 0.01

    # Erklärung:
    # Bei dx = 0.5 liegen die Koordinaten pro Achse bei 0.0 und 0.5.
    # Das sind 2 Positionen pro Achse → 2 * 2 = 4 Partikel insgesamt.
    particles = initialize_particles_cube(
        L=L,
        dx=dx,
        rho0=rho0,
        mass_per_particle=mass_per_particle,
    )

    # 1) Rückgabetyp prüfen
    assert isinstance(particles, ParticleSet2D)

    # 2) Anzahl Partikel (N) prüfen
    N = len(particles.x)
    assert N == 4

    # 3) Für SoA ist wichtig: Alle Felder sind 1D und haben exakt Länge N.
    #    Wir prüfen jedes Feld einzeln, damit der Test für Anfänger sehr klar ist.
    assert particles.x.ndim == 1
    assert len(particles.x) == N

    assert particles.y.ndim == 1
    assert len(particles.y) == N

    assert particles.vx.ndim == 1
    assert len(particles.vx) == N

    assert particles.vy.ndim == 1
    assert len(particles.vy) == N

    assert particles.rho.ndim == 1
    assert len(particles.rho) == N

    assert particles.p.ndim == 1
    assert len(particles.p) == N

    assert particles.m.ndim == 1
    assert len(particles.m) == N


def test_initialize_particles_cube_initial_values() -> None:
    """
    Test B: Initialwerte korrekt.

    Ein typischer Startzustand in SPH ist ein ruhendes Fluid:
    - Geschwindigkeit am Anfang 0 (damit keine “versteckte” Bewegung drin ist)
    - Druck am Anfang 0 als einfache Referenz (je nach Modell könnte man später auch
      anders starten, aber hier ist 0 eine klare, einfache Wahl)
    """

    L = 1.0
    dx = 0.5
    rho0 = 1000.0
    mass_per_particle = 0.01

    particles = initialize_particles_cube(
        L=L,
        dx=dx,
        rho0=rho0,
        mass_per_particle=mass_per_particle,
    )

    # vx und vy: überall 0.0 (Fluid ruht am Anfang)
    assert np.all(particles.vx == 0.0)
    assert np.all(particles.vy == 0.0)

    # rho: überall rho0 (Startdichte)
    assert np.all(particles.rho == rho0)

    # p: überall 0.0 (Startdruck)
    assert np.all(particles.p == 0.0)

    # m: überall mass_per_particle (konstante Masse pro Partikel)
    assert np.all(particles.m == mass_per_particle)


def test_initialize_particles_cube_positions_in_range() -> None:
    """
    Test C: Positionen plausibel im Bereich.

    Wir erwarten Partikel in einem Quadratbereich:
    - x in [0, L)
    - y in [0, L)

    Warum [0, L) und nicht [0, L]?
    - Das ist ein "halboffenes Intervall": 0 ist drin, L ist nicht drin.
    - So entstehen keine Partikel genau auf der oberen/rechten Kante bei L.
    """

    L = 1.0
    dx = 0.5
    rho0 = 1000.0
    mass_per_particle = 0.01

    particles = initialize_particles_cube(
        L=L,
        dx=dx,
        rho0=rho0,
        mass_per_particle=mass_per_particle,
    )

    # Untergrenze: Alle Koordinaten müssen >= 0 sein.
    assert np.all(particles.x >= 0.0)
    assert np.all(particles.y >= 0.0)

    # Obergrenze: Alle Koordinaten sollen < L sein.
    # Wir erlauben eine sehr kleine Toleranz, um mögliche Rundungsartefakte abzufangen.
    tol = 1e-12
    assert np.all(particles.x < L + tol)
    assert np.all(particles.y < L + tol)


def test_initialize_particles_cube_invalid_parameters_raise() -> None:
    """
    Test D: Ungültige Parameter → ValueError.

    Warum ist Input-Validierung wichtig?
    - Sie sorgt für frühe, klare Fehler (statt später “irgendwo” in der Simulation).
    - Das spart Debugging-Zeit und macht den Code robuster.

    Wir testen hier bewusst mehrere einzelne Fälle.
    """

    rho0 = 1000.0
    mass_per_particle = 0.01

    # L <= 0 ist physikalisch/technisch unsinnig (kein Bereich).
    with pytest.raises(ValueError):
        initialize_particles_cube(L=0.0, dx=0.5, rho0=rho0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=-1.0, dx=0.5, rho0=rho0, mass_per_particle=mass_per_particle)

    # dx <= 0 ist unsinnig (kein sinnvoller Abstand).
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.0, rho0=rho0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=-0.1, rho0=rho0, mass_per_particle=mass_per_particle)

    # rho0 <= 0: Dichte muss positiv sein.
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=0.0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=-10.0, mass_per_particle=mass_per_particle)

    # mass_per_particle <= 0: Masse muss positiv sein.
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=rho0, mass_per_particle=0.0)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=rho0, mass_per_particle=-1.0)

    # dx >= L: Dann gibt es pro Achse höchstens 1 Punkt → zu wenig/keine Partikel.
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=1.0, rho0=rho0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=2.0, rho0=rho0, mass_per_particle=mass_per_particle)


