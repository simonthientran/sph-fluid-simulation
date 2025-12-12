"""
DE:
Unit-Tests für `sph_sim.core.particles`.

Wichtig für Anfänger:
Unit-Tests sind kleine, automatische Prüfungen, die sicherstellen sollen, dass Code so
funktioniert, wie wir es erwarten. Wenn wir später etwas ändern, helfen Tests dabei,
Fehler früh zu entdecken.

EN:
Unit tests for `sph_sim.core.particles`.

Important for beginners:
Unit tests are small, automated checks that are intended to ensure that code
works as we expect. If we change something later, tests help
to detect errors early.
"""

from pathlib import Path
import sys

# --- Deutsch ---
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
#
# --- English ---
# -----------------------------------------------------------------------------
# src layout: why do we need to add `src/` to `sys.path`?
# -----------------------------------------------------------------------------
# In this project, the importable Python modules live under `src/`.
# Python does not automatically know this folder as a "search path for imports".
#
# Without this block, for example `import sph_sim...` would fail because Python cannot find
# the `sph_sim` package.
#
# This is fine for small learning projects. Professionally, you solve this cleanly later via
# packaging (e.g., `pyproject.toml` + `pip install -e .`), so no sys.path hacks
# are needed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    # --- Deutsch ---
    # `insert(0, ...)` setzt den Pfad ganz vorne, damit Python zuerst dort sucht.
    #
    # --- English ---
    # `insert(0, ...)` puts the path at the very front so Python searches there first.
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest

from sph_sim.core.particles import ParticleSet2D, initialize_particles_cube


def test_initialize_particles_cube_shapes_and_count() -> None:
    """
    DE:
    Test A: Anzahl Partikel & Shapes.

    Wir testen hier zwei Dinge:
    1) Die Funktion gibt wirklich ein `ParticleSet2D` zurück.
    2) Alle Felder sind 1D-Arrays und haben die erwartete Länge N.

    EN:
    Test A: particle count & shapes.

    We test two things here:
    1) The function really returns a `ParticleSet2D`.
    2) All fields are 1D arrays and have the expected length N.
    """

    # --- Deutsch ---
    # Test-Parameter (bewusst kleine Zahlen, damit man es im Kopf nachvollziehen kann)
    #
    # --- English ---
    # Test parameters (deliberately small numbers so you can reason about them mentally)
    L = 1.0
    dx = 0.5
    rho0 = 1000.0
    mass_per_particle = 0.01

    # --- Deutsch ---
    # Erklärung:
    # Bei dx = 0.5 liegen die Koordinaten pro Achse bei 0.0 und 0.5.
    # Das sind 2 Positionen pro Achse → 2 * 2 = 4 Partikel insgesamt.
    #
    # --- English ---
    # Explanation:
    # With dx = 0.5, the coordinates per axis are 0.0 and 0.5.
    # That is 2 positions per axis → 2 * 2 = 4 particles in total.
    particles = initialize_particles_cube(
        L=L,
        dx=dx,
        rho0=rho0,
        mass_per_particle=mass_per_particle,
    )

    # --- Deutsch ---
    # 1) Rückgabetyp prüfen
    #
    # --- English ---
    # 1) Check return type
    assert isinstance(particles, ParticleSet2D)

    # --- Deutsch ---
    # 2) Anzahl Partikel (N) prüfen
    #
    # --- English ---
    # 2) Check number of particles (N)
    N = len(particles.x)
    assert N == 4

    # --- Deutsch ---
    # 3) Für SoA ist wichtig: Alle Felder sind 1D und haben exakt Länge N.
    #    Wir prüfen jedes Feld einzeln, damit der Test für Anfänger sehr klar ist.
    #
    # --- English ---
    # 3) For SoA it is important: all fields are 1D and have exactly length N.
    #    We check each field individually so the test is very clear for beginners.
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
    DE:
    Test B: Initialwerte korrekt.

    Ein typischer Startzustand in SPH ist ein ruhendes Fluid:
    - Geschwindigkeit am Anfang 0 (damit keine “versteckte” Bewegung drin ist)
    - Druck am Anfang 0 als einfache Referenz (je nach Modell könnte man später auch
      anders starten, aber hier ist 0 eine klare, einfache Wahl)

    EN:
    Test B: initial values are correct.

    A typical initial condition in SPH is a fluid at rest:
    - Initial velocity is 0 (so there is no “hidden” motion)
    - Initial pressure is 0 as a simple reference (depending on the model you could
      start differently later, but here 0 is a clear, simple choice)
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

    # --- Deutsch ---
    # vx und vy: überall 0.0 (Fluid ruht am Anfang)
    #
    # --- English ---
    # vx and vy: 0.0 everywhere (the fluid is at rest initially)
    assert np.all(particles.vx == 0.0)
    assert np.all(particles.vy == 0.0)

    # --- Deutsch ---
    # rho: überall rho0 (Startdichte)
    #
    # --- English ---
    # rho: rho0 everywhere (initial density)
    assert np.all(particles.rho == rho0)

    # --- Deutsch ---
    # p: überall 0.0 (Startdruck)
    #
    # --- English ---
    # p: 0.0 everywhere (initial pressure)
    assert np.all(particles.p == 0.0)

    # --- Deutsch ---
    # m: überall mass_per_particle (konstante Masse pro Partikel)
    #
    # --- English ---
    # m: mass_per_particle everywhere (constant mass per particle)
    assert np.all(particles.m == mass_per_particle)


def test_initialize_particles_cube_positions_in_range() -> None:
    """
    DE:
    Test C: Positionen plausibel im Bereich.

    Wir erwarten Partikel in einem Quadratbereich:
    - x in [0, L)
    - y in [0, L)

    Warum [0, L) und nicht [0, L]?
    - Das ist ein "halboffenes Intervall": 0 ist drin, L ist nicht drin.
    - So entstehen keine Partikel genau auf der oberen/rechten Kante bei L.

    EN:
    Test C: positions are plausible and within the expected range.

    We expect particles in a square domain:
    - x in [0, L)
    - y in [0, L)

    Why [0, L) and not [0, L]?
    - This is a "half-open interval": 0 is included, L is not included.
    - This prevents particles exactly on the upper/right edge at L.
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

    # --- Deutsch ---
    # Untergrenze: Alle Koordinaten müssen >= 0 sein.
    #
    # --- English ---
    # Lower bound: all coordinates must be >= 0.
    assert np.all(particles.x >= 0.0)
    assert np.all(particles.y >= 0.0)

    # --- Deutsch ---
    # Obergrenze: Alle Koordinaten sollen < L sein.
    # Wir erlauben eine sehr kleine Toleranz, um mögliche Rundungsartefakte abzufangen.
    #
    # --- English ---
    # Upper bound: all coordinates should be < L.
    # We allow a very small tolerance to absorb possible rounding artifacts.
    tol = 1e-12
    assert np.all(particles.x < L + tol)
    assert np.all(particles.y < L + tol)


def test_initialize_particles_cube_invalid_parameters_raise() -> None:
    """
    DE:
    Test D: Ungültige Parameter → ValueError.

    Warum ist Input-Validierung wichtig?
    - Sie sorgt für frühe, klare Fehler (statt später “irgendwo” in der Simulation).
    - Das spart Debugging-Zeit und macht den Code robuster.

    Wir testen hier bewusst mehrere einzelne Fälle.

    EN:
    Test D: invalid parameters → ValueError.

    Why is input validation important?
    - It provides early, clear errors (instead of failing later “somewhere” in the simulation).
    - This saves debugging time and makes the code more robust.

    We intentionally test multiple individual cases here.
    """

    rho0 = 1000.0
    mass_per_particle = 0.01

    # --- Deutsch ---
    # L <= 0 ist physikalisch/technisch unsinnig (kein Bereich).
    #
    # --- English ---
    # L <= 0 is physically/technically meaningless (no domain).
    with pytest.raises(ValueError):
        initialize_particles_cube(L=0.0, dx=0.5, rho0=rho0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=-1.0, dx=0.5, rho0=rho0, mass_per_particle=mass_per_particle)

    # --- Deutsch ---
    # dx <= 0 ist unsinnig (kein sinnvoller Abstand).
    #
    # --- English ---
    # dx <= 0 is meaningless (no valid spacing).
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.0, rho0=rho0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=-0.1, rho0=rho0, mass_per_particle=mass_per_particle)

    # --- Deutsch ---
    # rho0 <= 0: Dichte muss positiv sein.
    #
    # --- English ---
    # rho0 <= 0: density must be positive.
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=0.0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=-10.0, mass_per_particle=mass_per_particle)

    # --- Deutsch ---
    # mass_per_particle <= 0: Masse muss positiv sein.
    #
    # --- English ---
    # mass_per_particle <= 0: mass must be positive.
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=rho0, mass_per_particle=0.0)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=0.5, rho0=rho0, mass_per_particle=-1.0)

    # --- Deutsch ---
    # dx >= L: Dann gibt es pro Achse höchstens 1 Punkt → zu wenig/keine Partikel.
    #
    # --- English ---
    # dx >= L: then there is at most 1 point per axis → too few/no particles.
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=1.0, rho0=rho0, mass_per_particle=mass_per_particle)
    with pytest.raises(ValueError):
        initialize_particles_cube(L=1.0, dx=2.0, rho0=rho0, mass_per_particle=mass_per_particle)


