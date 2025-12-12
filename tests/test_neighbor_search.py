"""
DE:
Unit-Tests für `sph_sim.core.neighbor_search`.

Worum geht es?
-------------
Wir testen hier eine sehr einfache Nachbarsuche über ein Uniform-Grid:
- `cell_index`: ordnet eine Position (x, y) einer Grid-Zelle (ix, iy) zu.
- `build_uniform_grid`: baut ein Dictionary "Zelle -> Partikel-Indizes".
- `query_neighbor_candidates`: liefert Kandidaten-Indizes aus den 3×3 Zellen um ein Partikel.

Wichtig:
--------
Die Kandidatenliste ist nicht gleich "echte Nachbarn".
Echte Nachbarn bestimmen wir später durch eine Distanzprüfung (z.B. r <= h).

EN:
Unit tests for `sph_sim.core.neighbor_search`.

What is this about?
-------------------
We test a very simple neighbor search using a uniform grid:
- `cell_index`: maps a position (x, y) to a grid cell (ix, iy).
- `build_uniform_grid`: builds a dictionary "cell -> particle indices".
- `query_neighbor_candidates`: returns candidate indices from the 3×3 cells around a particle.

Important:
----------
The candidate list is not the same as "true neighbors".
True neighbors are determined later by a distance check (e.g., r <= h).
"""

from pathlib import Path
import sys

# --- Deutsch ---
# -----------------------------------------------------------------------------
# src/-Layout: Warum müssen wir `src/` in `sys.path` einfügen?
# -----------------------------------------------------------------------------
# In diesem Projekt liegen die importierbaren Python-Module unter `src/`.
# Wenn wir `pytest` aus dem Projekt-Root starten, kennt Python `src/` nicht automatisch
# als Import-Pfad.
#
# Darum fügen wir `src/` hier einmalig in `sys.path` ein, damit `import sph_sim...` klappt.
# Professionell löst man das später über Packaging (z.B. `pip install -e .`), dann braucht
# man diesen Block nicht mehr.
#
# --- English ---
# -----------------------------------------------------------------------------
# src layout: why do we need to add `src/` to `sys.path`?
# -----------------------------------------------------------------------------
# In this project, the importable Python modules live under `src/`.
# When we run `pytest` from the project root, Python does not automatically know `src/`
# as an import path.
#
# Therefore we add `src/` once to `sys.path` so that `import sph_sim...` works.
# Professionally, you solve this later via packaging (e.g., `pip install -e .`), and then
# you no longer need this block.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest

from sph_sim.core.particles import initialize_particles_cube
from sph_sim.core.neighbor_search import cell_index, build_uniform_grid, query_neighbor_candidates


def test_cell_index_basic() -> None:
    """
    DE:
    Test A: `cell_index` ordnet Punkte den richtigen Zellen zu.

    Setup:
    - cell_size = 1.0 bedeutet: jede Zelle ist ein Quadrat der Kantenlänge 1.0
    - Dann gilt:
      - x in [0, 1)  -> ix = 0
      - x in [1, 2)  -> ix = 1
      - y in [2, 3)  -> iy = 2

    EN:
    Test A: `cell_index` assigns points to the correct cells.

    Setup:
    - cell_size = 1.0 means: each cell is a square with edge length 1.0
    - Then:
      - x in [0, 1)  -> ix = 0
      - x in [1, 2)  -> ix = 1
      - y in [2, 3)  -> iy = 2
    """

    cell_size = 1.0

    # --- Deutsch ---
    # Punkt (0.2, 0.2) liegt in der ersten Zelle (0, 0).
    #
    # --- English ---
    # Point (0.2, 0.2) lies in the first cell (0, 0).
    assert cell_index(x=0.2, y=0.2, cell_size=cell_size) == (0, 0)

    # --- Deutsch ---
    # Punkt (1.2, 0.2) liegt eine Zelle weiter rechts: (1, 0).
    #
    # --- English ---
    # Point (1.2, 0.2) lies one cell to the right: (1, 0).
    assert cell_index(x=1.2, y=0.2, cell_size=cell_size) == (1, 0)

    # --- Deutsch ---
    # Punkt (0.2, 2.7) liegt in y-Richtung in der dritten Zelle: iy = 2.
    #
    # --- English ---
    # Point (0.2, 2.7) lies in the third cell along y: iy = 2.
    assert cell_index(x=0.2, y=2.7, cell_size=cell_size) == (0, 2)


def test_build_uniform_grid_contains_all_particles_once() -> None:
    """
    DE:
    Test B: `build_uniform_grid` enthält alle Partikel genau einmal.

    Idee:
    - Wir erzeugen ein kleines Partikelset (4 Partikel).
    - Wir bauen das Grid.
    - Wir sammeln alle Indizes aus allen Zellen.
    - Dann prüfen wir: genau die Indizes 0..N-1 sind enthalten, und zwar ohne Duplikate.

    Warum ist das wichtig?
    - Das Grid ist eine Datenstruktur für Nachbarsuche.
    - Wenn ein Index fehlt: Partikel werden nie als Nachbarn gefunden.
    - Wenn ein Index doppelt vorkommt: manche Beiträge würden später doppelt gezählt.

    EN:
    Test B: `build_uniform_grid` contains all particles exactly once.

    Idea:
    - We create a small particle set (4 particles).
    - We build the grid.
    - We collect all indices from all cells.
    - Then we check: exactly the indices 0..N-1 are present, without duplicates.

    Why is this important?
    - The grid is a data structure for neighbor search.
    - If an index is missing: particles will never be found as neighbors.
    - If an index appears twice: some contributions would later be counted twice.
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    N = particles.n

    # --- Deutsch ---
    # cell_size kann 0.5 oder 1.0 sein; beides ist für diesen Test okay.
    # Wir nehmen 0.5, weil das ungefähr "eine Zelle pro Gitterabstand" ist.
    #
    # --- English ---
    # cell_size can be 0.5 or 1.0; both are fine for this test.
    # We choose 0.5 because it is roughly "one cell per grid spacing".
    grid = build_uniform_grid(particles=particles, cell_size=0.5)

    # --- Deutsch ---
    # Alle Indizes aus dem Dict zusammensammeln:
    # - grid.values() sind die Listen der Indizes pro Zelle
    # - wir hängen alles in eine gemeinsame Liste `all_indices`
    #
    # --- English ---
    # Collect all indices from the dict:
    # - grid.values() are the lists of indices per cell
    # - we append everything into one combined list `all_indices`
    all_indices: list[int] = []
    for indices_in_cell in grid.values():
        all_indices.extend(indices_in_cell)

    # --- Deutsch ---
    # 1) Anzahl muss genau N sein, wenn jeder Index genau einmal vorkommt.
    #
    # --- English ---
    # 1) The count must be exactly N if each index appears exactly once.
    assert len(all_indices) == N

    # --- Deutsch ---
    # 2) Sortiert müssen es genau [0, 1, ..., N-1] sein.
    #
    # --- English ---
    # 2) When sorted, it must be exactly [0, 1, ..., N-1].
    assert sorted(all_indices) == list(range(N))


def test_query_neighbor_candidates_returns_self_and_nearby() -> None:
    """
    DE:
    Test C: `query_neighbor_candidates` liefert eine sinnvolle Kandidatenliste.

    Setup:
    - Wir erzeugen wieder 4 Partikel in einem 2×2 Gitter.
    - cell_size = 0.5 bedeutet: ein Grid-Schritt entspricht einer Zellgröße.
    - Wir fragen Kandidaten für i=0 ab.

    Erwartungen (sehr basic):
    - Die Rückgabe ist eine Liste aus ints.
    - Die Liste ist nicht leer.
    - Der Index i selbst ist enthalten (self-candidate ist erlaubt).

    EN:
    Test C: `query_neighbor_candidates` returns a reasonable candidate list.

    Setup:
    - We create again 4 particles in a 2×2 grid.
    - cell_size = 0.5 means: one grid step corresponds to one cell size.
    - We query candidates for i=0.

    Expectations (very basic):
    - The return value is a list of ints.
    - The list is not empty.
    - The index i itself is included (self-candidate is allowed).
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    grid = build_uniform_grid(particles=particles, cell_size=0.5)

    cands = query_neighbor_candidates(i=0, particles=particles, grid=grid, cell_size=0.5)

    # --- Deutsch ---
    # Rückgabetyp: Liste
    #
    # --- English ---
    # Return type: list
    assert isinstance(cands, list)

    # --- Deutsch ---
    # Kandidatenliste darf nicht leer sein (mindestens i selbst sollte drin sein).
    #
    # --- English ---
    # Candidate list must not be empty (at least i itself should be included).
    assert len(cands) > 0

    # --- Deutsch ---
    # Alle Elemente müssen ints sein, weil es Partikel-Indizes sind.
    #
    # --- English ---
    # All elements must be ints because they are particle indices.
    assert all(isinstance(j, int) for j in cands)

    # --- Deutsch ---
    # i selbst darf enthalten sein (später filtern wir ggf. i==j weg oder behandeln es extra).
    #
    # --- English ---
    # i itself may be included (later we may filter i==j or treat it separately).
    assert 0 in cands


def test_query_neighbor_candidates_validates_index() -> None:
    """
    DE:
    Test D: `query_neighbor_candidates` prüft den Index i.

    Wir erwarten ValueError für:
    - i < 0
    - i >= N

    EN:
    Test D: `query_neighbor_candidates` validates the index i.

    We expect ValueError for:
    - i < 0
    - i >= N
    """

    particles = initialize_particles_cube(L=1.0, dx=0.5, rho0=1000.0, mass_per_particle=0.01)
    grid = build_uniform_grid(particles=particles, cell_size=0.5)
    N = particles.n

    # --- Deutsch ---
    # i = -1 ist ungültig -> ValueError
    #
    # --- English ---
    # i = -1 is invalid -> ValueError
    with pytest.raises(ValueError):
        query_neighbor_candidates(i=-1, particles=particles, grid=grid, cell_size=0.5)

    # --- Deutsch ---
    # i = N ist ungültig, weil gültige Indizes nur bis N-1 gehen -> ValueError
    #
    # --- English ---
    # i = N is invalid because valid indices go only up to N-1 -> ValueError
    with pytest.raises(ValueError):
        query_neighbor_candidates(i=N, particles=particles, grid=grid, cell_size=0.5)


