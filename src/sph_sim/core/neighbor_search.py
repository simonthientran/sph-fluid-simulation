"""
DE:
Nachbarsuche (Neighbor Search) für 2D-Partikel mit einem einfachen Uniform Grid.

Warum brauchen wir Nachbarsuche in SPH?
--------------------------------------
In SPH (Smoothed Particle Hydrodynamics) berechnen wir viele Summen über Nachbarn,
z.B. für Dichte, Druckkräfte, Viskosität usw. Formal steht oft eine Summe über *alle* Partikel j,
aber praktisch zählt meist nur die Umgebung innerhalb eines Radius h (compact support des Kernels).

Wenn wir für jedes Partikel i alle anderen Partikel j prüfen würden, wäre das O(N^2).
Das wird schnell sehr langsam, wenn N größer wird.

Die Idee eines Uniform Grids:
----------------------------
Wir teilen den Raum in gleich große Quadratzellen (cell_size).
Jedes Partikel wird einer Zelle zugeordnet, und wir speichern pro Zelle eine Liste von Partikel-Indizes.

Wenn wir Nachbarn für ein Partikel i suchen, müssen wir dann nicht mehr alle Partikel betrachten,
sondern nur Partikel aus der eigenen Zelle und den 8 angrenzenden Zellen (3×3 Block).

Wichtig:
--------
Diese Nachbarsuche liefert nur "Kandidaten".
Ob ein Kandidat wirklich ein Nachbar ist, hängt später von einer Distanzprüfung ab (z.B. r <= h).

EN:
Neighbor search for 2D particles using a simple uniform grid.

Why do we need neighbor search in SPH?
--------------------------------------
In SPH (Smoothed Particle Hydrodynamics), we compute many sums over neighbors,
e.g., for density, pressure forces, viscosity, etc. Formally, formulas often sum over *all* particles j,
but in practice only the local neighborhood within a radius h matters (kernel compact support).

If we checked all other particles j for every particle i, the cost would be O(N^2).
This becomes slow quickly as N grows.

The idea of a uniform grid:
---------------------------
We split space into equal-sized square cells (cell_size).
Each particle is assigned to a cell, and per cell we store a list of particle indices.

When we search neighbors for a particle i, we no longer consider all particles,
but only particles from its own cell and the 8 adjacent cells (a 3×3 block).

Important:
----------
This neighbor search returns only "candidates".
Whether a candidate is a true neighbor is decided later via a distance check (e.g., r <= h).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Dict, List, Tuple

import numpy as np

from sph_sim.core.particles import ParticleSet2D


def cell_index(x: float, y: float, cell_size: float) -> tuple[int, int]:
    """
    DE:
    Berechne die Zell-Koordinaten (ix, iy) für eine Position (x, y).

    Was ist eine "Zelle"?
    - Wir teilen den 2D-Raum in Quadrate fester Größe `cell_size`.
    - Jede Zelle bekommt ganzzahlige Koordinaten (ix, iy).

    Warum benutzen wir floor?
    - Beispiel: cell_size = 0.1
      - x = 0.00 .. 0.099... gehört zu ix = 0
      - x = 0.10 .. 0.199... gehört zu ix = 1
    - `np.floor(x / cell_size)` macht genau diese "Einteilung in Intervalle".

    Validierung:
    - `cell_size` muss > 0 sein.

    Parameter:
    - x, y: Position
    - cell_size: Kantenlänge einer Grid-Zelle

    Rückgabe:
    - (ix, iy): Zellkoordinaten als (int, int)

    EN:
    Compute the cell coordinates (ix, iy) for a position (x, y).

    What is a "cell"?
    - We split the 2D space into squares of fixed size `cell_size`.
    - Each cell gets integer coordinates (ix, iy).

    Why do we use floor?
    - Example: cell_size = 0.1
      - x = 0.00 .. 0.099... belongs to ix = 0
      - x = 0.10 .. 0.199... belongs to ix = 1
    - `np.floor(x / cell_size)` performs exactly this "binning into intervals".

    Validation:
    - `cell_size` must be > 0.

    Parameters:
    - x, y: position
    - cell_size: edge length of a grid cell

    Returns:
    - (ix, iy): cell coordinates as (int, int)
    """

    if cell_size <= 0.0:
        raise ValueError("cell_size muss > 0 sein.")

    ix = int(np.floor(x / cell_size))
    iy = int(np.floor(y / cell_size))
    return (ix, iy)


def build_uniform_grid(
    particles: ParticleSet2D,
    cell_size: float,
) -> Dict[tuple[int, int], list[int]]:
    """
    DE:
    Baue ein Uniform-Grid (Dictionary), das Partikel-Indizes pro Zelle speichert.

    Idee:
    - Wir ordnen jedes Partikel einer Zelle (ix, iy) zu.
    - In einem Dictionary speichern wir dann:
      Key   = (ix, iy)   (Zellkoordinaten)
      Value = [i0, i1, ...] (Liste von Partikel-Indizes in dieser Zelle)

    Warum ein Dict?
    - Nicht jede Zelle ist belegt.
    - Ein Dict speichert nur Zellen, die wirklich Partikel enthalten.
      (Das ist speichereffizient und einfach zu nutzen.)

    Warum Listen von Indizes und nicht Kopien der Daten?
    - Unsere Partikeldaten liegen schon im `ParticleSet2D` (SoA: x, y, vx, ... als Arrays).
    - Wenn wir Daten kopieren würden, hätten wir:
      - mehr Speicherverbrauch
      - Gefahr von Inkonsistenzen (Kopie vs. Original)
    - Indizes sind klein, schnell, und verweisen eindeutig auf die Originaldaten.

    Validierung:
    - `cell_size` muss > 0 sein.

    Parameter:
    - particles: Partikelsatz (Positionen liegen in particles.x und particles.y)
    - cell_size: Kantenlänge einer Grid-Zelle

    Rückgabe:
    - grid: Dict[(ix, iy)] -> list of particle indices

    EN:
    Build a uniform grid (dictionary) that stores particle indices per cell.

    Idea:
    - We assign each particle to a cell (ix, iy).
    - In a dictionary we store:
      Key   = (ix, iy)   (cell coordinates)
      Value = [i0, i1, ...] (list of particle indices in that cell)

    Why a dict?
    - Not every cell is occupied.
    - A dict stores only cells that actually contain particles.
      (This is memory-efficient and easy to use.)

    Why lists of indices instead of copies of the data?
    - Our particle data already lives in `ParticleSet2D` (SoA: x, y, vx, ... as arrays).
    - If we copied data, we would have:
      - more memory usage
      - risk of inconsistencies (copy vs original)
    - Indices are small, fast, and clearly reference the original data.

    Validation:
    - `cell_size` must be > 0.

    Parameters:
    - particles: particle set (positions are in particles.x and particles.y)
    - cell_size: edge length of a grid cell

    Returns:
    - grid: Dict[(ix, iy)] -> list of particle indices
    """

    if cell_size <= 0.0:
        raise ValueError("cell_size muss > 0 sein.")

    grid: Dict[tuple[int, int], list[int]] = {}

    # --- Deutsch ---
    # Wir gehen über alle Partikel i und stecken i in die passende Zelle.
    # Wir speichern nur Indizes, keine Datenkopien.
    #
    # --- English ---
    # We iterate over all particles i and put i into the appropriate cell.
    # We store indices only, no data copies.
    for i in range(particles.n):
        ix, iy = cell_index(float(particles.x[i]), float(particles.y[i]), cell_size=cell_size)
        key = (ix, iy)

        # --- Deutsch ---
        # Wenn es die Zelle noch nicht gibt, legen wir eine neue Liste an.
        # Danach hängen wir den Index i an.
        #
        # --- English ---
        # If the cell does not exist yet, we create a new list.
        # Then we append the index i.
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    return grid


def query_neighbor_candidates(
    i: int,
    particles: ParticleSet2D,
    grid: Dict[tuple[int, int], list[int]],
    cell_size: float,
) -> list[int]:
    """
    DE:
    Liefere Kandidaten-Indizes für Nachbarn von Partikel i aus einem Uniform-Grid.

    Was bedeutet "Kandidaten"?
    - Wir sammeln alle Partikel aus der eigenen Zelle und den 8 Nachbarzellen (3×3).
    - Das ist nur eine Vorauswahl.
    - Später prüfen wir für jeden Kandidaten noch die echte Distanz (z.B. r <= h).
      Erst dann wissen wir, ob es wirklich ein Nachbar ist.

    Warum 3×3 Zellen?
    - Ein Partikel kann nahe an einer Zellgrenze liegen.
    - Dann können echte Nachbarn in der benachbarten Zelle sitzen.
    - Darum betrachten wir die eigene Zelle und alle direkten Nachbarn.

    Validierung:
    - 0 <= i < particles.n
    - cell_size > 0

    Parameter:
    - i: Index des Fokus-Partikels
    - particles: Partikelsatz (für die Position von i)
    - grid: Uniform-Grid aus `build_uniform_grid`
    - cell_size: Kantenlänge einer Grid-Zelle

    Rückgabe:
    - Liste von Kandidatenindices (kann i selbst enthalten)

    EN:
    Return candidate neighbor indices for particle i from a uniform grid.

    What does "candidates" mean?
    - We collect all particles from the particle's own cell and the 8 neighboring cells (3×3).
    - This is only a preselection.
    - Later, for each candidate we still check the actual distance (e.g., r <= h).
      Only then do we know whether it is a true neighbor.

    Why 3×3 cells?
    - A particle can be close to a cell boundary.
    - Then true neighbors may be located in an adjacent cell.
    - Therefore we check the particle's own cell and all direct neighbors.

    Validation:
    - 0 <= i < particles.n
    - cell_size > 0

    Parameters:
    - i: index of the focus particle
    - particles: particle set (for the position of i)
    - grid: uniform grid from `build_uniform_grid`
    - cell_size: edge length of a grid cell

    Returns:
    - list of candidate indices (may include i itself)
    """

    if cell_size <= 0.0:
        raise ValueError("cell_size muss > 0 sein.")

    if i < 0 or i >= particles.n:
        raise ValueError("i muss ein gültiger Partikelindex sein (0 <= i < particles.n).")

    ix, iy = cell_index(float(particles.x[i]), float(particles.y[i]), cell_size=cell_size)

    candidates: list[int] = []

    # --- Deutsch ---
    # Wir prüfen die 3×3 Umgebung um die Zelle (ix, iy).
    # Offsets {-1, 0, 1} in x und y ergeben genau 9 Zellen:
    # - die eigene Zelle (0, 0)
    # - und alle direkten Nachbarzellen.
    #
    # Wenn eine Zelle im Dict nicht existiert, gibt es dort keine Partikel → überspringen.
    #
    # --- English ---
    # We check the 3×3 neighborhood around the cell (ix, iy).
    # Offsets {-1, 0, 1} in x and y produce exactly 9 cells:
    # - the own cell (0, 0)
    # - and all directly adjacent neighbor cells.
    #
    # If a cell key does not exist in the dict, there are no particles there → skip it.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (ix + dx, iy + dy)
            if key in grid:
                candidates.extend(grid[key])

    return candidates


