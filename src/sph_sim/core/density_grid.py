"""
DE:
Grid-basierte SPH-Dichteberechnung (2D) als didaktische Zwischenstufe.

Ziel dieser Datei
-----------------
Wir berechnen die gleiche SPH-Dichte wie in der naiven Referenz (O(N^2)),
aber wir reduzieren die Anzahl der geprüften Paare mit einer Nachbarsuche (Uniform Grid).

Physik / Formel (identisch zur naiven Methode)
---------------------------------------------
Für jedes Partikel i gilt:

    rho[i] = sum_j ( m[j] * W(r_ij, h) )

- r_ij ist die Distanz zwischen Partikel i und j.
- W(r,h) ist ein SPH-Kernel (hier: Poly6).
- h ist die smoothing length / der Einflussradius.

Warum ist das Ergebnis identisch zur naiven Methode?
---------------------------------------------------
Der Poly6-Kernel hat "compact support":
- Für r > h ist W(r,h) = 0.

In der naiven Methode summieren wir formal über alle j.
Aber alle j mit r_ij > h tragen sowieso 0 bei.

Die Grid-Methode macht genau das explizit:
- Wir holen nur Kandidaten j aus einer lokalen Umgebung (3×3 Zellen).
- Wir prüfen r_ij <= h.
- Nur dann addieren wir m[j] * W(r_ij, h).

Das Grid ändert also NICHT die Physik, sondern nur die Suchstrategie.

Performance-Idee
----------------
Naiv: O(N^2), weil für jedes i alle j geprüft werden.
Mit Grid: ungefähr O(N * k), wobei k die typische Kandidatenzahl pro Partikel ist.

EN:
Grid-based SPH density computation (2D) as a didactic intermediate step.

Goal of this file
-----------------
We compute the same SPH density as in the naive reference (O(N^2)),
but we reduce the number of checked pairs using neighbor search (uniform grid).

Physics / formula (identical to the naive method)
-------------------------------------------------
For each particle i:

    rho[i] = sum_j ( m[j] * W(r_ij, h) )

- r_ij is the distance between particle i and j.
- W(r,h) is an SPH kernel (here: Poly6).
- h is the smoothing length / influence radius.

Why is the result identical to the naive method?
------------------------------------------------
The Poly6 kernel has "compact support":
- For r > h, W(r,h) = 0.

In the naive method we formally sum over all j.
But all j with r_ij > h contribute 0 anyway.

The grid method makes this explicit:
- We only query candidate j from a local area (3×3 cells).
- We check r_ij <= h.
- Only then we add m[j] * W(r_ij, h).

So the grid does NOT change physics, only the search strategy.

Performance idea
----------------
Naive: O(N^2) because for each i we check all j.
With a grid: roughly O(N * k), where k is the typical number of candidates per particle.
"""

from __future__ import annotations

import numpy as np

from sph_sim.core.particles import ParticleSet2D
from sph_sim.core.kernels import poly6_kernel
from sph_sim.core.neighbor_search import build_uniform_grid, query_neighbor_candidates


def compute_density_grid(
    particles: ParticleSet2D,
    h: float,
    cell_size: float | None = None,
) -> np.ndarray:
    """
    DE:
    Berechne die SPH-Dichte \(\rho\) mit einer Grid-basierten Kandidatensuche.

    Wichtig: Physik identisch zur naiven Methode
    --------------------------------------------
    Wir berechnen exakt dieselbe Formel:

        rho[i] = sum_j ( m[j] * W(r_ij, h) )

    Der Unterschied ist nur, wie wir die j finden:
    - Naiv: alle j prüfen.
    - Grid: Kandidaten aus 3×3 Zellen holen und dann r<=h prüfen.

    Validierung
    ----------
    - h muss > 0 sein.

    cell_size
    ---------
    - Wenn cell_size None ist, setzen wir cell_size = h.

      Warum ist das sinnvoll?
      - h ist der Einflussradius.
      - Wenn cell_size = h gilt, dann liegen alle echten Nachbarn (r<=h)
        in der eigenen Zelle oder in einer der 8 Nachbarzellen (3×3 Block).
      - Genau diese 3×3 Zellen liefert `query_neighbor_candidates`.

    Rückgabe
    --------
    - Ein neues Array rho (Shape: (N,)), ohne `particles` zu verändern.

    EN:
    Compute SPH density \(\rho\) using a grid-based candidate search.

    Important: physics identical to the naive method
    -----------------------------------------------
    We compute the exact same formula:

        rho[i] = sum_j ( m[j] * W(r_ij, h) )

    The difference is only how we find j:
    - Naive: check all j.
    - Grid: query candidates from 3×3 cells and then check r<=h.

    Validation
    ----------
    - h must be > 0.

    cell_size
    ---------
    - If cell_size is None, we set cell_size = h.

      Why is this sensible?
      - h is the influence radius.
      - If cell_size = h, then all true neighbors (r<=h) lie in the same cell
        or one of the 8 adjacent cells (a 3×3 block).
      - `query_neighbor_candidates` returns exactly these 3×3 cells.

    Returns
    -------
    - A new rho array (shape: (N,)) without modifying `particles`.
    """

    # --- Deutsch ---
    # Validierung: h muss > 0 sein.
    #
    # --- English ---
    # Validation: h must be > 0.
    if h <= 0.0:
        raise ValueError("h muss > 0 sein.")

    # --- Deutsch ---
    # Standard: cell_size = h.
    # Warum? Dann reichen 3×3 Zellen um ein Partikel, um alle Nachbarn mit r<=h zu finden.
    #
    # --- English ---
    # Default: cell_size = h.
    # Why? Then a 3×3 block of cells around a particle is sufficient to find all neighbors with r<=h.
    if cell_size is None:
        cell_size = float(h)

    # --- Deutsch ---
    # Optional: wir prüfen, dass cell_size > 0 ist.
    #
    # --- English ---
    # Optional: we check that cell_size > 0.
    if float(cell_size) <= 0.0:
        raise ValueError("cell_size muss > 0 sein.")

    # --- Deutsch ---
    # WICHTIG für korrekte Physik:
    # Unsere Kandidatensuche schaut nur 3×3 Zellen an.
    # Damit wir garantiert *alle* Nachbarn mit r<=h enthalten, muss cell_size >= h gelten.
    #
    # --- English ---
    # IMPORTANT for correct physics:
    # Our candidate query only checks a 3×3 block of cells.
    # To guarantee we include *all* neighbors with r<=h, we need cell_size >= h.
    if float(cell_size) < float(h):
        raise ValueError("cell_size muss >= h sein, sonst könnten echte Nachbarn (r<=h) übersehen werden.")

    N = particles.n

    # --- Deutsch ---
    # Ergebnisarray: rho für jedes Partikel i.
    #
    # --- English ---
    # Result array: rho for each particle i.
    rho = np.zeros(N, dtype=np.float64)

    # --- Deutsch ---
    # Grid einmal bauen (O(N)):
    # - Key: Zellindex (ix, iy)
    # - Value: Liste von Partikelindizes in dieser Zelle
    #
    # --- English ---
    # Build the grid once (O(N)):
    # - Key: cell index (ix, iy)
    # - Value: list of particle indices in that cell
    grid = build_uniform_grid(particles=particles, cell_size=float(cell_size))

    # --- Deutsch ---
    # Hauptschleife über alle Partikel i.
    #
    # Warum ist das weniger Arbeit als naiv?
    # - Naiv prüfen wir für jedes i alle j (N Stück).
    # - Hier prüfen wir nur Kandidaten aus der lokalen Umgebung (typisch viel weniger als N).
    #
    # --- English ---
    # Main loop over all particles i.
    #
    # Why is this less work than naive?
    # - Naive checks all j (N of them) for each i.
    # - Here we only check candidates from the local area (typically far less than N).
    for i in range(N):
        candidates = query_neighbor_candidates(
            i=int(i),
            particles=particles,
            grid=grid,
            cell_size=float(cell_size),
        )

        xi = float(particles.x[i])
        yi = float(particles.y[i])

        # --- Deutsch ---
        # Summe über Kandidaten:
        # - Wir berechnen r_ij.
        # - Wenn r_ij > h: überspringen (weil W=0, compact support).
        # - Sonst: addiere m_j * W(r_ij, h).
        #
        # --- English ---
        # Sum over candidates:
        # - Compute r_ij.
        # - If r_ij > h: skip (because W=0, compact support).
        # - Else: add m_j * W(r_ij, h).
        for j in candidates:
            j_int = int(j)

            dx = float(particles.x[j_int]) - xi
            dy = float(particles.y[j_int]) - yi
            r = float(np.sqrt(dx * dx + dy * dy))

            if r > float(h):
                continue

            w_ij = float(poly6_kernel(r, float(h)))
            rho[i] += float(particles.m[j_int]) * w_ij

    return rho


