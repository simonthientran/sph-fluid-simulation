"""
DE:
Kräfte / Beschleunigungen in SPH (Smoothed Particle Hydrodynamics).

In SPH berechnen wir oft nicht direkt "Kräfte", sondern **Beschleunigungen**.
Warum?
- Newton: \(F = m \cdot a\)
- Also: \(a = F / m\)
- In der Praxis ist es in SPH üblich, Formeln direkt so zu schreiben, dass am Ende
  eine Beschleunigung pro Partikel herauskommt.

Was machen wir in dieser Datei?
-------------------------------
Wir implementieren eine sehr einfache, didaktische Referenz-Funktion:

    compute_pressure_acceleration(...)

Sie berechnet die Beschleunigung durch **Druck** (pressure) für jedes Partikel
in 2D (also ax und ay).

Wichtig (Didaktik):
-------------------
- Wir nutzen eine **Uniform-Grid**-Struktur, um Nachbarschafts-Kandidaten zu finden.
  Das ist reine Performance/Algorithmik und KEINE Physik.
- Danach filtern wir physikalisch korrekt mit \(r \\le h\), weil der Kernel nur
  innerhalb des Radius \(h\) wirkt (compact support).
- Wir verwenden den **Spiky-Kernel-Gradienten** für Druckkräfte, weil er dafür
  in SPH sehr typisch ist.

EN:
Forces / accelerations in SPH (Smoothed Particle Hydrodynamics).

In SPH we often compute not "forces" directly, but **accelerations**.
Why?
- Newton: \(F = m \cdot a\)
- So: \(a = F / m\)
- In practice, SPH formulas are often written such that the result is directly
  an acceleration per particle.

What do we do in this file?
---------------------------
We implement a simple, didactic reference function:

    compute_pressure_acceleration(...)

It computes acceleration due to **pressure** for each particle in 2D (ax and ay).

Important (teaching):
---------------------
- We use a **uniform grid** to find neighbor candidates.
  This is pure performance/algorithmic structure and NOT physics.
- Afterwards we apply the physical filter \(r \\le h\), because the kernel only acts
  within radius \(h\) (compact support).
- We use the **Spiky kernel gradient** for pressure forces, because this is very common
  in SPH.
"""

import numpy as np

from sph_sim.core.kernels import spiky_kernel_gradient
from sph_sim.core.neighbor_search import build_uniform_grid, query_neighbor_candidates
from sph_sim.core.particles import ParticleSet2D


def compute_pressure_acceleration(
    particles: ParticleSet2D,
    rho: np.ndarray,
    p: np.ndarray,
    h: float,
    cell_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DE:
    Berechne die Druck-bedingte Beschleunigung \((a_x, a_y)\) für jedes Partikel.

    Kontext: Druckkräfte in SPH
    ---------------------------
    In SPH wird der Druckterm oft als Summe über Nachbarn formuliert.
    Eine typische (symmetrische) Form für die Beschleunigung am Partikel i ist:

        a_i = - Σ_j  m_j * ( (p_i + p_j) / (2 * ρ_j) ) * ∇W_spiky(r_ij, h)

    Dabei:
    - i ist das aktuelle Partikel, j ist ein Nachbar.
    - \(m_j\) ist die Masse des Nachbar-Partikels.
    - \(p_i, p_j\) sind die Drücke.
    - \(ρ_j\) ist die Dichte des Nachbarn.
    - \(∇W\) ist der Kernel-Gradient (hier: Spiky).

    Schritt-für-Schritt (was/warum):
    -------------------------------
    1) Warum das Minus?
       - Druck wirkt von "hoch" nach "niedrig".
       - Der Gradient zeigt in Richtung steigender Werte.
       - Deshalb kommt ein Minuszeichen: Kraft/Acceleration zeigt "entgegen" dem Gradienten.

    2) Warum (p_i + p_j) / 2 ?
       - Wir mitteln die Drücke von i und j (symmetrisch).
       - Symmetrische Formen sind in SPH oft stabiler und fairer, weil beide Partikel
         gleich behandelt werden.

    3) Warum durch rho im Nenner?
       - In Kontinuumsmechanik steht grob: Druckkraft ~ ∇p / ρ.
       - Hohe Dichte bedeutet: "mehr Masse pro Volumen" → gleiche Druckänderung führt
         zu kleinerer Beschleunigung.
       - Hier verwenden wir ρ_j (des Nachbarn), wie in vielen einfachen SPH-Formeln.

    Performance/Algorithmik:
    ------------------------
    - Naiv wäre die Summe über alle j: O(N^2).
    - Wir nutzen ein Uniform Grid:
      - Erst Kandidaten aus den umliegenden Zellen holen.
      - Dann physikalisch filtern: nur r <= h zählt wirklich.

    Parameter
    ---------
    particles:
        Partikelzustand (Positionen, Massen, ...). Wir lesen hier nur.
    rho:
        Dichten der Partikel (1D-Array, Länge N).
    p:
        Drücke der Partikel (1D-Array, Länge N).
    h:
        Smoothing length / Kernel-Radius (muss > 0 sein).
    cell_size:
        Zellgröße für das Uniform Grid. Wenn None, setzen wir cell_size = h.

    Rückgabe
    --------
    (ax, ay):
        Zwei 1D-Arrays (float64) der Länge N mit den Beschleunigungskomponenten.

    EN:
    Compute pressure-driven acceleration \((a_x, a_y)\) for each particle.

    Context: pressure forces in SPH
    -------------------------------
    In SPH, the pressure term is often written as a sum over neighbors.
    A typical (symmetric) form for acceleration at particle i is:

        a_i = - Σ_j  m_j * ( (p_i + p_j) / (2 * ρ_j) ) * ∇W_spiky(r_ij, h)

    Where:
    - i is the current particle, j is a neighbor.
    - \(m_j\) is the neighbor particle mass.
    - \(p_i, p_j\) are pressures.
    - \(ρ_j\) is the neighbor density.
    - \(∇W\) is the kernel gradient (here: Spiky).

    Step-by-step (what/why):
    ------------------------
    1) Why the minus sign?
       - Pressure pushes from "high" to "low".
       - The gradient points towards increasing values.
       - Therefore we use a minus sign: force/acceleration points opposite the gradient.

    2) Why (p_i + p_j) / 2 ?
       - We average the pressures of i and j (symmetric).
       - Symmetric forms are often more stable in SPH and treat both particles equally.

    3) Why density in the denominator?
       - In continuum mechanics, pressure force is roughly: ∇p / ρ.
       - Higher density means more mass per volume → the same pressure change leads to
         smaller acceleration.
       - Here we use ρ_j (neighbor density), as in many simple SPH formulations.

    Performance/algorithmics:
    -------------------------
    - The naive sum over all j would be O(N^2).
    - We use a uniform grid:
      - First get candidates from nearby cells.
      - Then apply the physical filter: only r <= h really contributes.

    Parameters
    ----------
    particles:
        Particle state (positions, masses, ...). We only read it here.
    rho:
        Particle densities (1D array, length N).
    p:
        Particle pressures (1D array, length N).
    h:
        Smoothing length / kernel radius (must be > 0).
    cell_size:
        Cell size for the uniform grid. If None, we set cell_size = h.

    Returns
    -------
    (ax, ay):
        Two 1D arrays (float64) of length N with acceleration components.
    """

    # --- Deutsch ---
    # --- Validierung: h muss > 0 sein ----------------------------------------
    #
    # --- English ---
    # --- Validation: h must be > 0 -------------------------------------------
    if float(h) <= 0.0:
        raise ValueError("h muss > 0 sein.")

    # --- Deutsch ---
    # cell_size Standard:
    # - Für eine einfache, korrekte Kandidatensuche ist cell_size = h eine sinnvolle Wahl.
    # - Dann reichen die 3x3 Nachbarzellen aus, um alle Nachbarn im Radius h zu finden.
    #
    # --- English ---
    # cell_size default:
    # - For a simple, correct candidate search, cell_size = h is a sensible choice.
    # - Then the 3x3 neighbor cells are enough to find all neighbors within radius h.
    if cell_size is None:
        cell_size = float(h)

    if float(cell_size) <= 0.0:
        raise ValueError("cell_size muss > 0 sein.")

    # --- Deutsch ---
    # rho und p in float64-Arrays umwandeln (einheitlicher Datentyp).
    #
    # --- English ---
    # Convert rho and p to float64 arrays (uniform dtype).
    rho_arr = np.asarray(rho, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)

    # --- Deutsch ---
    # --- Validierung: Form muss (N,) sein ------------------------------------
    # Wir erwarten 1D-Arrays mit der gleichen Länge wie die Anzahl der Partikel.
    #
    # --- English ---
    # --- Validation: shape must be (N,) --------------------------------------
    # We expect 1D arrays with the same length as the number of particles.
    N = particles.n
    if rho_arr.ndim != 1 or rho_arr.shape[0] != N:
        raise ValueError("rho muss ein 1D-Array der Länge N sein.")
    if p_arr.ndim != 1 or p_arr.shape[0] != N:
        raise ValueError("p muss ein 1D-Array der Länge N sein.")

    # --- Deutsch ---
    # Uniform Grid einmal bauen:
    # - Das Grid speichert: Zelle -> Liste von Partikelindizes.
    # - Das ist eine Suchstruktur, keine Physik.
    #
    # --- English ---
    # Build the uniform grid once:
    # - The grid stores: cell -> list of particle indices.
    # - This is a search structure, not physics.
    grid = build_uniform_grid(particles=particles, cell_size=float(cell_size))

    # --- Deutsch ---
    # Ergebnis-Arrays: Beschleunigungen (ax, ay), initial 0.
    #
    # --- English ---
    # Result arrays: accelerations (ax, ay), initialized to 0.
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)

    # --- Deutsch ---
    # --- Hauptschleife über alle Partikel i ----------------------------------
    # Für jedes i sammeln wir Kandidaten j aus dem Grid und filtern dann mit r <= h.
    #
    # --- English ---
    # --- Main loop over all particles i --------------------------------------
    # For each i we collect candidate neighbors j from the grid and then filter with r <= h.
    for i in range(N):
        # --- Deutsch ---
        # Kandidaten aus 3x3 Zellen um i holen.
        #
        # --- English ---
        # Get candidates from the 3x3 cells around i.
        candidates = query_neighbor_candidates(
            i=int(i),
            particles=particles,
            grid=grid,
            cell_size=float(cell_size),
        )

        xi = float(particles.x[i])
        yi = float(particles.y[i])

        # --- Deutsch ---
        # Innere Schleife: Beiträge aller Kandidaten j aufsummieren.
        #
        # --- English ---
        # Inner loop: sum contributions from all candidate neighbors j.
        for j in candidates:
            j_int = int(j)

            # --- Deutsch ---
            # Selbstbeitrag überspringen:
            # - j == i würde r == 0 ergeben.
            # - Wir vermeiden so unnötige Sonderfälle.
            #
            # --- English ---
            # Skip self contribution:
            # - j == i would produce r == 0.
            # - This avoids unnecessary special cases.
            if j_int == i:
                continue

            xj = float(particles.x[j_int])
            yj = float(particles.y[j_int])
            dx = xj - xi
            dy = yj - yi

            # --- Deutsch ---
            # Distanz r:
            # Wir brauchen r für den Support-Test (r <= h) und für den Kernel-Gradienten.
            #
            # --- English ---
            # Distance r:
            # We need r for the support test (r <= h) and for the kernel gradient.
            r = float(np.sqrt(dx * dx + dy * dy))

            # --- Deutsch ---
            # Physik-Filter: Nur Nachbarn innerhalb des Kernel-Radius zählen.
            #
            # --- English ---
            # Physics filter: only neighbors within the kernel radius contribute.
            if r > float(h):
                continue

            # --- Deutsch ---
            # Spiky-Gradient ∇W:
            # - Ergebnis ist ein Vektor (gx, gy).
            # - Für r==0 würde er (0,0) liefern; durch "self skip" kommt das hier praktisch nicht vor.
            #
            # --- English ---
            # Spiky gradient ∇W:
            # - Result is a vector (gx, gy).
            # - For r==0 it returns (0,0); due to "self skip" this practically does not happen here.
            gx, gy = spiky_kernel_gradient(dx=float(dx), dy=float(dy), h=float(h))

            # --- Deutsch ---
            # Druck-Term (symmetrisch, Schritt-für-Schritt):
            #
            # factor = - m_j * ((p_i + p_j) / (2 * rho_j))
            #
            # - m_j: Nachbar-Masse skaliert den Beitrag (mehr Masse → stärkerer Beitrag).
            # - (p_i + p_j)/2: symmetrischer Mittelwert der Drücke.
            # - / rho_j: Dichte im Nenner (ähnlich zu ∇p/ρ in der Kontinuumsmechanik).
            # - Minuszeichen: Druck wirkt gegen den Gradienten (von hoch nach niedrig).
            #
            # Danach:
            # ax_i += factor * gx
            # ay_i += factor * gy
            #
            # --- English ---
            # Pressure term (symmetric, step-by-step):
            #
            # factor = - m_j * ((p_i + p_j) / (2 * rho_j))
            #
            # - m_j: neighbor mass scales the contribution (more mass → stronger contribution).
            # - (p_i + p_j)/2: symmetric mean of pressures.
            # - / rho_j: density in the denominator (similar to ∇p/ρ in continuum mechanics).
            # - Minus sign: pressure acts against the gradient (from high to low).
            #
            # Then:
            # ax_i += factor * gx
            # ay_i += factor * gy
            factor = -float(particles.m[j_int]) * ((float(p_arr[i]) + float(p_arr[j_int])) / (2.0 * float(rho_arr[j_int])))
            ax[i] += factor * float(gx)
            ay[i] += factor * float(gy)

    return (ax, ay)


