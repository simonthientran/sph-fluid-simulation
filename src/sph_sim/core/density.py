"""
DE:
Naive SPH-Dichteberechnung (Referenz-Implementierung).

Ziel dieser Datei
----------------
In SPH (Smoothed Particle Hydrodynamics) berechnen wir die Dichte an jedem Partikel i,
indem wir Beiträge von allen Partikeln j aufsummieren.

Warum machen wir das so?
- In SPH ist Dichte keine "magische" Zahl, sondern wird aus der Nachbarschaft geschätzt:

  rho[i] = sum_j ( m[j] * W(r_ij, h) )

  - m[j] ist die Masse des Partikels j
  - W(...) ist ein Kernel (Gewichtungsfunktion), der mit Abstand abnimmt
  - r_ij ist der Abstand zwischen Partikel i und j
  - h ist die smoothing length (Einflussradius)

Warum "naiv"?
-------------
Diese Implementierung nutzt eine doppelte Schleife über alle Partikelpaare (i, j).
Das ist fachlich korrekt, aber langsam:

- Äußere Schleife: i = 0..N-1
- Innere Schleife: j = 0..N-1

Das sind insgesamt N * N Iterationen → **O(N^2)**.
Bei vielen Partikeln (z.B. 100.000) ist das zu langsam.

Warum behalten wir diese Funktion trotzdem?
-------------------------------------------
- Als Referenz ("Ground Truth") ist sie sehr wertvoll:
  - Sie ist einfach zu verstehen.
  - Sie ist schwer "falsch" zu implementieren, weil sie direkt der Formel folgt.
  - Wir können spätere, schnellere Versionen (Neighbor-Lists, Grid, KD-Tree, GPU)
    damit testen und vergleichen.

Wichtig:
- Diese Funktion verändert `particles` nicht.
- Sie liefert ein neues Array `rho` zurück.

EN:
Naive SPH density computation (reference implementation).

Goal of this file
In SPH (Smoothed Particle Hydrodynamics), we compute the density at each particle i
by summing contributions from all particles j.

Why do we do this?
- In SPH, density is not a "magical" number; it is estimated from the neighborhood:

  rho[i] = sum_j ( m[j] * W(r_ij, h) )

  - m[j] is the mass of particle j
  - W(...) is a kernel (weighting function) that decays with distance
  - r_ij is the distance between particle i and j
  - h is the smoothing length (influence radius)

Why "naive"?
This implementation uses a double loop over all particle pairs (i, j).
This is scientifically correct, but slow:

- Outer loop: i = 0..N-1
- Inner loop: j = 0..N-1

That is a total of N * N iterations → **O(N^2)**.
For many particles (e.g., 100,000), this is too slow.

Why do we keep this function anyway?
- As a reference ("ground truth"), it is very valuable:
  - It is easy to understand.
  - It is hard to implement "wrong" because it follows the formula directly.
  - We can test and compare later, faster versions (neighbor lists, grid, KD-tree, GPU)
    against it.

Important:
- This function does not modify `particles`.
- It returns a new array `rho`.
"""

import numpy as np

from sph_sim.core.particles import ParticleSet2D
from sph_sim.core.kernels import poly6_kernel


def compute_density_naive(particles: ParticleSet2D, h: float) -> np.ndarray:
    """
    DE:
    Berechne die SPH-Dichte für jedes Partikel mit einer naiven O(N^2)-Methode.

    Parameter
    - particles: Partikelset (Struct-of-Arrays). Wir lesen nur daraus und ändern nichts.
    - h: smoothing length (Einflussradius des Kernels). Muss > 0 sein.

    Rückgabe
    - rho: 1D-NumPy-Array der Länge N mit der Dichte pro Partikel.

    Didaktische Erklärung der Formel:
    - Für jedes Partikel i wollen wir wissen, "wie viel Masse in seiner Umgebung" liegt.
    - Dazu summieren wir über alle Partikel j:
        - Der Beitrag von j ist m[j] * W(r_ij, h).
        - W ist groß, wenn j nahe an i ist, und wird 0, wenn j weiter als h entfernt ist.

    EN:
    Compute the SPH density for each particle using a naive O(N^2) method.

    Parameters
    - particles: particle set (struct-of-arrays). We only read from it and do not modify it.
    - h: smoothing length (kernel influence radius). Must be > 0.

    Returns
    - rho: 1D NumPy array of length N with the density per particle.

    Didactic explanation of the formula:
    - For each particle i, we want to know "how much mass lies in its neighborhood".
    - For this, we sum over all particles j:
        - The contribution from j is m[j] * W(r_ij, h).
        - W is large when j is close to i, and becomes 0 when j is farther than h away.
    """

    # --- Deutsch ---
    # --- Validierung ----------------------------------------------------------
    # h ist ein Längenmaß (Radius). h <= 0 ist unsinnig und würde die Kernel-Formel brechen.
    #
    # --- English ---
    # --- Validation -----------------------------------------------------------
    # h is a length scale (radius). h <= 0 is meaningless and would break the kernel formula.
    if h <= 0.0:
        raise ValueError("h muss > 0 sein.")

    # --- Deutsch ---
    # Anzahl der Partikel (N). Diese Property ist in ParticleSet2D definiert.
    #
    # --- English ---
    # Number of particles (N). This property is defined in ParticleSet2D.
    N = particles.n

    # --- Deutsch ---
    # Ergebnisarray vorbereiten:
    # - Startwert 0, weil wir gleich Beiträge aufsummieren (+=).
    # - float64 ist Standard für numerische Stabilität in diesem Projekt.
    #
    # --- English ---
    # Prepare the result array:
    # - Initial value 0 because we will accumulate contributions (+=) next.
    # - float64 is the standard for numerical stability in this project.
    rho = np.zeros(N, dtype=np.float64)

    # --- Deutsch ---
    # Für Lesbarkeit holen wir uns die Arrays in lokale Variablen.
    # Das ändert nichts am Inhalt von `particles` (wir lesen nur).
    #
    # --- English ---
    # For readability, we copy the arrays into local variables.
    # This does not change the content of `particles` (we only read).
    x = particles.x
    y = particles.y
    m = particles.m

    # --- Deutsch ---
    # --- Doppelte Schleife über alle Paare (i, j) -----------------------------
    # Warum laufen wir über alle j?
    # - Die Dichte an Partikel i ist eine Summe von Beiträgen aller Partikel j.
    # - Auch Partikel, die weit weg sind, sind formal Teil der Summe.
    #   Praktisch wird ihr Beitrag aber 0, weil der Kernel bei r > h = 0 ist
    #   ("compact support").
    #
    # Warum ist das O(N^2)?
    # - Für jedes i (N Möglichkeiten) laufen wir über alle j (N Möglichkeiten).
    # - Das ergibt N * N Schritte.
    #
    # Warum trotzdem fachlich korrekt?
    # - Genau so steht es in der Formel: Summe über alle j.
    # - Keine Annahmen, keine Abkürzungen, keine Datenstrukturen für Nachbarn.
    #
    # --- English ---
    # --- Double loop over all pairs (i, j) -----------------------------------
    # Why do we loop over all j?
    # - The density at particle i is a sum of contributions from all particles j.
    # - Even particles that are far away are formally part of the sum.
    #   In practice, their contribution is 0 because the kernel is 0 for r > h
    #   ("compact support").
    #
    # Why is this O(N^2)?
    # - For each i (N possibilities), we loop over all j (N possibilities).
    # - That yields N * N steps.
    #
    # Why is it still scientifically correct?
    # - This is exactly what the formula states: sum over all j.
    # - No assumptions, no shortcuts, no data structures for neighbors.
    for i in range(N):
        for j in range(N):
            # --- Deutsch ---
            # Abstand zwischen Partikel i und j in 2D:
            # - dx, dy sind Komponenten der Differenz.
            #
            # --- English ---
            # Distance between particle i and j in 2D:
            # - dx, dy are the components of the difference.
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            # --- Deutsch ---
            # Euklidischer Abstand r = sqrt(dx^2 + dy^2)
            # (wir schreiben es bewusst "langsam und klar", keine Optimierung)
            #
            # --- English ---
            # Euclidean distance r = sqrt(dx^2 + dy^2)
            # (we write it deliberately "slow and clear", no optimization)
            r = np.sqrt(dx * dx + dy * dy)

            # --- Deutsch ---
            # Kernelwert W(r, h): Gewicht abhängig von Distanz.
            #
            # --- English ---
            # Kernel value W(r, h): weight depending on distance.
            w_ij = poly6_kernel(r, h)

            # --- Deutsch ---
            # Beitrag von Partikel j zur Dichte an Partikel i:
            # rho[i] += m[j] * W(r_ij, h)
            #
            # --- English ---
            # Contribution of particle j to the density at particle i:
            # rho[i] += m[j] * W(r_ij, h)
            rho[i] += m[j] * w_ij

    # --- Deutsch ---
    # Wir geben das neue Array zurück. `particles` bleibt unverändert.
    #
    # --- English ---
    # We return the new array. `particles` remains unchanged.
    return rho


