"""
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
"""

import numpy as np

from sph_sim.core.particles import ParticleSet2D
from sph_sim.core.kernels import poly6_kernel


def compute_density_naive(particles: ParticleSet2D, h: float) -> np.ndarray:
    """
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
    """

    # --- Validierung ----------------------------------------------------------
    # h ist ein Längenmaß (Radius). h <= 0 ist unsinnig und würde die Kernel-Formel brechen.
    if h <= 0.0:
        raise ValueError("h muss > 0 sein.")

    # Anzahl der Partikel (N). Diese Property ist in ParticleSet2D definiert.
    N = particles.n

    # Ergebnisarray vorbereiten:
    # - Startwert 0, weil wir gleich Beiträge aufsummieren (+=).
    # - float64 ist Standard für numerische Stabilität in diesem Projekt.
    rho = np.zeros(N, dtype=np.float64)

    # Für Lesbarkeit holen wir uns die Arrays in lokale Variablen.
    # Das ändert nichts am Inhalt von `particles` (wir lesen nur).
    x = particles.x
    y = particles.y
    m = particles.m

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
    for i in range(N):
        for j in range(N):
            # Abstand zwischen Partikel i und j in 2D:
            # - dx, dy sind Komponenten der Differenz.
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            # Euklidischer Abstand r = sqrt(dx^2 + dy^2)
            # (wir schreiben es bewusst "langsam und klar", keine Optimierung)
            r = np.sqrt(dx * dx + dy * dy)

            # Kernelwert W(r, h): Gewicht abhängig von Distanz.
            w_ij = poly6_kernel(r, h)

            # Beitrag von Partikel j zur Dichte an Partikel i:
            # rho[i] += m[j] * W(r_ij, h)
            rho[i] += m[j] * w_ij

    # Wir geben das neue Array zurück. `particles` bleibt unverändert.
    return rho


