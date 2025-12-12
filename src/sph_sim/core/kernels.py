"""
SPH-Kerne (Kernel-Funktionen) für Smoothed Particle Hydrodynamics (SPH).

Was ist ein SPH-Kernel?
-----------------------
In SPH ersetzen wir “harte” Punktabfragen (z.B. eine Dirac-Delta-Verteilung) durch eine
glatte Gewichtungsfunktion \(W(r, h)\).

Intuition:
- Jedes Partikel beeinflusst seine Umgebung.
- Wie stark dieser Einfluss ist, hängt vor allem von der Entfernung \(r\) ab.
- Der Kernel \(W(r, h)\) sagt uns: “Wie groß ist das Gewicht eines Nachbarn in Entfernung r?”

Parameter \(h\) (glättende Länge / smoothing length):
- \(h\) bestimmt den Einflussradius.
- Kleine \(h\) → sehr lokaler Einfluss, weniger Nachbarn
- Große \(h\) → weiterer Einfluss, mehr Nachbarn

Wofür nutzen wir den Poly6-Kernel?
---------------------------------
Der Poly6-Kernel wird häufig für die **Dichteschätzung** benutzt.
Warum?
- Er ist **glatt** (hat keine “Ecken”/Sprünge) und fällt sanft ab.
- “Glatt” ist wichtig, weil wir in SPH oft Ableitungen (Gradienten) brauchen.
  Wenn eine Funktion glatt ist, verhalten sich Ableitungen stabiler und die Simulation
  wird weniger “zappelig”.

Was bedeutet "compact support" (kompakte Trägerfunktion)?
--------------------------------------------------------
Viele SPH-Kerne haben *compact support*:

- Für \(r > h\) ist der Kernel exakt **0**.
- Das heißt: Ein Partikel beeinflusst nur Nachbarn innerhalb des Radius \(h\).

Warum ist das wichtig?
- Performance: Wir müssen nicht über alle Partikel rechnen, sondern nur über Nachbarn.
- Physikalische/algorithmische Kontrolle: Der Einfluss ist lokal begrenzt.

Diese Datei:
------------
Hier implementieren wir den Poly6-Kernel \(W(r, h)\) als Funktion, die sowohl
Skalare als auch NumPy-Arrays verarbeiten kann.
Das ist praktisch, weil wir später oft viele Distanzen auf einmal berechnen möchten
(vektorisiert in NumPy, ohne Python-Schleifen).
"""

from __future__ import annotations

import numpy as np


def poly6_kernel(r: float | np.ndarray, h: float) -> float | np.ndarray:
    """
    Poly6-Kernel \(W(r, h)\) (hier in 2D) zur Gewichtung nach Distanz.

    Mathematische Definition:
    - Für \(0 \\le r \\le h\):

      \(W(r, h) = C \\cdot (h^2 - r^2)^3\)

    - Für \(r > h\):

      \(W(r, h) = 0\)

    2D-Normierungskonstante:
    - Wir verwenden hier:

      \(C = \\frac{4}{\\pi \\cdot h^8}\)

    Warum ist der Poly6-Kernel "glatt"?
    - Der Ausdruck \((h^2 - r^2)^3\) ist ein Polynom.
    - Polynome sind überall glatt (differenzierbar).
    - Am Rand \(r = h\) wird der Wert 0 und der Übergang ist weich.

    Warum unterstützen wir Arrays?
    - In SPH berechnen wir oft viele Distanzen gleichzeitig (zu vielen Nachbarn).
    - NumPy kann das sehr schnell, wenn wir Arrays verwenden (Vektorisierung).

    Parameter
    - r: Distanz(en) zwischen zwei Partikeln (Skalar oder NumPy-Array)
    - h: smoothing length (muss > 0 sein)

    Rückgabe
    - Skalar-Eingabe → float
    - Array-Eingabe  → NumPy-Array gleicher Form
    """

    # --- Validierung: h muss sinnvoll sein ------------------------------------
    # h ist ein Längenmaß. Ein nicht-positives h würde keinen physikalischen Sinn ergeben
    # und würde außerdem zu Division durch 0 oder negativen Potenzen führen.
    if h <= 0.0:
        raise ValueError("h muss > 0 sein.")

    # --- r kann Skalar oder Array sein ----------------------------------------
    # Wir wandeln r immer in ein NumPy-Array (float64) um.
    # So können wir den gleichen Codepfad für Skalar und Array benutzen.
    r_arr = np.asarray(r, dtype=np.float64)

    # Merke dir, ob die Eingabe ein Skalar war:
    # - Bei Skalar wird `np.asarray` zu einem 0D-Array (ndim == 0).
    is_scalar_input = r_arr.ndim == 0

    # --- Distanz darf nicht negativ sein --------------------------------------
    # Distanzen sind per Definition nicht-negativ. Falls jemand negative Werte übergibt,
    # interpretieren wir sie als "Betrag der Distanz".
    r_abs = np.abs(r_arr)

    # --- Ergebnis-Array vorbereiten -------------------------------------------
    # Standardwert ist 0: Für r > h (außerhalb des compact support) ist der Kernel 0.
    W = np.zeros_like(r_abs, dtype=np.float64)

    # --- Maske: Welche Werte liegen innerhalb des Einflussradius? --------------
    # inside_mask ist True genau dort, wo r <= h gilt.
    inside_mask = r_abs <= h

    # --- Kernel nur für r <= h berechnen --------------------------------------
    # Damit vermeiden wir unnötige Rechnungen und setzen automatisch W=0 außerhalb.
    #
    # Achtung: Wir rechnen mit h^8. Das ist okay, solange h > 0 ist (haben wir geprüft).
    C = 4.0 / (np.pi * (h**8))

    # Für alle r innerhalb:
    # (h^2 - r^2)^3
    # Wir berechnen das nur für die "inside"-Elemente.
    h2 = h * h
    r2_inside = r_abs[inside_mask] ** 2
    W[inside_mask] = C * (h2 - r2_inside) ** 3

    # --- Rückgabe: Typ so lassen wie Eingabe ----------------------------------
    # Wenn die Eingabe ein Skalar war, geben wir wieder einen Python-float zurück.
    if is_scalar_input:
        return float(W.item())

    # Sonst geben wir das NumPy-Array zurück.
    return W


