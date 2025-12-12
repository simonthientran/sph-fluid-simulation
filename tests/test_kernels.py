"""
Unit-Tests für `sph_sim.core.kernels`.

Wichtig für Anfänger:
Diese Tests prüfen das Verhalten unserer Kernel-Funktion `poly6_kernel(r, h)`.
Ein Kernel ist in SPH eine Gewichtungsfunktion: nahe Nachbarn zählen mehr, weit entfernte
Nachbarn zählen weniger – und außerhalb des Radius h ist der Einfluss genau 0.
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

from sph_sim.core.kernels import poly6_kernel


def test_poly6_kernel_raises_for_nonpositive_h() -> None:
    """
    Test A: h Validierung.

    Warum?
    - h ist eine Länge (Influenzradius).
    - h muss > 0 sein, sonst ist das physikalisch unsinnig und mathematisch problematisch
      (z.B. wegen Division durch h^8).
    """

    # h = 0 → muss Fehler werfen
    with pytest.raises(ValueError):
        poly6_kernel(r=0.0, h=0.0)

    # h < 0 → muss ebenfalls Fehler werfen
    with pytest.raises(ValueError):
        poly6_kernel(r=0.0, h=-1.0)


def test_poly6_kernel_is_zero_outside_support() -> None:
    """
    Test B: außerhalb des Support ist der Kernel 0.

    "Compact support" bedeutet:
    - Für r > h ist W(r, h) exakt 0.
    - Das ist wichtig für Performance: nur Nachbarn innerhalb h müssen berücksichtigt werden.
    """

    h = 1.0

    # r ist klar außerhalb (1.5 > 1.0)
    r = 1.5
    W = poly6_kernel(r=r, h=h)

    # Für Skalar-Eingabe erwarten wir auch einen Skalar zurück (float).
    assert isinstance(W, float)

    # Und außerhalb muss das Gewicht exakt 0 sein.
    assert W == 0.0


def test_poly6_kernel_positive_inside_support() -> None:
    """
    Test C: innerhalb des Support ist der Kernel positiv.

    Idee:
    - Für 0 <= r < h ist (h^2 - r^2) > 0, und damit ist (h^2 - r^2)^3 > 0.
    - Mit einer positiven Konstante C ist W dann > 0.
    """

    h = 1.0

    # Drei Distanzen "innerhalb": 0.0, 0.3, 0.9
    # (0.9 ist nahe am Rand, aber immer noch < h)
    W0 = poly6_kernel(r=0.0, h=h)
    W1 = poly6_kernel(r=0.3, h=h)
    W2 = poly6_kernel(r=0.9, h=h)

    # Alle Werte müssen strikt positiv sein.
    assert W0 > 0.0
    assert W1 > 0.0
    assert W2 > 0.0


def test_poly6_kernel_decreases_with_distance() -> None:
    """
    Test D: typisches monotones Verhalten.

    Für viele SPH-Kerne gilt:
    - Je weiter weg ein Nachbar ist, desto kleiner soll sein Gewicht werden.
    - Der Poly6-Kernel fällt im Bereich 0..h typischerweise mit wachsendem r ab.

    Wir testen das an drei Distanzen:
    W(0.1) > W(0.5) > W(0.9)
    """

    h = 1.0

    W_near = poly6_kernel(r=0.1, h=h)
    W_mid = poly6_kernel(r=0.5, h=h)
    W_far = poly6_kernel(r=0.9, h=h)

    # Je kleiner r, desto größer das Gewicht.
    assert W_near > W_mid
    assert W_mid > W_far


def test_poly6_kernel_array_input_shape() -> None:
    """
    Test E: Array-Eingabe funktioniert.

    Warum testen wir das?
    - In der Praxis berechnen wir viele Kernel-Werte auf einmal (für viele Nachbarn).
    - Dafür wollen wir Arrays direkt unterstützen (NumPy-Vektorisierung).
    """

    r = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    h = 1.0

    W = poly6_kernel(r=r, h=h)

    # Bei Array-Eingabe muss auch ein NumPy-Array herauskommen.
    assert isinstance(W, np.ndarray)

    # Shape muss gleich bleiben: Eingabe hatte shape (3,), Ausgabe auch.
    assert W.shape == r.shape

    # Für r=2.0 liegt die Distanz außerhalb des Supports (2.0 > 1.0) → 0.
    assert W[-1] == 0.0


