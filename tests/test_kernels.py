"""
DE:
Unit-Tests für `sph_sim.core.kernels`.

Wichtig für Anfänger:
Diese Tests prüfen das Verhalten unserer Kernel-Funktion `poly6_kernel(r, h)`.
Ein Kernel ist in SPH eine Gewichtungsfunktion: nahe Nachbarn zählen mehr, weit entfernte
Nachbarn zählen weniger – und außerhalb des Radius h ist der Einfluss genau 0.

EN:
Unit tests for `sph_sim.core.kernels`.

Important for beginners:
These tests check the behavior of our kernel function `poly6_kernel(r, h)`.
A kernel in SPH is a weighting function: close neighbors count more, distant neighbors
count less — and outside the radius h the influence is exactly 0.
"""

from pathlib import Path
import sys

# --- Deutsch ---
# -----------------------------------------------------------------------------
# src/-Layout: Warum müssen wir `src/` in `sys.path` einfügen?
# -----------------------------------------------------------------------------
# Dieses Projekt nutzt eine "src-Struktur": der Python-Code liegt in `src/`.
# Wenn wir Tests starten, kennt Python diesen Ordner nicht automatisch als Import-Pfad.
#
# Darum fügen wir `src/` hier einmalig zum Suchpfad hinzu, damit `import sph_sim...` klappt.
# Später kann man das professionell über Packaging lösen (z.B. `pip install -e .`),
# dann braucht man diesen sys.path-Block nicht mehr.
#
# --- English ---
# -----------------------------------------------------------------------------
# src layout: why do we need to add `src/` to `sys.path`?
# -----------------------------------------------------------------------------
# This project uses a "src structure": the Python code lives in `src/`.
# When we start tests, Python does not automatically know this folder as an import path.
#
# Therefore we add `src/` once to the search path here so that `import sph_sim...` works.
# Later, you can solve this professionally via packaging (e.g., `pip install -e .`),
# then you no longer need this sys.path block.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest

from sph_sim.core.kernels import poly6_kernel


def test_poly6_kernel_raises_for_nonpositive_h() -> None:
    """
    DE:
    Test A: h Validierung.

    Warum?
    - h ist eine Länge (Influenzradius).
    - h muss > 0 sein, sonst ist das physikalisch unsinnig und mathematisch problematisch
      (z.B. wegen Division durch h^8).

    EN:
    Test A: h validation.

    Why?
    - h is a length (influence radius).
    - h must be > 0, otherwise this is physically meaningless and mathematically problematic
      (e.g., due to division by h^8).
    """

    # --- Deutsch ---
    # h = 0 → muss Fehler werfen
    #
    # --- English ---
    # h = 0 → must raise an error
    with pytest.raises(ValueError):
        poly6_kernel(r=0.0, h=0.0)

    # --- Deutsch ---
    # h < 0 → muss ebenfalls Fehler werfen
    #
    # --- English ---
    # h < 0 → must also raise an error
    with pytest.raises(ValueError):
        poly6_kernel(r=0.0, h=-1.0)


def test_poly6_kernel_is_zero_outside_support() -> None:
    """
    DE:
    Test B: außerhalb des Support ist der Kernel 0.

    "Compact support" bedeutet:
    - Für r > h ist W(r, h) exakt 0.
    - Das ist wichtig für Performance: nur Nachbarn innerhalb h müssen berücksichtigt werden.

    EN:
    Test B: outside the support the kernel is 0.

    "Compact support" means:
    - For r > h, W(r, h) is exactly 0.
    - This is important for performance: only neighbors within h need to be considered.
    """

    h = 1.0

    # --- Deutsch ---
    # r ist klar außerhalb (1.5 > 1.0)
    #
    # --- English ---
    # r is clearly outside (1.5 > 1.0)
    r = 1.5
    W = poly6_kernel(r=r, h=h)

    # --- Deutsch ---
    # Für Skalar-Eingabe erwarten wir auch einen Skalar zurück (float).
    #
    # --- English ---
    # For scalar input, we also expect a scalar return value (float).
    assert isinstance(W, float)

    # --- Deutsch ---
    # Und außerhalb muss das Gewicht exakt 0 sein.
    #
    # --- English ---
    # And outside, the weight must be exactly 0.
    assert W == 0.0


def test_poly6_kernel_positive_inside_support() -> None:
    """
    DE:
    Test C: innerhalb des Support ist der Kernel positiv.

    Idee:
    - Für 0 <= r < h ist (h^2 - r^2) > 0, und damit ist (h^2 - r^2)^3 > 0.
    - Mit einer positiven Konstante C ist W dann > 0.

    EN:
    Test C: inside the support the kernel is positive.

    Idea:
    - For 0 <= r < h, (h^2 - r^2) > 0, and therefore (h^2 - r^2)^3 > 0.
    - With a positive constant C, W is then > 0.
    """

    h = 1.0

    # --- Deutsch ---
    # Drei Distanzen "innerhalb": 0.0, 0.3, 0.9
    # (0.9 ist nahe am Rand, aber immer noch < h)
    #
    # --- English ---
    # Three distances "inside": 0.0, 0.3, 0.9
    # (0.9 is close to the boundary, but still < h)
    W0 = poly6_kernel(r=0.0, h=h)
    W1 = poly6_kernel(r=0.3, h=h)
    W2 = poly6_kernel(r=0.9, h=h)

    # --- Deutsch ---
    # Alle Werte müssen strikt positiv sein.
    #
    # --- English ---
    # All values must be strictly positive.
    assert W0 > 0.0
    assert W1 > 0.0
    assert W2 > 0.0


def test_poly6_kernel_decreases_with_distance() -> None:
    """
    DE:
    Test D: typisches monotones Verhalten.

    Für viele SPH-Kerne gilt:
    - Je weiter weg ein Nachbar ist, desto kleiner soll sein Gewicht werden.
    - Der Poly6-Kernel fällt im Bereich 0..h typischerweise mit wachsendem r ab.

    Wir testen das an drei Distanzen:
    W(0.1) > W(0.5) > W(0.9)

    EN:
    Test D: typical monotonic behavior.

    For many SPH kernels:
    - The farther away a neighbor is, the smaller its weight should become.
    - The Poly6 kernel typically decreases over 0..h as r increases.

    We test this at three distances:
    W(0.1) > W(0.5) > W(0.9)
    """

    h = 1.0

    W_near = poly6_kernel(r=0.1, h=h)
    W_mid = poly6_kernel(r=0.5, h=h)
    W_far = poly6_kernel(r=0.9, h=h)

    # --- Deutsch ---
    # Je kleiner r, desto größer das Gewicht.
    #
    # --- English ---
    # The smaller r is, the larger the weight.
    assert W_near > W_mid
    assert W_mid > W_far


def test_poly6_kernel_array_input_shape() -> None:
    """
    DE:
    Test E: Array-Eingabe funktioniert.

    Warum testen wir das?
    - In der Praxis berechnen wir viele Kernel-Werte auf einmal (für viele Nachbarn).
    - Dafür wollen wir Arrays direkt unterstützen (NumPy-Vektorisierung).

    EN:
    Test E: array input works.

    Why do we test this?
    - In practice, we compute many kernel values at once (for many neighbors).
    - For this, we want to support arrays directly (NumPy vectorization).
    """

    r = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    h = 1.0

    W = poly6_kernel(r=r, h=h)

    # --- Deutsch ---
    # Bei Array-Eingabe muss auch ein NumPy-Array herauskommen.
    #
    # --- English ---
    # For array input, the result must also be a NumPy array.
    assert isinstance(W, np.ndarray)

    # --- Deutsch ---
    # Shape muss gleich bleiben: Eingabe hatte shape (3,), Ausgabe auch.
    #
    # --- English ---
    # The shape must stay the same: input had shape (3,), output as well.
    assert W.shape == r.shape

    # --- Deutsch ---
    # Für r=2.0 liegt die Distanz außerhalb des Supports (2.0 > 1.0) → 0.
    #
    # --- English ---
    # For r=2.0, the distance is outside the support (2.0 > 1.0) → 0.
    assert W[-1] == 0.0


