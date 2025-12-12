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

from sph_sim.core.kernels import poly6_kernel, spiky_kernel_gradient


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


def test_spiky_kernel_gradient_raises_for_nonpositive_h() -> None:
    """
    DE:
    Test F: h Validierung für den Spiky-Gradienten.

    Warum?
    - h ist eine Länge (Influenzradius / smoothing length).
    - Bei h <= 0 wäre die Formel unsinnig und würde in der Konstante h^5 Probleme machen.

    EN:
    Test F: h validation for the spiky gradient.

    Why?
    - h is a length (influence radius / smoothing length).
    - For h <= 0 the formula is meaningless and would cause issues in the constant h^5.
    """

    # --- Deutsch ---
    # h = 0 → ValueError
    #
    # --- English ---
    # h = 0 → ValueError
    with pytest.raises(ValueError):
        spiky_kernel_gradient(dx=1.0, dy=0.0, h=0.0)

    # --- Deutsch ---
    # h < 0 → ValueError
    #
    # --- English ---
    # h < 0 → ValueError
    with pytest.raises(ValueError):
        spiky_kernel_gradient(dx=1.0, dy=0.0, h=-1.0)


def test_spiky_kernel_gradient_is_zero_outside_support() -> None:
    """
    DE:
    Test G: außerhalb des Supports (r > h) ist der Gradient (0, 0).

    Warum?
    - SPH-Kerne haben compact support.
    - Außerhalb des Radius h soll es keinen Beitrag geben.

    EN:
    Test G: outside the support (r > h) the gradient is (0, 0).

    Why?
    - SPH kernels have compact support.
    - Outside radius h there should be no contribution.
    """

    h = 1.0

    # --- Deutsch ---
    # Wir wählen dx so, dass r eindeutig größer als h ist (r = 2.0).
    #
    # --- English ---
    # We choose dx so that r is clearly larger than h (r = 2.0).
    gx, gy = spiky_kernel_gradient(dx=2.0, dy=0.0, h=h)

    assert gx == 0.0
    assert gy == 0.0


def test_spiky_kernel_gradient_is_zero_at_r_equals_zero() -> None:
    """
    DE:
    Test H: Sonderfall r == 0 → (0, 0).

    Warum?
    - In der Formel würde /r stehen (Richtung dx/r, dy/r).
    - Bei r == 0 wäre das Division durch 0.
    - Außerdem ist die Richtung bei exakt gleicher Position nicht definiert.

    EN:
    Test H: special case r == 0 → (0, 0).

    Why?
    - The formula would contain /r (direction dx/r, dy/r).
    - At r == 0 this would be division by zero.
    - Also the direction is not defined when the positions are exactly equal.
    """

    gx, gy = spiky_kernel_gradient(dx=0.0, dy=0.0, h=1.0)
    assert gx == 0.0
    assert gy == 0.0


def test_spiky_kernel_gradient_direction_dx_positive_dy_zero() -> None:
    """
    DE:
    Test I: Richtungstest (sehr einfach).

    Setup:
    - dy = 0 → Wir erwarten gy ≈ 0.
    - dx > 0 und r < h → gx sollte ungleich 0 sein.

    Vorzeichen-Idee:
    - In unserer Implementierung ist die Konstante C positiv.
    - Für 0 < r < h gilt (h - r)^2 > 0 und /r > 0.
    - Also ist factor positiv.
    - Wenn dx > 0, dann ist gx = factor * dx > 0.

    EN:
    Test I: direction test (very simple).

    Setup:
    - dy = 0 → we expect gy ≈ 0.
    - dx > 0 and r < h → gx should be non-zero.

    Sign idea:
    - In our implementation the constant C is positive.
    - For 0 < r < h, (h - r)^2 > 0 and /r > 0.
    - Therefore factor is positive.
    - If dx > 0, then gx = factor * dx > 0.
    """

    h = 1.0
    dx = 0.2
    dy = 0.0

    gx, gy = spiky_kernel_gradient(dx=dx, dy=dy, h=h)

    # --- Deutsch ---
    # gy sollte (numerisch) 0 sein, weil dy = 0 ist.
    # Wir nutzen eine Toleranz, weil Fließkommazahlen nie perfekt sind.
    #
    # --- English ---
    # gy should be (numerically) 0 because dy = 0.
    # We use a tolerance because floating-point numbers are never perfect.
    assert np.isclose(gy, 0.0, atol=1e-15)

    # --- Deutsch ---
    # gx muss ungleich 0 sein (wir sind innerhalb des Supports und dx != 0).
    #
    # --- English ---
    # gx must be non-zero (we are inside the support and dx != 0).
    assert gx != 0.0

    # --- Deutsch ---
    # Vorzeichen: bei dx > 0 erwarten wir gx > 0 (siehe Erklärung oben).
    #
    # --- English ---
    # Sign: for dx > 0 we expect gx > 0 (see explanation above).
    assert gx > 0.0

