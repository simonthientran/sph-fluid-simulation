"""
DE:
Pytest-Tests für `compute_pressure_eos` (EOS-basierte Druckberechnung).

Was testen wir?
--------------
Wir testen die Funktion:

    compute_pressure_eos(rho, rho0, k, clamp_negative=True)

Dabei prüfen wir:
- Shape (Ausgabeform)
- p = 0 bei rho = rho0 (wenn geclamped)
- linearer Zusammenhang ohne Clamping
- Clamping negativer Drücke
- Input-Validierung

EN:
Pytest tests for `compute_pressure_eos` (EOS-based pressure computation).

What do we test?
---------------
We test the function:

    compute_pressure_eos(rho, rho0, k, clamp_negative=True)

We check:
- shape (output shape)
- p = 0 at rho = rho0 (when clamped)
- linear relation without clamping
- clamping of negative pressures
- input validation
"""

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
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pytest

from sph_sim.core.pressure import compute_pressure_eos


def test_pressure_shape() -> None:
    """
    DE:
    Test A: Output-Shape ist identisch zu rho.

    Warum?
    - Druck wird "pro Partikel" berechnet.
    - Daher muss die Ausgabe die gleiche Länge/Form wie rho haben.

    EN:
    Test A: output shape matches rho.

    Why?
    - Pressure is computed "per particle".
    - Therefore the output must have the same length/shape as rho.
    """

    rho = np.array([1000.0, 1001.0], dtype=np.float64)
    p = compute_pressure_eos(rho=rho, rho0=1000.0, k=1.0)

    assert isinstance(p, np.ndarray)
    assert p.shape == rho.shape


def test_pressure_zero_at_rho0_when_clamped() -> None:
    """
    DE:
    Test B: Bei rho = rho0 ist p = 0 (wenn clamp_negative=True).

    Warum?
    - Formel: p = k * (rho - rho0)
    - Wenn rho == rho0, dann ist (rho - rho0) = 0 → p = 0
    - Clamping ändert daran nichts.

    EN:
    Test B: at rho = rho0, p = 0 (when clamp_negative=True).

    Why?
    - Formula: p = k * (rho - rho0)
    - If rho == rho0, then (rho - rho0) = 0 → p = 0
    - Clamping does not change that.
    """

    rho0 = 1000.0
    rho = np.array([rho0], dtype=np.float64)
    p = compute_pressure_eos(rho=rho, rho0=rho0, k=2.0, clamp_negative=True)

    assert p.shape == rho.shape
    assert p[0] == 0.0


def test_pressure_linear_relation_without_clamp() -> None:
    """
    DE:
    Test C: Linearer Zusammenhang ohne Clamping.

    Setup:
    - rho0 = 1000
    - k = 2
    - rho = [1000, 1001, 999]
    - clamp_negative = False

    Erwartung:
    - p = k * (rho - rho0)
      -> [2*(0), 2*(1), 2*(-1)] = [0, 2, -2]

    EN:
    Test C: linear relation without clamping.

    Setup:
    - rho0 = 1000
    - k = 2
    - rho = [1000, 1001, 999]
    - clamp_negative = False

    Expectation:
    - p = k * (rho - rho0)
      -> [2*(0), 2*(1), 2*(-1)] = [0, 2, -2]
    """

    rho0 = 1000.0
    k = 2.0
    rho = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)

    p = compute_pressure_eos(rho=rho, rho0=rho0, k=k, clamp_negative=False)

    expected = np.array([0.0, 2.0, -2.0], dtype=np.float64)
    assert np.allclose(p, expected)


def test_pressure_clamp_negative() -> None:
    """
    DE:
    Test D: Negative Drücke werden bei clamp_negative=True auf 0 geklemmt.

    Setup:
    - rho0 = 1000
    - k = 2
    - rho = [1000, 1001, 999]
    - ohne Clamping wäre p = [0, 2, -2]
    - mit Clamping erwarten wir p = [0, 2, 0]

    EN:
    Test D: negative pressures are clamped to 0 when clamp_negative=True.

    Setup:
    - rho0 = 1000
    - k = 2
    - rho = [1000, 1001, 999]
    - without clamping p would be [0, 2, -2]
    - with clamping we expect p = [0, 2, 0]
    """

    rho0 = 1000.0
    k = 2.0
    rho = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)

    p = compute_pressure_eos(rho=rho, rho0=rho0, k=k, clamp_negative=True)

    expected = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    assert np.allclose(p, expected)
    assert np.all(p >= 0.0)


def test_pressure_validates_inputs() -> None:
    """
    DE:
    Test E: Input-Validierung wirft ValueError.

    Wir testen:
    - rho0 <= 0  → ValueError
    - k <= 0     → ValueError
    - rho nicht 1D → ValueError

    EN:
    Test E: input validation raises ValueError.

    We test:
    - rho0 <= 0  → ValueError
    - k <= 0     → ValueError
    - rho not 1D → ValueError
    """

    rho_good = np.array([1000.0, 1001.0], dtype=np.float64)

    with pytest.raises(ValueError):
        compute_pressure_eos(rho=rho_good, rho0=0.0, k=1.0)

    with pytest.raises(ValueError):
        compute_pressure_eos(rho=rho_good, rho0=-1000.0, k=1.0)

    with pytest.raises(ValueError):
        compute_pressure_eos(rho=rho_good, rho0=1000.0, k=0.0)

    with pytest.raises(ValueError):
        compute_pressure_eos(rho=rho_good, rho0=1000.0, k=-1.0)

    rho_not_1d = np.array([[1000.0, 1001.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        compute_pressure_eos(rho=rho_not_1d, rho0=1000.0, k=1.0)


