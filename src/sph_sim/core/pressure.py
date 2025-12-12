"""
DE:
EOS-basierte Druckberechnung für SPH (Smoothed Particle Hydrodynamics).

Was bedeutet "EOS"?
-------------------
EOS steht für "Equation of State" (= Zustandsgleichung).
Eine Zustandsgleichung ist eine Beziehung, die aus einer Zustandsgröße (z.B. Dichte rho)
eine andere Zustandsgröße (z.B. Druck p) berechnet.

Warum brauchen wir Druck in SPH?
--------------------------------
In vielen SPH-Formulierungen erzeugt der Druck Kräfte, die:
- eine Flüssigkeit "zusammendrücken" bzw. "auseinander drücken" können,
- Dichte-Schwankungen begrenzen,
- und grob gesagt "Inkompressibilität" annähern (bei Flüssigkeiten möchten wir oft,
  dass rho ungefähr konstant bleibt).

Welche EOS nutzen wir hier?
---------------------------
Für den Einstieg verwenden wir eine sehr einfache, lineare EOS:

    p = k * (rho - rho0)

- rho: aktuelle Dichte (pro Partikel)
- rho0: Referenzdichte / Ruhedichte (Sollwert)
- k: Steifigkeit (wie stark Druck auf Dichte-Abweichungen reagiert)

Warum gibt es rho0 und k?
-------------------------
- rho0 setzt den "Nullpunkt" des Drucks:
  Wenn rho == rho0, dann p == 0.
- k bestimmt die "Härte" des Mediums:
  Größeres k → schon kleine Dichteabweichungen erzeugen großen Druck.

Warum clamping (p >= 0)?
------------------------
In einfachen Lern-Setups kann negativer Druck zu Instabilitäten führen
(z.B. "Zugspannungen" / tensile instability).
Darum wird oft für Anfänger-Experimente p nach unten bei 0 begrenzt:

    p = max(p, 0)

Wichtig:
- Das ist ein Stabilitäts-Trick, nicht die einzige "richtige" Physik.
- In fortgeschritteneren Setups kann man negative Drücke zulassen oder anders behandeln.

EN:
EOS-based pressure computation for SPH (Smoothed Particle Hydrodynamics).

What does "EOS" mean?
---------------------
EOS stands for "Equation of State".
An equation of state is a relationship that computes one state variable (e.g., density rho)
from another state variable (e.g., pressure p).

Why do we need pressure in SPH?
-------------------------------
In many SPH formulations, pressure generates forces that:
- can "compress" or "expand" a fluid,
- limit density fluctuations,
- and roughly approximate "incompressibility" (for liquids we often want rho to stay
  close to a constant value).

Which EOS do we use here?
-------------------------
For learning, we use a very simple linear EOS:

    p = k * (rho - rho0)

- rho: current density (per particle)
- rho0: reference/rest density (target value)
- k: stiffness (how strongly pressure reacts to density deviations)

Why do we have rho0 and k?
--------------------------
- rho0 sets the "zero point" of pressure:
  If rho == rho0, then p == 0.
- k controls the "stiffness" of the medium:
  Larger k → even small density deviations produce large pressure.

Why clamping (p >= 0)?
----------------------
In simple learning setups, negative pressure can cause instabilities
(e.g., "tensile instability").
Therefore, for beginner experiments, we often clamp p at 0 from below:

    p = max(p, 0)

Important:
- This is a stability trick, not the only "correct" physics.
- In more advanced setups you may allow negative pressures or treat them differently.
"""

from __future__ import annotations

import numpy as np


def compute_pressure_eos(
    rho: np.ndarray,
    rho0: float,
    k: float,
    clamp_negative: bool = True,
) -> np.ndarray:
    """
    DE:
    Berechne Druck p aus Dichte rho mit einer (linearen) Zustandsgleichung (EOS).

    Formel
    ------
        p = k * (rho - rho0)

    Optional (Stabilität):
    - Wenn clamp_negative=True, begrenzen wir nach unten:
        p = max(p, 0)

    Validierung
    ----------
    - rho muss ein 1D-Array sein (ein Wert pro Partikel).
    - rho0 muss > 0 sein (Referenzdichte).
    - k muss > 0 sein (Steifigkeit).

    Rückgabe
    --------
    - Neues NumPy-Array p (dtype float64), gleiche Form wie rho.
    - Wir verändern das Eingabearray rho nicht.

    EN:
    Compute pressure p from density rho using a (linear) equation of state (EOS).

    Formula
    -------
        p = k * (rho - rho0)

    Optional (stability):
    - If clamp_negative=True, we clamp from below:
        p = max(p, 0)

    Validation
    ----------
    - rho must be a 1D array (one value per particle).
    - rho0 must be > 0 (reference density).
    - k must be > 0 (stiffness).

    Returns
    -------
    - New NumPy array p (dtype float64), same shape as rho.
    - We do not modify the input array rho.
    """

    # --- Deutsch ---
    # rho in ein NumPy-Array (float64) umwandeln:
    # - So sind die Rechenregeln klar.
    # - Und wir bekommen eine konsistente Ausgabe (float64).
    #
    # --- English ---
    # Convert rho into a NumPy array (float64):
    # - This makes computation rules clear.
    # - And we get a consistent output type (float64).
    rho_arr = np.asarray(rho, dtype=np.float64)

    # --- Deutsch ---
    # Validierung: rho muss 1D sein.
    # (Ein Dichtewert pro Partikel → 1D-Array der Länge N.)
    #
    # --- English ---
    # Validation: rho must be 1D.
    # (One density value per particle → 1D array of length N.)
    if rho_arr.ndim != 1:
        raise ValueError("rho muss ein 1D-Array sein.")

    # --- Deutsch ---
    # Validierung: rho0 und k müssen > 0 sein.
    #
    # --- English ---
    # Validation: rho0 and k must be > 0.
    if float(rho0) <= 0.0:
        raise ValueError("rho0 muss > 0 sein.")
    if float(k) <= 0.0:
        raise ValueError("k muss > 0 sein.")

    # --- Deutsch ---
    # EOS: linearer Zusammenhang zwischen Dichteabweichung und Druck.
    #
    # --- English ---
    # EOS: linear relationship between density deviation and pressure.
    p = float(k) * (rho_arr - float(rho0))

    # --- Deutsch ---
    # Optional: negativer Druck wird auf 0 geklemmt.
    # Das ist ein Stabilitäts-Trick für einfache Lern-Setups.
    #
    # --- English ---
    # Optional: clamp negative pressure to 0.
    # This is a stability trick for simple learning setups.
    if clamp_negative:
        p = np.maximum(p, 0.0)

    # --- Deutsch ---
    # Wir geben ein neues Array zurück (kein inplace auf rho).
    #
    # --- English ---
    # We return a new array (no in-place modification of rho).
    return np.asarray(p, dtype=np.float64)


