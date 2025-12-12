"""
DE:
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

EN:
SPH kernels (kernel functions) for Smoothed Particle Hydrodynamics (SPH).

What is an SPH kernel?
In SPH, we replace “hard” point queries (e.g., a Dirac delta distribution) with a
a smooth weighting function \(W(r, h)\).

- Each particle influences its surroundings.
- How strong this influence is depends primarily on the distance \(r\).
- The kernel \(W(r, h)\) tells us: “How large is the weight of a neighbor at distance r?”

Parameter \(h\) (smoothing length / smoothing length):
- \(h\) defines the influence radius.
- Small \(h\) → very local influence, fewer neighbors
- Large \(h\) → wider influence, more neighbors

What do we use the Poly6 kernel for?
The Poly6 kernel is often used for **density estimation**.
Why?
- It is **smooth** (has no “corners”/jumps) and decays gently.
- “Smooth” is important because in SPH we often need derivatives (gradients).
  If a function is smooth, derivatives behave more stably and the simulation
  becomes less “jittery”.

What does "compact support" (compact support) mean?
Many SPH kernels have *compact support*:

- For \(r > h\), the kernel is exactly **0**.
- That means: a particle influences only neighbors within the radius \(h\).

Why is this important?
- Performance: we do not need to compute over all particles, only over neighbors.
- Physical/algorithmic control: the influence is locally bounded.

This file:
Here we implement the Poly6 kernel \(W(r, h)\) as a function that can handle both
scalars and NumPy arrays.
This is convenient because later we often want to compute many distances at once
(vectorized in NumPy, without Python loops).
"""

from __future__ import annotations

import numpy as np


def poly6_kernel(r: float | np.ndarray, h: float) -> float | np.ndarray:
    """
    DE:
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

    EN:
    Poly6 kernel \(W(r, h)\) (here in 2D) for distance-based weighting.

    Mathematical definition:
    - For \(0 \\le r \\le h\):
    - For \(r > h\):
    2D normalization constant:
    - We use:

    Why is the Poly6 kernel "smooth"?
    - The expression \((h^2 - r^2)^3\) is a polynomial.
    - Polynomials are smooth (differentiable) everywhere.
    - At the boundary \(r = h\), the value becomes 0 and the transition is smooth.

    Why do we support arrays?
    - In SPH, we often compute many distances at once (to many neighbors).
    - NumPy can do this very fast when we use arrays (vectorization).

    Parameters
    - r: distance(s) between two particles (scalar or NumPy array)
    - h: smoothing length (must be > 0)

    Returns
    - Scalar input → float
    - Array input → NumPy array of the same shape
    """

    # --- Deutsch ---
    # --- Validierung: h muss sinnvoll sein ------------------------------------
    # h ist ein Längenmaß. Ein nicht-positives h würde keinen physikalischen Sinn ergeben
    # und würde außerdem zu Division durch 0 oder negativen Potenzen führen.
    #
    # --- English ---
    # --- Validation: h must be meaningful ------------------------------------
    # h is a length scale. A non-positive h would not make physical sense
    # and would also lead to division by 0 or negative powers.
    if h <= 0.0:
        raise ValueError("h muss > 0 sein.")

    # --- Deutsch ---
    # --- r kann Skalar oder Array sein ----------------------------------------
    # Wir wandeln r immer in ein NumPy-Array (float64) um.
    # So können wir den gleichen Codepfad für Skalar und Array benutzen.
    #
    # --- English ---
    # --- r can be scalar or array --------------------------------------------
    # We always convert r into a NumPy array (float64).
    # This allows us to use the same code path for scalar and array inputs.
    r_arr = np.asarray(r, dtype=np.float64)

    # --- Deutsch ---
    # Merke dir, ob die Eingabe ein Skalar war:
    # - Bei Skalar wird `np.asarray` zu einem 0D-Array (ndim == 0).
    #
    # --- English ---
    # Remember whether the input was a scalar:
    # - For a scalar, `np.asarray` becomes a 0D array (ndim == 0).
    is_scalar_input = r_arr.ndim == 0

    # --- Deutsch ---
    # --- Distanz darf nicht negativ sein --------------------------------------
    # Distanzen sind per Definition nicht-negativ. Falls jemand negative Werte übergibt,
    # interpretieren wir sie als "Betrag der Distanz".
    #
    # --- English ---
    # --- Distance must not be negative ---------------------------------------
    # Distances are non-negative by definition. If someone passes negative values,
    # we interpret them as the "magnitude of the distance".
    r_abs = np.abs(r_arr)

    # --- Deutsch ---
    # --- Ergebnis-Array vorbereiten -------------------------------------------
    # Standardwert ist 0: Für r > h (außerhalb des compact support) ist der Kernel 0.
    #
    # --- English ---
    # --- Prepare result array -------------------------------------------------
    # The default value is 0: for r > h (outside the compact support), the kernel is 0.
    W = np.zeros_like(r_abs, dtype=np.float64)

    # --- Deutsch ---
    # --- Maske: Welche Werte liegen innerhalb des Einflussradius? --------------
    # inside_mask ist True genau dort, wo r <= h gilt.
    #
    # --- English ---
    # --- Mask: which values are within the influence radius? -----------------
    # inside_mask is True exactly where r <= h holds.
    inside_mask = r_abs <= h

    # --- Deutsch ---
    # --- Kernel nur für r <= h berechnen --------------------------------------
    # Damit vermeiden wir unnötige Rechnungen und setzen automatisch W=0 außerhalb.
    #
    # Achtung: Wir rechnen mit h^8. Das ist okay, solange h > 0 ist (haben wir geprüft).
    #
    # --- English ---
    # --- Compute kernel only for r <= h --------------------------------------
    # This avoids unnecessary computation and automatically keeps W=0 outside.
    #
    # Note: we compute with h^8. This is fine as long as h > 0 (which we validated).
    C = 4.0 / (np.pi * (h**8))

    # --- Deutsch ---
    # Für alle r innerhalb:
    # (h^2 - r^2)^3
    # Wir berechnen das nur für die "inside"-Elemente.
    #
    # --- English ---
    # For all r inside:
    # We compute this only for the "inside" elements.
    h2 = h * h
    r2_inside = r_abs[inside_mask] ** 2
    W[inside_mask] = C * (h2 - r2_inside) ** 3

    # --- Deutsch ---
    # --- Rückgabe: Typ so lassen wie Eingabe ----------------------------------
    # Wenn die Eingabe ein Skalar war, geben wir wieder einen Python-float zurück.
    #
    # --- English ---
    # --- Return: keep the type consistent with the input ----------------------
    # If the input was a scalar, we return a Python float again.
    if is_scalar_input:
        return float(W.item())

    # --- Deutsch ---
    # Sonst geben wir das NumPy-Array zurück.
    #
    # --- English ---
    # Otherwise, we return the NumPy array.
    return W


def spiky_kernel_gradient(dx: float, dy: float, h: float) -> tuple[float, float]:
    """
    DE:
    Gradient des Spiky-Kernels in 2D (als Vektor) für Druckkräfte in SPH.

    Warum ist das ein Vektor?
    -------------------------
    Ein Gradient ist die Ableitung einer Skalarfunktion nach dem Ort.
    In 2D bedeutet das:

        ∇W = (∂W/∂x, ∂W/∂y)

    Also: zwei Komponenten (x- und y-Richtung).

    Warum brauchen wir den Spiky-Gradienten?
    ----------------------------------------
    In vielen SPH-Formulierungen werden **Druckkräfte** mit einem Kernel-Gradienten
    berechnet. Der Spiky-Kernel (bzw. sein Gradient) ist dafür sehr üblich, weil er
    sich numerisch gut verhält (starker, “spitzer” Gradient nahe am Partikel).

    Sonderfall r == 0
    -----------------
    Wir brauchen die Richtung (dx/r, dy/r). Dafür müssten wir durch r teilen.
    Wenn r == 0 wäre das eine Division durch 0.
    Physikalisch ist die Richtung bei r==0 auch “undefiniert” (keine bevorzugte Richtung).
    Daher geben wir in diesem Fall (0, 0) zurück.

    Support / compact support
    -------------------------
    Wie bei vielen SPH-Kernen gilt: für r > h ist der Beitrag 0.
    Deshalb geben wir außerhalb des Radius h ebenfalls (0, 0) zurück.

    Verwendete Form (typisch, 2D)
    -----------------------------
    - Konstante:

        C = -30 / (pi * h^5)

      (negativ, weil der Kernel mit r abnimmt und der Gradient nach “außen” zeigt)

    - Betrag proportional zu (h - r)^2
    - Richtung = (dx/r, dy/r)

    Implementiert als:

        factor = C * (h - r)^2 / r
        gx = factor * dx
        gy = factor * dy

    EN:
    Gradient of the Spiky kernel in 2D (as a vector) for pressure forces in SPH.

    Why is this a vector?
    ---------------------
    A gradient is the derivative of a scalar function with respect to position.
    In 2D this means:

        ∇W = (∂W/∂x, ∂W/∂y)

    So: two components (x and y direction).

    Why do we need the Spiky gradient?
    ----------------------------------
    In many SPH formulations, **pressure forces** are computed using a kernel gradient.
    The Spiky kernel (or rather its gradient) is commonly used because it behaves well
    numerically (a strong, “spiky” gradient close to the particle).

    Special case r == 0
    -------------------
    We need the direction (dx/r, dy/r), which requires division by r.
    If r == 0, this would be a division by zero.
    Physically, the direction at r==0 is also “undefined” (no preferred direction).
    Therefore we return (0, 0) in this case.

    Support / compact support
    -------------------------
    As with many SPH kernels: for r > h the contribution is 0.
    Therefore outside the radius h we also return (0, 0).

    Used form (typical, 2D)
    -----------------------
    - Constant:

        C = -30 / (pi * h^5)

      (negative because the kernel decreases with r and the gradient points “outwards”)

    - Magnitude proportional to (h - r)^2
    - Direction = (dx/r, dy/r)

    Implemented as:

        factor = C * (h - r)^2 / r
        gx = factor * dx
        gy = factor * dy
    """

    # --- Deutsch ---
    # Validierung: h muss > 0 sein.
    #
    # --- English ---
    # Validation: h must be > 0.
    if h <= 0.0:
        raise ValueError("h muss > 0 sein.")

    # --- Deutsch ---
    # Distanz r aus dem Differenzvektor (dx, dy).
    #
    # --- English ---
    # Distance r from the difference vector (dx, dy).
    r = float(np.sqrt(float(dx) * float(dx) + float(dy) * float(dy)))

    # --- Deutsch ---
    # Sonderfälle:
    # - r == 0: Richtung ist nicht definiert (Division durch 0 vermeiden).
    # - r > h: außerhalb des Supports → Gradient ist 0.
    #
    # --- English ---
    # Special cases:
    # - r == 0: direction is undefined (avoid division by 0).
    # - r > h: outside support → gradient is 0.
    if r == 0.0 or r > float(h):
        return (0.0, 0.0)

    # --- Deutsch ---
    # 2D-Konstante für den Spiky-Gradienten:
    # Negativ, weil der Kernel mit r abnimmt.
    #
    # --- English ---
    # 2D constant for the spiky gradient:
    # Negative because the kernel decreases with r.
    C = 30.0 / (np.pi * (float(h) ** 5))

    # --- Deutsch ---
    # Faktor:
    # - (h - r)^2 liefert die typische “spiky” Form.
    # - / r sorgt für die Richtungskomponenten (dx/r, dy/r) in einem Schritt.
    #
    # --- English ---
    # Factor:
    # - (h - r)^2 gives the typical “spiky” shape.
    # - / r applies the direction components (dx/r, dy/r) in one step.
    factor = float(C) * (float(h) - r) ** 2 / r

    gx = factor * float(dx)
    gy = factor * float(dy)

    return (float(gx), float(gy))


