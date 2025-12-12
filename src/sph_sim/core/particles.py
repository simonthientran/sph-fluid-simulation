"""
DE:
Partikel-Datenstrukturen und Initialisierung für eine 2D-SPH-Simulation.

SPH (Smoothed Particle Hydrodynamics) ist ein Verfahren, um Flüssigkeiten (oder auch Gase)
zu simulieren, ohne ein festes Gitter (wie bei vielen klassischen CFD-Methoden) zu benutzen.
Stattdessen wird die Flüssigkeit als **Menge vieler Partikel** beschrieben.

Wichtige Idee (sehr kurz, aber entscheidend):
- In SPH folgen wir einer **Lagrange-Sicht**: Wir “kleben” an der Flüssigkeit und beobachten,
  wie sich einzelne Materialpunkte bewegen.
- Ein SPH-Partikel ist kein “Kügelchen” wie bei Murmeln, sondern eher ein **Träger von
  Zustandsgrößen** (Position, Geschwindigkeit, Dichte, Druck, Masse, ...), der einen kleinen
  Teil des Fluids repräsentiert.

Warum diese Datei wichtig ist:
- In fast allen SPH-Algorithmen rechnen wir später sehr oft über alle Partikel.
- Deshalb ist ein gutes Speicherlayout entscheidend, damit der Code schnell ist und später
  auch gut auf GPU/Numba/CUDA übertragbar ist.

Wir verwenden deshalb ein Speicherlayout namens **Struct-of-Arrays (SoA)**:
- Statt pro Partikel ein Objekt zu haben (AoS = Array-of-Structs), speichern wir pro Feld
  **ein Array**:
  - x[i], y[i]           → Position des i-ten Partikels
  - vx[i], vy[i]         → Geschwindigkeit
  - rho[i]               → Dichte
  - p[i]                 → Druck
  - m[i]                 → Masse

Vorteile von SoA (intuitiv):
- NumPy kann sehr effizient mit zusammenhängenden Arrays rechnen (Vektorisierung).
- CPU-Caches werden besser genutzt (oft werden z.B. nur Positionen gebraucht → nur x/y lesen).
- Für GPU ist SoA oft ein natürlicheres Layout (coalesced memory access).

Diese Datei stellt zwei Bausteine bereit:
1) `ParticleSet2D`: Eine Dataclass, die alle Partikelfelder (SoA) hält und streng validiert.
2) `initialize_particles_cube(...)`: Eine einfache Start-Konfiguration: gleichmäßiges 2D-Gitter
   von Partikeln in einem Quadrat [0, L) × [0, L).

EN:
Particle data structures and initialization for a 2D SPH simulation.

SPH (Smoothed Particle Hydrodynamics) is a method to simulate fluids (or gases)
without using a fixed grid (as in many classical CFD methods).
Instead, the fluid is described as a **set of many particles**.

Key idea (very short, but essential):
- In SPH we follow a **Lagrangian view**: we “stick” to the fluid and observe
  how individual material points move.
- An SPH particle is not a “little ball” like a marble, but rather a **carrier of
  state quantities** (position, velocity, density, pressure, mass, ...), representing a small
  part of the fluid.

Why this file is important:
- In almost all SPH algorithms, we will later compute very often over all particles.
- Therefore, a good memory layout is crucial so that the code is fast and later
  is also easy to port to GPU/Numba/CUDA.

Therefore, we use a memory layout called **Struct-of-Arrays (SoA)**:
- Instead of having one object per particle (AoS = Array-of-Structs), we store one array
  **per field**:
  - x[i], y[i]           → position of the i-th particle
  - vx[i], vy[i]         → velocity
  - rho[i]               → density
  - p[i]                 → pressure
  - m[i]                 → mass

Advantages of SoA (intuitively):
- NumPy can compute very efficiently with contiguous arrays (vectorization).
- CPU caches are used more effectively (often only positions are needed → read only x/y).
- For GPUs, SoA is often a more natural layout (coalesced memory access).

This file provides two building blocks:
1) `ParticleSet2D`: a dataclass that holds all particle fields (SoA) and validates them strictly.
2) `initialize_particles_cube(...)`: a simple initial configuration: a uniform 2D grid
   of particles in a square [0, L) × [0, L).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleSet2D:
    """
    DE:
    Container für ein 2D-Partikelset im Struct-of-Arrays-Layout (SoA).

    Für Anfänger (warum so streng?):
    - In einer Simulation ist es extrem wichtig, dass alle Felder dieselbe Länge haben:
      Das i-te Element in jedem Array gehört zum selben Partikel.
    - Wir erzwingen außerdem `float64`, damit spätere Berechnungen konsistent sind
      (kein Mix aus float32/float64) und NumPy-Operationen stabiler/vorhersagbarer bleiben.

    Felder (alle 1D, Länge N):
    - x, y  : Position
    - vx, vy: Geschwindigkeit
    - rho   : Dichte
    - p     : Druck
    - m     : Masse (typisch konstant pro Partikel, aber als Array gespeichert für Flexibilität)

    EN:
    Container for a 2D particle set in a struct-of-arrays layout (SoA).

    For beginners (why so strict?):
    - In a simulation, it is extremely important that all fields have the same length:
      the i-th element in each array belongs to the same particle.
    - We also enforce `float64` so later computations are consistent
      (no mix of float32/float64) and NumPy operations remain more stable/predictable.

    Fields (all 1D, length N):
    - x, y  : position
    - vx, vy: velocity
    - rho   : density
    - p     : pressure
    - m     : mass (typically constant per particle, but stored as an array for flexibility)
    """

    # Position (2D)
    x: np.ndarray
    y: np.ndarray

    # --- Deutsch ---
    # Geschwindigkeit (2D)
    #
    # --- English ---
    # Velocity (2D)
    vx: np.ndarray
    vy: np.ndarray

    # --- Deutsch ---
    # Zustand
    #
    # --- English ---
    # State
    rho: np.ndarray
    p: np.ndarray
    m: np.ndarray

    def __post_init__(self) -> None:
        """
        DE:
        Validierung und Normalisierung der Eingaben nach dem Erzeugen der Dataclass.

        Was passiert hier genau?
        1) Wir konvertieren alle Eingaben zu NumPy-Arrays mit dtype float64.
           - `np.asarray(...)` akzeptiert z.B. Listen oder andere Array-Typen.
           - `dtype=np.float64` erzwingt einen einheitlichen Datentyp.
        2) Wir prüfen, dass alle Arrays 1D sind.
           - In diesem Projekt repräsentiert ein Partikelset eine Liste von Partikeln.
           - Mehrdimensionale Arrays wären hier ein Fehler (z.B. ein 2D-Gitterarray statt Liste).
        3) Wir prüfen, dass alle Arrays dieselbe Länge haben.
           - Nur dann ist die Zuordnung “Index i = Partikel i” eindeutig.

        Bei Problemen werfen wir `ValueError` mit einer klaren Fehlermeldung.

        EN:
        Validation and normalization of the inputs after creating the dataclass.

        What happens here exactly?
        1) We convert all inputs to NumPy arrays with dtype float64.
           - `np.asarray(...)` accepts, for example, lists or other array types.
           - `dtype=np.float64` enforces a consistent data type.
        2) We check that all arrays are 1D.
           - In this project, a particle set represents a list of particles.
           - Multidimensional arrays would be an error here (e.g., a 2D grid array instead of a list).
        3) We check that all arrays have the same length.
           - Only then is the mapping “index i = particle i” unambiguous.

        If there are problems, we raise `ValueError` with a clear error message.
        """

        # --- Deutsch ---
        # --- Schritt 1: Alles in float64-NumPy-Arrays umwandeln -----------------
        # Wir machen das explizit Feld für Feld, damit es für Anfänger nachvollziehbar bleibt.
        #
        # --- English ---
        # --- Step 1: convert everything to float64 NumPy arrays ----------------
        # We do this explicitly field by field so it is easy for beginners to follow.
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.vx = np.asarray(self.vx, dtype=np.float64)
        self.vy = np.asarray(self.vy, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        self.p = np.asarray(self.p, dtype=np.float64)
        self.m = np.asarray(self.m, dtype=np.float64)

        # --- Deutsch ---
        # --- Schritt 2: Dimensionen prüfen -------------------------------------
        # In NumPy ist `ndim` die Anzahl der Achsen (1D → ndim == 1).
        #
        # --- English ---
        # --- Step 2: check dimensions -----------------------------------------
        # In NumPy, `ndim` is the number of axes (1D → ndim == 1).
        arrays = {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "rho": self.rho,
            "p": self.p,
            "m": self.m,
        }

        for name, arr in arrays.items():
            if arr.ndim != 1:
                raise ValueError(
                    f"'{name}' muss ein 1D-Array sein (ndim == 1), aber ndim == {arr.ndim}."
                )

        # --- Deutsch ---
        # --- Schritt 3: Längen prüfen ------------------------------------------
        # Alle Arrays müssen dieselbe Länge N haben (N = Anzahl Partikel).
        #
        # --- English ---
        # --- Step 3: check lengths --------------------------------------------
        # All arrays must have the same length N (N = number of particles).
        lengths = {name: arr.shape[0] for name, arr in arrays.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) != 1:
            # --- Deutsch ---
            # Wir geben alle Längen aus, damit der Fehler sofort verständlich ist.
            #
            # --- English ---
            # We output all lengths so the error is immediately understandable.
            parts = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(
                "Alle Partikelfelder müssen dieselbe Länge haben (gleiche Anzahl Partikel). "
                f"Gefundene Längen: {parts}."
            )

    @property
    def n(self) -> int:
        """
        DE:
        Anzahl der Partikel.

        EN:
        Number of particles.
        """

        # --- Deutsch ---
        # `x` ist garantiert 1D und hat dieselbe Länge wie alle anderen Felder.
        #
        # --- English ---
        # `x` is guaranteed to be 1D and has the same length as all other fields.
        return int(self.x.shape[0])


def initialize_particles_cube(
    L: float,
    dx: float,
    rho0: float,
    mass_per_particle: float,
) -> ParticleSet2D:
    """
    DE:
    Erzeuge ein gleichmäßiges 2D-Gitter von Partikeln in einem quadratischen Bereich.

    Ziel (intuitiv):
    - Wir wollen eine “Startwolke” an Partikeln, die ein Quadrat füllt.
    - Diese Partikel können später z.B. ein Flüssigkeitsbecken, einen Block, etc. darstellen.

    Geometrie:
    - Bereich: x in [0, L) und y in [0, L)
      Das bedeutet:
      - 0 ist dabei, L ist nicht dabei (halboffenes Intervall).
      - Das verhindert Partikel genau auf der rechten/oberen Kante bei L.
    - Abstand: dx
      Das ist die Gitterweite. Kleinere dx → mehr Partikel.

    Vorgehensweise (bewusst simpel und nachvollziehbar):
    1) `coords_1d = np.arange(0.0, L, dx)`
       - erzeugt 1D-Koordinaten: 0, dx, 2*dx, ..., < L
    2) `X, Y = np.meshgrid(coords_1d, coords_1d, indexing="xy")`
       - erzeugt ein 2D-Gitter aus allen Kombinationen (x, y)
    3) `x = X.ravel()`, `y = Y.ravel()`
       - macht daraus wieder 1D-Listen von Partikeln (SoA-Format)

    Initialwerte:
    - vx, vy: 0.0 (Fluid ruht am Anfang)
    - rho   : rho0 (Startdichte überall gleich)
    - p     : 0.0 (Startdruck überall 0, häufige einfache Wahl)
    - m     : mass_per_particle (alle Partikel gleiche Masse)

    Parameter
    - L: Seitenlänge des Quadrats (muss > 0)
    - dx: Abstand zwischen Partikeln (muss > 0 und typischerweise < L)
    - rho0: Startdichte (muss > 0)
    - mass_per_particle: Masse pro Partikel (muss > 0)

    Rückgabe
    - Ein `ParticleSet2D` mit SoA-Feldern (alles float64, strikt validiert).

    EN:
    Create a uniform 2D grid of particles in a square domain.

    Goal (intuitive):
    - We want an “initial cloud” of particles that fills a square.
    - These particles can later represent, for example, a fluid tank, a block, etc.

    Geometry:
    - Domain: x in [0, L) and y in [0, L)
      This means:
      - 0 is included, L is not included (half-open interval).
      - This prevents particles exactly on the right/top boundary at L.
    - Spacing: dx
      This is the grid spacing. Smaller dx → more particles.

    Procedure (deliberately simple and easy to follow):
    1) `coords_1d = np.arange(0.0, L, dx)`
       - creates 1D coordinates: 0, dx, 2*dx, ..., < L
    2) `X, Y = np.meshgrid(coords_1d, coords_1d, indexing="xy")`
       - creates a 2D grid from all combinations (x, y)
    3) `x = X.ravel()`, `y = Y.ravel()`
       - turns them back into 1D lists of particles (SoA format)

    Initial values:
    - vx, vy: 0.0 (fluid is at rest initially)
    - rho   : rho0 (initial density is uniform everywhere)
    - p     : 0.0 (initial pressure is 0 everywhere, a common simple choice)
    - m     : mass_per_particle (all particles have the same mass)

    Parameters
    - L: side length of the square (must be > 0)
    - dx: spacing between particles (must be > 0 and typically < L)
    - rho0: initial density (must be > 0)
    - mass_per_particle: mass per particle (must be > 0)

    Returns
    - A `ParticleSet2D` with SoA fields (all float64, strictly validated).
    """

    # --- Deutsch ---
    # --- Validierung der Eingaben ---------------------------------------------
    # Wir machen die Checks bewusst früh, damit Fehler klar und schnell sichtbar sind.
    #
    # --- English ---
    # --- Input validation -----------------------------------------------------
    # We perform the checks early so errors are clear and appear quickly.
    if L <= 0.0:
        raise ValueError("L muss > 0 sein.")
    if dx <= 0.0:
        raise ValueError("dx muss > 0 sein.")
    if rho0 <= 0.0:
        raise ValueError("rho0 muss > 0 sein.")
    if mass_per_particle <= 0.0:
        raise ValueError("mass_per_particle muss > 0 sein.")

    # --- Deutsch ---
    # Diese Bedingung verhindert “fast leere” Initialisierungen:
    # - Wenn dx >= L, dann gibt es in einer Dimension höchstens 1 Punkt (oder 0 wegen Rundung).
    # - Für viele SPH-Setups ist das unbrauchbar.
    #
    # --- English ---
    # This condition prevents “almost empty” initializations:
    # - If dx >= L, then in one dimension there is at most 1 point (or 0 due to rounding).
    # - For many SPH setups, this is unusable.
    if dx >= L:
        raise ValueError("dx muss kleiner als L sein (dx < L), sonst entstehen zu wenige Partikel.")

    # --- Deutsch ---
    # --- 1D-Koordinaten erzeugen ----------------------------------------------
    # `np.arange(0.0, L, dx)` liefert Werte < L in Schritten von dx.
    #
    # --- English ---
    # --- Generate 1D coordinates ---------------------------------------------
    # `np.arange(0.0, L, dx)` returns values < L in steps of dx.
    coords_1d = np.arange(0.0, L, dx, dtype=np.float64)

    # --- Deutsch ---
    # Wenn L und dx ungünstig sind (z.B. extrem klein / numerische Probleme),
    # kann das theoretisch leer sein. Dann erklären wir den Fehler sauber.
    #
    # --- English ---
    # If L and dx are unfavorable (e.g., extremely small / numerical issues),
    # it can theoretically be empty. Then we explain the error clearly.
    if coords_1d.size == 0:
        raise ValueError(
            "Es konnten keine Gitterkoordinaten erzeugt werden: "
            "coords_1d ist leer. Prüfe L und dx (z.B. L zu klein oder dx zu groß)."
        )

    # --- Deutsch ---
    # --- 2D-Gitter aus 1D-Koordinaten bauen -----------------------------------
    # `indexing="xy"` ist das übliche kartesische Koordinatensystem:
    # X variiert in der zweiten Achse, Y in der ersten.
    #
    # --- English ---
    # --- Build 2D grid from 1D coordinates -----------------------------------
    # `indexing="xy"` is the standard Cartesian coordinate convention:
    # X varies along the second axis, Y along the first axis.
    X, Y = np.meshgrid(coords_1d, coords_1d, indexing="xy")

    # --- Deutsch ---
    # --- In SoA-Listenform bringen --------------------------------------------
    # `ravel()` liefert eine 1D-Ansicht (wenn möglich) oder eine Kopie.
    #
    # --- English ---
    # --- Convert to SoA list form --------------------------------------------
    # `ravel()` returns a 1D view (if possible) or a copy.
    x = X.ravel()
    y = Y.ravel()

    # --- Deutsch ---
    # Anzahl Partikel ist jetzt einfach die Länge von x (und y).
    #
    # --- English ---
    # The number of particles is now simply the length of x (and y).
    n = int(x.size)

    # --- Deutsch ---
    # --- Initialzustände als 1D-Arrays erzeugen --------------------------------
    # Wir nutzen `np.full`, damit es super klar ist: alle Werte gleich.
    #
    # --- English ---
    # --- Create initial states as 1D arrays ----------------------------------
    # We use `np.full` to make it very clear: all values are the same.
    vx = np.full(n, 0.0, dtype=np.float64)
    vy = np.full(n, 0.0, dtype=np.float64)
    rho = np.full(n, float(rho0), dtype=np.float64)
    p = np.full(n, 0.0, dtype=np.float64)
    m = np.full(n, float(mass_per_particle), dtype=np.float64)

    # --- Deutsch ---
    # ParticleSet2D übernimmt die restliche Validierung (1D, gleiche Länge, float64).
    #
    # --- English ---
    # ParticleSet2D performs the remaining validation (1D, same length, float64).
    return ParticleSet2D(x=x, y=y, vx=vx, vy=vy, rho=rho, p=p, m=m)


