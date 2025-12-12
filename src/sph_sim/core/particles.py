"""
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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleSet2D:
    """
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
    """

    # Position (2D)
    x: np.ndarray
    y: np.ndarray

    # Geschwindigkeit (2D)
    vx: np.ndarray
    vy: np.ndarray

    # Zustand
    rho: np.ndarray
    p: np.ndarray
    m: np.ndarray

    def __post_init__(self) -> None:
        """
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
        """

        # --- Schritt 1: Alles in float64-NumPy-Arrays umwandeln -----------------
        # Wir machen das explizit Feld für Feld, damit es für Anfänger nachvollziehbar bleibt.
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.vx = np.asarray(self.vx, dtype=np.float64)
        self.vy = np.asarray(self.vy, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        self.p = np.asarray(self.p, dtype=np.float64)
        self.m = np.asarray(self.m, dtype=np.float64)

        # --- Schritt 2: Dimensionen prüfen -------------------------------------
        # In NumPy ist `ndim` die Anzahl der Achsen (1D → ndim == 1).
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

        # --- Schritt 3: Längen prüfen ------------------------------------------
        # Alle Arrays müssen dieselbe Länge N haben (N = Anzahl Partikel).
        lengths = {name: arr.shape[0] for name, arr in arrays.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) != 1:
            # Wir geben alle Längen aus, damit der Fehler sofort verständlich ist.
            parts = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(
                "Alle Partikelfelder müssen dieselbe Länge haben (gleiche Anzahl Partikel). "
                f"Gefundene Längen: {parts}."
            )

    @property
    def n(self) -> int:
        """Anzahl der Partikel."""

        # `x` ist garantiert 1D und hat dieselbe Länge wie alle anderen Felder.
        return int(self.x.shape[0])


def initialize_particles_cube(
    L: float,
    dx: float,
    rho0: float,
    mass_per_particle: float,
) -> ParticleSet2D:
    """
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
    """

    # --- Validierung der Eingaben ---------------------------------------------
    # Wir machen die Checks bewusst früh, damit Fehler klar und schnell sichtbar sind.
    if L <= 0.0:
        raise ValueError("L muss > 0 sein.")
    if dx <= 0.0:
        raise ValueError("dx muss > 0 sein.")
    if rho0 <= 0.0:
        raise ValueError("rho0 muss > 0 sein.")
    if mass_per_particle <= 0.0:
        raise ValueError("mass_per_particle muss > 0 sein.")

    # Diese Bedingung verhindert “fast leere” Initialisierungen:
    # - Wenn dx >= L, dann gibt es in einer Dimension höchstens 1 Punkt (oder 0 wegen Rundung).
    # - Für viele SPH-Setups ist das unbrauchbar.
    if dx >= L:
        raise ValueError("dx muss kleiner als L sein (dx < L), sonst entstehen zu wenige Partikel.")

    # --- 1D-Koordinaten erzeugen ----------------------------------------------
    # `np.arange(0.0, L, dx)` liefert Werte < L in Schritten von dx.
    coords_1d = np.arange(0.0, L, dx, dtype=np.float64)

    # Wenn L und dx ungünstig sind (z.B. extrem klein / numerische Probleme),
    # kann das theoretisch leer sein. Dann erklären wir den Fehler sauber.
    if coords_1d.size == 0:
        raise ValueError(
            "Es konnten keine Gitterkoordinaten erzeugt werden: "
            "coords_1d ist leer. Prüfe L und dx (z.B. L zu klein oder dx zu groß)."
        )

    # --- 2D-Gitter aus 1D-Koordinaten bauen -----------------------------------
    # `indexing="xy"` ist das übliche kartesische Koordinatensystem:
    # X variiert in der zweiten Achse, Y in der ersten.
    X, Y = np.meshgrid(coords_1d, coords_1d, indexing="xy")

    # --- In SoA-Listenform bringen --------------------------------------------
    # `ravel()` liefert eine 1D-Ansicht (wenn möglich) oder eine Kopie.
    x = X.ravel()
    y = Y.ravel()

    # Anzahl Partikel ist jetzt einfach die Länge von x (und y).
    n = int(x.size)

    # --- Initialzustände als 1D-Arrays erzeugen --------------------------------
    # Wir nutzen `np.full`, damit es super klar ist: alle Werte gleich.
    vx = np.full(n, 0.0, dtype=np.float64)
    vy = np.full(n, 0.0, dtype=np.float64)
    rho = np.full(n, float(rho0), dtype=np.float64)
    p = np.full(n, 0.0, dtype=np.float64)
    m = np.full(n, float(mass_per_particle), dtype=np.float64)

    # ParticleSet2D übernimmt die restliche Validierung (1D, gleiche Länge, float64).
    return ParticleSet2D(x=x, y=y, vx=vx, vy=vy, rho=rho, p=p, m=m)


