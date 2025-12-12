"""
DE:
Zeitintegration (Integrator) für SPH-Partikel.

Worum geht es hier?
-------------------
In SPH berechnen wir zuerst **Beschleunigungen** (a) aus Kräften (z.B. Druckkräfte).
Diese Beschleunigungen müssen wir dann in **Geschwindigkeiten** (v) und schließlich in
**Positionen** (x) integrieren, damit sich die Partikel bewegen.

Dieses Modul enthält absichtlich nur einen sehr einfachen, gut erklärbaren Integrator:
**Semi-Implicit (Symplectic) Euler**.

Warum ist das wichtig?
----------------------
- Ohne Integration bleiben Partikel an Ort und Stelle, selbst wenn Kräfte existieren.
- Integration ist der Schritt, der aus "Kräften" eine "Bewegung" macht.

Warum Semi-Implicit Euler (für Anfänger)?
-----------------------------------------
Er ist extrem einfach, aber oft stabiler als der ganz naive "Explicit Euler":

1) v_{t+dt} = v_t + a_t * dt
2) x_{t+dt} = x_t + v_{t+dt} * dt

Der Unterschied zu "Explicit Euler" ist, dass wir beim Positionsschritt bereits die
aktualisierte Geschwindigkeit v_{t+dt} nutzen.

Wichtige Begriffe (sehr kurz):
------------------------------
- Beschleunigung a: "Änderungsrate" der Geschwindigkeit (Einheit: m/s^2)
- Geschwindigkeit v: "Änderungsrate" der Position (Einheit: m/s)
- Position x: Ort im Raum (Einheit: m)
- dt: Zeitschritt (Einheit: s)

Hinweis zu dt:
--------------
dt sollte klein sein, sonst kann die Simulation numerisch instabil werden.
Das ist kein "Physik-Problem", sondern ein numerisches Problem: wir nähern eine
kontinuierliche Zeitentwicklung nur mit kleinen Schritten an.

EN:
Time integration (integrator) for SPH particles.

What is this about?
In SPH we first compute **accelerations** (a) from forces (e.g., pressure forces).
We then have to integrate these accelerations into **velocities** (v) and finally into
**positions** (x), so that particles actually move.

This module intentionally contains only one very simple, easy-to-explain integrator:
**Semi-Implicit (Symplectic) Euler**.

Why is this important?
- Without integration, particles stay where they are, even if forces exist.
- Integration is the step that turns "forces" into "motion".

Why semi-implicit Euler (for beginners)?
It is extremely simple, but often more stable than fully naive "explicit Euler":

1) v_{t+dt} = v_t + a_t * dt
2) x_{t+dt} = x_t + v_{t+dt} * dt

The difference to "explicit Euler" is that the position update already uses the
updated velocity v_{t+dt}.

Key terms (very short):
- Acceleration a: "rate of change" of velocity (unit: m/s^2)
- Velocity v: "rate of change" of position (unit: m/s)
- Position x: location in space (unit: m)
- dt: time step (unit: s)

Note on dt:
dt should be small, otherwise the simulation can become numerically unstable.
This is not a "physics problem" but a numerical one: we approximate a continuous-time
evolution with small discrete steps.
"""

import numpy as np

from sph_sim.core.particles import ParticleSet2D


def step_semi_implicit_euler(
    particles: ParticleSet2D,
    ax: np.ndarray,
    ay: np.ndarray,
    dt: float,
) -> None:
    """
    DE:
    Führe einen Zeitschritt mit Semi-Implicit (Symplectic) Euler durch (in-place).

    Was macht diese Funktion?
    -------------------------
    Sie aktualisiert die Partikelzustände **direkt im ParticleSet2D**:

    1) Geschwindigkeit aus Beschleunigung:
       - v_x += a_x * dt
       - v_y += a_y * dt

    2) Position aus (neuer) Geschwindigkeit:
       - x += v_x * dt
       - y += v_y * dt

    Wichtig:
    - Die Arrays im `particles`-Objekt werden in-place verändert.
    - Es wird keine neue Partikelstruktur erzeugt.

    Warum zuerst v und dann x?
    --------------------------
    Semi-Implicit Euler nutzt die aktualisierte Geschwindigkeit v_{t+dt} für die
    Positionsänderung. Das ist oft stabiler als der reine Explicit Euler, der x mit
    der alten Geschwindigkeit v_t aktualisieren würde.

    Parameter:
    ----------
    particles:
        Partikelzustand (Position und Geschwindigkeit werden aktualisiert).
    ax, ay:
        Beschleunigungen pro Partikel (Shape: (N,)).
    dt:
        Zeitschritt. Muss > 0 sein.

    EN:
    Perform one time step with semi-implicit (symplectic) Euler (in-place).

    What does this function do?
    It updates the particle state **directly inside the ParticleSet2D**:

    1) Velocity from acceleration:
       - v_x += a_x * dt
       - v_y += a_y * dt

    2) Position from the (new) velocity:
       - x += v_x * dt
       - y += v_y * dt

    Important:
    - Arrays inside `particles` are modified in-place.
    - No new particle structure is created.

    Why update v first and then x?
    Semi-implicit Euler uses the updated velocity v_{t+dt} for the position update.
    This is often more stable than pure explicit Euler, which would update x using
    the old velocity v_t.

    Parameters:
    particles:
        Particle state (position and velocity are updated).
    ax, ay:
        Accelerations per particle (shape: (N,)).
    dt:
        Time step. Must be > 0.
    """

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Validierung: dt muss > 0 sein
    # -------------------------------------------------------------------------
    # Warum?
    # - dt ist ein Zeitintervall.
    # - dt <= 0 ergibt keinen sinnvollen "Vorwärts-Zeitschritt".
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Validation: dt must be > 0
    # -------------------------------------------------------------------------
    # Why?
    # - dt is a time interval.
    # - dt <= 0 does not represent a meaningful "forward time step".
    if float(dt) <= 0.0:
        raise ValueError("dt muss > 0 sein.")

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Validierung: ax und ay müssen 1D sein und die Länge N haben
    # -------------------------------------------------------------------------
    # Warum?
    # - Wir brauchen für jedes Partikel genau einen Beschleunigungswert pro Achse.
    # - Wenn die Shapes nicht passen, würden wir falsche Partikel aktualisieren oder
    #   NumPy würde Broadcasting machen (das wäre hier ein Bug).
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Validation: ax and ay must be 1D and have length N
    # -------------------------------------------------------------------------
    # Why?
    # - We need exactly one acceleration value per particle and axis.
    # - If shapes do not match, we would update wrong particles or NumPy would do
    #   broadcasting (which would be a bug here).
    ax_arr = np.asarray(ax, dtype=np.float64)
    ay_arr = np.asarray(ay, dtype=np.float64)

    if ax_arr.ndim != 1:
        raise ValueError("ax muss ein 1D-Array sein.")
    if ay_arr.ndim != 1:
        raise ValueError("ay muss ein 1D-Array sein.")

    N = int(particles.n)
    if ax_arr.shape != (N,):
        raise ValueError(f"ax muss Shape ({N},) haben, aber hat {ax_arr.shape}.")
    if ay_arr.shape != (N,):
        raise ValueError(f"ay muss Shape ({N},) haben, aber hat {ay_arr.shape}.")

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Schritt 1: Geschwindigkeit aktualisieren
    # -------------------------------------------------------------------------
    # Beschleunigung a sagt: "Wie schnell ändert sich die Geschwindigkeit?"
    #
    # Mathematisch:
    # - v(t+dt) = v(t) + a(t) * dt
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Step 1: update velocity
    # -------------------------------------------------------------------------
    # Acceleration a means: "How fast does the velocity change?"
    #
    # Mathematically:
    # - v(t+dt) = v(t) + a(t) * dt
    particles.vx += ax_arr * float(dt)
    particles.vy += ay_arr * float(dt)

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Schritt 2: Position aktualisieren (mit der neuen Geschwindigkeit)
    # -------------------------------------------------------------------------
    # Geschwindigkeit v sagt: "Wie schnell ändert sich die Position?"
    #
    # Mathematisch:
    # - x(t+dt) = x(t) + v(t+dt) * dt
    #
    # Warum v(t+dt) und nicht v(t)?
    # - Genau das ist Semi-Implicit Euler.
    # - Oft stabiler, besonders bei Feder-/Druck-artigen Kräften.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Step 2: update position (using the new velocity)
    # -------------------------------------------------------------------------
    # Velocity v means: "How fast does position change?"
    #
    # Mathematically:
    # - x(t+dt) = x(t) + v(t+dt) * dt
    #
    # Why v(t+dt) instead of v(t)?
    # - That is exactly what semi-implicit Euler does.
    # - Often more stable, especially for spring-/pressure-like forces.
    particles.x += particles.vx * float(dt)
    particles.y += particles.vy * float(dt)


