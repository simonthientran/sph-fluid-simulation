"""
DE:
Learning-Visualisierung für SPH (Smoothed Particle Hydrodynamics).

Was ist dieses Skript?
----------------------
Diese Datei ist bewusst wie ein kleines “lebendes Lehrbuch” geschrieben:
Sie zeigt in einer einzigen Abbildung mehrere wichtige Bausteine einer einfachen SPH-Welt,
die wir bisher implementiert haben:

1) **Partikel-Layout** (Punkte im Quadrat):
   - Wir initialisieren Partikel in einem regelmäßigen Gitter in einem Quadrat [0, L)×[0, L).
   - Das ist ein typischer Startzustand, um eine Flüssigkeit als Partikelmenge darzustellen.

2) **Poly6-Kernel W(r, h)** als Kurve:
   - Der Kernel ist eine glatte Gewichtungsfunktion, die nur von der Distanz r abhängt.
   - Innerhalb des Radius h ist W(r,h) positiv, außerhalb (r > h) ist W(r,h) exakt 0.
   - Das nennt man “compact support” und ist wichtig für Performance (nur Nachbarn zählen).

3) **Naive SPH-Dichte pro Partikel** als Farbe:
   - Die Dichte pro Partikel i wird als Summe über alle Partikel j berechnet:

     rho_i = sum_j m_j * W(r_ij, h)

   - Das ist fachlich korrekt, aber in der naiven Form langsam (O(N^2)).
   - Für kleine Beispiele ist es aber perfekt, um das Prinzip zu verstehen.

4) **Fokus-Partikel** + Kreis mit Radius h + Markierung der Nachbarn:
   - Wir wählen ein Partikel aus (focus_index).
   - Wir zeichnen einen Kreis mit Radius h um dieses Partikel.
   - Wir markieren alle Partikel, die innerhalb dieses Kreises liegen (r <= h).

Wie nutzt man das?
------------------
Du kannst diese Datei direkt ausführen:

    python -m sph_sim.vis.learning_viz

Oder du importierst die Funktion `run_learning_viz` und rufst sie in einem Notebook auf:

    from sph_sim.vis.learning_viz import run_learning_viz
    run_learning_viz(L=1.0, dx=0.1, h=0.15, focus_index=0)

Hinweis zur Geschwindigkeit:
---------------------------
`compute_density_naive` ist absichtlich eine Referenz-Implementierung mit doppelter Schleife.
Das bedeutet O(N^2). Wenn du dx sehr klein machst, steigt N stark an und das Skript wird langsam.
Für das Lernen ist das okay – wähle im Zweifel ein größeres dx (z.B. 0.1 oder 0.2).

Erweiterbarkeit (Design-Idee):
------------------------------
Wir halten die Visualisierung in mehrere kleine Funktionen getrennt, statt alles in eine
riesige Funktion zu packen. So können wir später neue Konzepte (z.B. Druck, Kräfte,
Neighbor-Grids) als eigene Plot-Funktionen ergänzen, ohne Chaos im Code.

EN:
Learning visualization for SPH (Smoothed Particle Hydrodynamics).

What is this script?
This file is deliberately written like a small “living textbook”:
It shows several important building blocks of a simple SPH world in a single figure,
which we have implemented so far:

1) **Particle layout** (points in a square):
   - We initialize particles on a regular grid in a square [0, L)×[0, L).
   - This is a typical initial condition to represent a fluid as a set of particles.

2) **Poly6 kernel W(r, h)** as a curve:
   - The kernel is a smooth weighting function that depends only on the distance r.
   - Inside the radius h, W(r,h) is positive; outside (r > h), W(r,h) is exactly 0.
   - This is called “compact support” and is important for performance (only neighbors matter).

3) **Naive SPH density per particle** as color:
   - The density for particle i is computed as a sum over all particles j:

     rho_i = sum_j m_j * W(r_ij, h)

   - This is scientifically correct, but slow in the naive form (O(N^2)).
   - For small examples, it is perfect to understand the principle.

4) **Focus particle** + circle with radius h + marking of neighbors:
   - We select one particle (focus_index).
   - We draw a circle with radius h around this particle.
   - We mark all particles that lie inside this circle (r <= h).

How do you use this?
You can run this file directly:

    python -m sph_sim.vis.learning_viz

Or you import the function `run_learning_viz` and call it from a notebook:

    from sph_sim.vis.learning_viz import run_learning_viz
    run_learning_viz(L=1.0, dx=0.1, h=0.15, focus_index=0)

Note on performance:
`compute_density_naive` is intentionally a reference implementation with a double loop.
That means O(N^2). If you make dx very small, N increases strongly and the script becomes slow.
For learning, this is fine—if in doubt, choose a larger dx (e.g., 0.1 or 0.2).

Extensibility (design idea):
We keep the visualization split into several small functions instead of putting everything
into one huge function. This allows us to add new concepts later (e.g., pressure, forces,
neighbor grids) as separate plotting functions without creating a mess in the code.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

import json
import time

from sph_sim.core.particles import initialize_particles_cube
from sph_sim.core.kernels import poly6_kernel
from sph_sim.core.density import compute_density_naive
from sph_sim.core.neighbor_search import build_uniform_grid, query_neighbor_candidates


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """
    DE:
    Minimales Debug-Logging für diese Session (NDJSON in eine Datei).

    Wichtig:
    - Das ist nur für Debugging im Tutor-Setup gedacht.
    - Wir schreiben KEINE sensiblen Daten, nur Plot-/Parameterwerte.

    EN:
    Minimal debug logging for this session (NDJSON to a file).

    Important:
    - This is only meant for debugging in the tutor setup.
    - We write NO sensitive data, only plot/parameter values.
    """

    payload = {
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }

    # --- Deutsch ---
    # NDJSON: ein JSON-Objekt pro Zeile.
    #
    # --- English ---
    # NDJSON: one JSON object per line.
    with open("/home/simon/dev/sph-project/.cursor/debug.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _validate_positive(name: str, value: float) -> None:
    """
    DE:
    Kleine Hilfsfunktion: Prüfe, dass ein Parameter > 0 ist.

    Warum eine Hilfsfunktion?
    - Wir brauchen diese Prüfung an mehreren Stellen.
    - Eine zentrale Funktion hält die Fehlermeldungen konsistent und den Code lesbar.

    EN:
    Small helper function: check that a parameter is > 0.

    Why a helper function?
    - We need this check in multiple places.
    - A single central function keeps error messages consistent and the code readable.
    """

    if value <= 0.0:
        raise ValueError(f"{name} muss > 0 sein.")


def _distances_to_focus(x: np.ndarray, y: np.ndarray, focus_index: int) -> np.ndarray:
    """
    DE:
    Berechne die Distanz jedes Partikels zum Fokus-Partikel.

    Rückgabe:
    - r: 1D-Array (Länge N), r[k] ist die Distanz vom Partikel k zum Fokus-Partikel.

    Hinweis:
    - Wir nutzen hier NumPy-Vektorisierung, weil es sehr lesbar ist:
      dx = x - x_focus ist ein Array-Array-Operator.

    EN:
    Compute the distance of every particle to the focus particle.

    Return value:
    - r: 1D array (length N), r[k] is the distance from particle k to the focus particle.

    Note:
    - We use NumPy vectorization here because it is very readable:
      dx = x - x_focus is an array-minus-scalar operation.
    """

    x_focus = float(x[focus_index])
    y_focus = float(y[focus_index])

    dx = x - x_focus
    dy = y - y_focus

    # --- Deutsch ---
    # Euklidische Distanz r = sqrt(dx^2 + dy^2) für alle Partikel gleichzeitig
    #
    # --- English ---
    # Euclidean distance r = sqrt(dx^2 + dy^2) for all particles at once
    return np.sqrt(dx * dx + dy * dy)


def draw_uniform_grid(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    *,
    linestyle: str = ":",
    alpha: float = 0.25,
    linewidth: float = 0.8,
    color: str = "gray",
    zorder: float = 1.0,
) -> None:
    """
    DE:
    Zeichne ein Uniform Grid (Zell-Linien) in eine Matplotlib-Achse.

    Was machen wir hier genau?
    - Wir zeichnen vertikale und horizontale Linien im Abstand `cell_size`.
    - Die Linien werden über den gesamten Partikelbereich gelegt.

    Didaktik:
    - Dieses Grid ist eine *Suchstruktur* (Datenstruktur) für die Nachbarsuche.
    - Es ist keine Physik und ändert nichts an den Partikeln.
    - Es hilft nur dabei, Kandidaten für Nachbarn schneller zu finden.

    Validierung:
    - cell_size muss > 0 sein.

    Parameter:
    - ax: Matplotlib-Achse, in die gezeichnet wird
    - x, y: Partikelpositionen (1D-Arrays)
    - cell_size: Kantenlänge einer Zelle
    - linestyle, alpha, linewidth, color, zorder: Stil der Linien (für eine dezente Darstellung)

    EN:
    Draw a uniform grid (cell lines) into a Matplotlib axis.

    What exactly do we do here?
    - We draw vertical and horizontal lines spaced by `cell_size`.
    - The lines span the full particle area.

    Teaching notes:
    - This grid is a *search structure* (data structure) for neighbor search.
    - It is not physics and it does not change the particles.
    - It only helps to find neighbor candidates faster.

    Validation:
    - cell_size must be > 0.

    Parameters:
    - ax: Matplotlib axis to draw into
    - x, y: particle positions (1D arrays)
    - cell_size: edge length of a cell
    - linestyle, alpha, linewidth, color, zorder: line style (for a subtle visualization)
    """

    _validate_positive("cell_size", float(cell_size))

    # --- Deutsch ---
    # Wichtiger Plot-Detail:
    # Wir orientieren das Grid am *sichtbaren Achsenbereich*.
    #
    # Warum?
    # - Matplotlib kann Achsenränder automatisch vergrößern (Margins).
    # - Wenn wir das Grid nur für [min(x), max(x)] zeichnen, wirkt es so,
    #   als würde das Grid "nicht passen" oder als fehlten Zellen.
    #
    # --- English ---
    # Important plotting detail:
    # We align the grid to the *visible axis area*.
    #
    # Why?
    # - Matplotlib can automatically expand axis margins.
    # - If we draw the grid only for [min(x), max(x)], it can look misaligned
    #   or like cells are missing at the borders.
    x_min, x_max = (float(v) for v in ax.get_xlim())
    y_min, y_max = (float(v) for v in ax.get_ylim())

    # --- Deutsch ---
    # Zelllinien sollen "sauber" auf Zellgrenzen liegen.
    # Darum runden wir die Start-/Endpunkte auf Vielfache von cell_size:
    # - Start: floor(...)
    # - Ende:  ceil(...)
    #
    # --- English ---
    # Cell lines should lie cleanly on cell boundaries.
    # Therefore we round start/end to multiples of cell_size:
    # - start: floor(...)
    # - end:   ceil(...)
    start_x = float(np.floor(x_min / cell_size) * cell_size)
    end_x = float(np.ceil(x_max / cell_size) * cell_size)
    start_y = float(np.floor(y_min / cell_size) * cell_size)
    end_y = float(np.ceil(y_max / cell_size) * cell_size)

    # --- Deutsch ---
    # Wir erzeugen eine Liste aller Zellgrenzen im sichtbaren Bereich.
    # `+ cell_size` am Ende sorgt dafür, dass die letzte Linie sicher dabei ist.
    #
    # --- English ---
    # We create a list of all cell boundaries in the visible area.
    # The `+ cell_size` ensures the last line is included.
    xs = np.arange(start_x, end_x + cell_size, cell_size, dtype=np.float64)
    ys = np.arange(start_y, end_y + cell_size, cell_size, dtype=np.float64)

    # region agent log
    _agent_debug_log(
        hypothesis_id="A",
        location="learning_viz.py:draw_uniform_grid",
        message="grid bounds and generated lines",
        data={
            "cell_size": float(cell_size),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "start_x": start_x,
            "end_x": end_x,
            "start_y": start_y,
            "end_y": end_y,
            "num_x_lines": int(xs.size),
            "num_y_lines": int(ys.size),
            "ax_xlim_before": list(ax.get_xlim()),
            "ax_ylim_before": list(ax.get_ylim()),
        },
    )
    # endregion agent log

    # region agent log
    _agent_debug_log(
        hypothesis_id="B",
        location="learning_viz.py:draw_uniform_grid",
        message="grid alignment remainders (symmetry diagnostic)",
        data={
            "cell_size": float(cell_size),
            "x_min_mod_cell": float(np.mod(x_min, cell_size)),
            "x_max_mod_cell": float(np.mod(x_max, cell_size)),
            "y_min_mod_cell": float(np.mod(y_min, cell_size)),
            "y_max_mod_cell": float(np.mod(y_max, cell_size)),
        },
    )
    # endregion agent log

    # --- Deutsch ---
    # Wir zeichnen die Linien dünn und grau, damit sie nicht vom Inhalt ablenken.
    #
    # --- English ---
    # We draw the lines thin and gray so they do not distract from the content.
    for xv in xs:
        ax.axvline(
            float(xv),
            color=color,
            linewidth=float(linewidth),
            alpha=float(alpha),
            linestyle=linestyle,
            zorder=float(zorder),
        )
    for yv in ys:
        ax.axhline(
            float(yv),
            color=color,
            linewidth=float(linewidth),
            alpha=float(alpha),
            linestyle=linestyle,
            zorder=float(zorder),
        )

    # region agent log
    _agent_debug_log(
        hypothesis_id="A",
        location="learning_viz.py:draw_uniform_grid",
        message="axis limits after drawing grid lines",
        data={
            "ax_xlim_after": list(ax.get_xlim()),
            "ax_ylim_after": list(ax.get_ylim()),
        },
    )
    # endregion agent log


def plot_particles(
    ax,
    x,
    y,
    focus_index: int,
    h: float,
    *,
    cell_size: float | None = None,
    candidate_indices: list[int] | None = None,
    true_neighbor_indices: list[int] | None = None,
    raster_dx: float | None = None,
) -> None:
    """
    DE:
    Plot: Partikel als Punkte + Fokus-Partikel + Kreis mit Radius h + Nachbarn.

    Was zeigen wir hier?
    - Alle Partikelpositionen als Scatter-Plot (Punkte).
    - Ein Fokus-Partikel (z.B. Partikel 0) deutlich hervorgehoben.
    - Einen Kreis mit Radius h um den Fokus: das ist der Einflussradius des Kernels.
    - Optional: alle Nachbarn innerhalb des Radius (r <= h) anders einfärben.
    - Optional: Uniform Grid Linien (Zellen), um die Suchstruktur zu sehen.
    - Optional: dx-basiertes Raster für Orientierung (symmetrisch/gleichmäßig).
    - Optional: Kandidaten (aus 3×3 Zellen) vs. echte Nachbarn (r <= h).

    Warum ist das didaktisch wichtig?
    - In SPH hat jedes Partikel nur "Nachbarn" innerhalb einer bestimmten Reichweite (h).
    - Diese Visualisierung macht den Begriff "compact support" sofort sichtbar.
    - Das Uniform Grid zeigt, wie wir Kandidaten reduzieren, bevor wir Distanz prüfen.

    EN:
    Plot: particles as points + focus particle + circle with radius h + neighbors.

    What do we show here?
    - All particle positions as a scatter plot (points).
    - A focus particle (e.g., particle 0) clearly highlighted.
    - A circle with radius h around the focus: this is the kernel's influence radius.
    - Optionally: color all neighbors within the radius (r <= h) differently.
    - Optionally: uniform grid lines (cells) to visualize the search structure.
    - Optionally: dx-based raster for orientation (symmetric/even).
    - Optionally: candidates (from 3×3 cells) vs true neighbors (r <= h).

    Why is this important for teaching?
    - In SPH, each particle only has "neighbors" within a certain range (h).
    - This visualization makes the concept of "compact support" immediately visible.
    - The uniform grid shows how we reduce candidates before doing the distance check.
    """

    # --- Deutsch ---
    # Distanz jedes Partikels zum Fokus-Partikel berechnen.
    #
    # --- English ---
    # Compute the distance of every particle to the focus particle.
    r = _distances_to_focus(x, y, focus_index)

    # --- Deutsch ---
    # Grundschicht: Alle Partikel (leicht grau) – das ist der "Hintergrund".
    #
    # --- English ---
    # Base layer: all particles (light gray) — this is the "background".
    ax.scatter(x, y, s=30, c="lightgray", label="alle Partikel", zorder=2)

    # --- Deutsch ---
    # Wir unterscheiden zwei Fälle:
    #
    # Fall 1 (einfach): Wenn keine Kandidaten/echten Nachbarn übergeben wurden,
    # nutzen wir das alte Verhalten: "alle innerhalb r <= h" werden markiert.
    #
    # Fall 2 (Uniform Grid): Wenn Listen übergeben wurden, zeigen wir:
    # - Kandidaten (orange)
    # - echte Nachbarn (grün) = Kandidaten mit r <= h
    #
    # --- English ---
    # We distinguish two cases:
    #
    # Case 1 (simple): if no candidates/true neighbors are provided,
    # we use the old behavior: mark "all within r <= h".
    #
    # Case 2 (uniform grid): if lists are provided, we show:
    # - candidates (orange)
    # - true neighbors (green) = candidates with r <= h
    if candidate_indices is None or true_neighbor_indices is None:
        # --- Deutsch ---
        # Nachbarn sind alle Partikel mit r <= h.
        # (Wir zählen auch das Fokus-Partikel selbst dazu, weil r=0 <= h ist.)
        #
        # --- English ---
        # Neighbors are all particles with r <= h.
        # (We also count the focus particle itself because r=0 <= h.)
        inside_mask = r <= h

        # --- Deutsch ---
        # Nachbarn (innerhalb h) – auffälliger, damit man sieht "wer zählt".
        #
        # --- English ---
        # Neighbors (inside h) — more prominent so you can see "who counts".
        ax.scatter(
            x[inside_mask],
            y[inside_mask],
            s=45,
            c="tab:blue",
            label=f"Nachbarn (r ≤ h), Anzahl={int(np.sum(inside_mask) - 1)}",
        )
    else:
        candidates_set = set(int(j) for j in candidate_indices)
        true_neighbors_set = set(int(j) for j in true_neighbor_indices)

        # --- Deutsch ---
        # Wir definieren drei disjunkte Gruppen (ohne Fokus):
        # - Kandidaten (orange)
        # - echte Nachbarn (grün)
        # - Fokus (rot, separat)
        #
        # --- English ---
        # We define three disjoint groups (excluding focus):
        # - candidates (orange)
        # - true neighbors (green)
        # - focus (red, separate)
        candidates_without_focus = sorted(candidates_set - {focus_index})
        true_neighbors_only = sorted(true_neighbors_set - {focus_index})

        # --- Deutsch ---
        # Zuerst: alle Kandidaten orange.
        # Danach: echte Nachbarn grün darüber (damit sichtbar ist: "Kandidaten ⊃ Nachbarn").
        #
        # --- English ---
        # First: all candidates in orange.
        # Then: true neighbors in green on top (so you can see: "candidates ⊃ neighbors").
        if len(candidates_without_focus) > 0:
            ax.scatter(
                x[candidates_without_focus],
                y[candidates_without_focus],
                s=55,
                c="tab:orange",
                label=f"Kandidaten (aus Grid) = {len(candidates_without_focus)}",
                zorder=3,
            )

        if len(true_neighbors_only) > 0:
            ax.scatter(
                x[true_neighbors_only],
                y[true_neighbors_only],
                s=60,
                c="tab:green",
                label=f"Echte Nachbarn (r ≤ h) = {len(true_neighbors_only)}",
                zorder=4,
            )

    # --- Deutsch ---
    # Fokus-Partikel (stark hervorgehoben).
    #
    # --- English ---
    # Focus particle (strongly highlighted).
    ax.scatter(
        [x[focus_index]],
        [y[focus_index]],
        s=120,
        c="tab:red",
        marker="*",
        label=f"Fokus-Partikel i={focus_index}",
        zorder=5,
    )

    # --- Deutsch ---
    # Über den anderen Punkten zeichnen, damit der Fokus immer sichtbar bleibt.
    #
    # --- English ---
    # Draw above the other points so the focus is always visible.
    #
    # (Wir setzen zorder=5 oben in ax.scatter.)
    #
    # --- Deutsch ---
    # Kreis um das Fokus-Partikel:
    # - Das ist die geometrische Darstellung des Einflussradius h.
    #
    # --- English ---
    # Circle around the focus particle:
    # - This is the geometric visualization of the influence radius h.
    circle = Circle(
        (float(x[focus_index]), float(y[focus_index])),
        radius=float(h),
        fill=False,
        color="tab:red",
        linewidth=1.1,
        alpha=0.7,
        label="Radius h (Einflussbereich)",
    )
    ax.add_patch(circle)

    # --- Deutsch ---
    # Achsen/Look & Feel
    #
    # --- English ---
    # Axes / look & feel
    # --- Deutsch ---
    # Gleiche Skalierung in x und y (sonst werden Kreise optisch zu Ellipsen).
    #
    # --- English ---
    # Same scaling in x and y (otherwise circles visually become ellipses).
    ax.set_aspect("equal", adjustable="box")

    # --- Deutsch ---
    # Achsen-Limits bewusst setzen (wichtig für 2 Dinge):
    #
    # (1) Grid-Symmetrie / Zelloptik:
    # - Wir wollen, dass die sichtbaren Plot-Grenzen auf Zellgrenzen liegen.
    # - Sonst "schneiden" Zellen am Rand komisch ab und das Grid wirkt verzerrt.
    #
    # (2) Kernel-Radius (Kreis) komplett sichtbar:
    # - Der Kreis um den Fokus hat Radius h.
    # - Daher müssen wir sicherstellen, dass x_focus ± h und y_focus ± h sichtbar sind.
    #
    # --- English ---
    # Set axis limits deliberately (important for 2 things):
    #
    # (1) Grid symmetry / cell appearance:
    # - We want the visible plot bounds to lie on cell boundaries.
    # - Otherwise cells at the border look cut off and the grid appears distorted.
    #
    # (2) Kernel radius (circle) fully visible:
    # - The circle around the focus has radius h.
    # - Therefore we must ensure x_focus ± h and y_focus ± h are visible.
    x_data_min = float(np.min(x))
    x_data_max = float(np.max(x))
    y_data_min = float(np.min(y))
    y_data_max = float(np.max(y))

    x_focus = float(x[focus_index])
    y_focus = float(y[focus_index])

    # --- Deutsch ---
    # Wir definieren den "sichtbaren Bereich" so, dass 2 Dinge gleichzeitig stimmen:
    #
    # (1) Kreis (Radius h) ist komplett sichtbar.
    # (2) Randabstände sind symmetrisch: links/rechts/oben/unten gleich groß.
    #
    # Dazu geben wir bewusst einen Rand (Padding) um das Partikelgitter:
    # - Wenn wir ein dx-Raster zeigen, ist ein sehr natürlicher Rand: dx/2
    #   (dann liegt der erste Partikel "mittig" in der ersten Rasterzelle).
    #
    # --- English ---
    # We define the "visible area" so that 2 things hold at the same time:
    #
    # (1) The circle (radius h) is fully visible.
    # (2) Border margins are symmetric: left/right/top/bottom are equal.
    #
    # Therefore we deliberately add padding around the particle grid:
    # - If we show a dx raster, a very natural padding is dx/2
    #   (then the first particle lies "centered" in the first raster cell).
    if raster_dx is not None:
        pad = 0.5 * float(raster_dx)
    else:
        pad = 0.0

    x_needed_min = min(x_data_min - pad, x_focus - float(h))
    x_needed_max = max(x_data_max + pad, x_focus + float(h))
    y_needed_min = min(y_data_min - pad, y_focus - float(h))
    y_needed_max = max(y_data_max + pad, y_focus + float(h))

    # --- Deutsch ---
    # Welches Raster soll für "Snap" gelten?
    #
    # Priorität:
    # 1) raster_dx (wenn wir ein symmetrisches, dx-basiertes Raster zeigen wollen)
    # 2) cell_size  (Such-Grid)
    #
    # Hintergrund:
    # - Wenn dx und cell_size keine Vielfachen sind (z.B. dx=0.1, h=0.15),
    #   kann man nicht beide Raster gleichzeitig perfekt "symmetrisch" ausrichten.
    # - Für das Lernbild ist die Orientierung über dx meistens wichtiger,
    #   und der Suchbereich wird dann als Rechteck gezeigt.
    #
    # --- English ---
    # Which raster should be used for snapping the axis bounds?
    #
    # Priority:
    # 1) raster_dx (if we want a symmetric dx-based raster)
    # 2) cell_size  (search grid)
    #
    # Background:
    # - If dx and cell_size are not multiples (e.g. dx=0.1, h=0.15),
    #   you cannot align both rasters perfectly at the same time.
    # - For a learning figure, dx orientation is usually more important,
    #   and the search area is then shown as a rectangle.
    snap_size: float | None = None
    if raster_dx is not None:
        snap_size = float(raster_dx)
    elif cell_size is not None:
        snap_size = float(cell_size)

    if snap_size is not None:
        x0 = float(np.floor(x_needed_min / snap_size) * snap_size)
        x1 = float(np.ceil(x_needed_max / snap_size) * snap_size)
        y0 = float(np.floor(y_needed_min / snap_size) * snap_size)
        y1 = float(np.ceil(y_needed_max / snap_size) * snap_size)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    else:
        ax.set_xlim(x_needed_min, x_needed_max)
        ax.set_ylim(y_needed_min, y_needed_max)

    # region agent log
    # --- Deutsch ---
    # Symmetrie-Diagnose:
    # Wir messen, wie weit der Fokus vom linken/rechten und unteren/oberen Rand entfernt ist.
    # Perfekt symmetrisch um den Fokus wäre: left==right und bottom==top.
    #
    # --- English ---
    # Symmetry diagnostic:
    # We measure how far the focus is from the left/right and bottom/top bounds.
    # Perfect symmetry around the focus would be: left==right and bottom==top.
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    left = float(x_focus - float(x_lim[0]))
    right = float(float(x_lim[1]) - x_focus)
    bottom = float(y_focus - float(y_lim[0]))
    top = float(float(y_lim[1]) - y_focus)
    _agent_debug_log(
        hypothesis_id="F",
        location="learning_viz.py:plot_particles",
        message="symmetry distances from focus to bounds",
        data={
            "focus_index": int(focus_index),
            "x_focus": float(x_focus),
            "y_focus": float(y_focus),
            "x_lim": [float(x_lim[0]), float(x_lim[1])],
            "y_lim": [float(y_lim[0]), float(y_lim[1])],
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
            "left_minus_right": float(left - right),
            "bottom_minus_top": float(bottom - top),
            "snap_size": None if snap_size is None else float(snap_size),
            "raster_dx": None if raster_dx is None else float(raster_dx),
            "cell_size": None if cell_size is None else float(cell_size),
            "h": float(h),
        },
    )
    # endregion agent log

    # region agent log
    _agent_debug_log(
        hypothesis_id="B",
        location="learning_viz.py:plot_particles",
        message="axis limits after explicit set_xlim/set_ylim (symmetry diagnostic)",
        data={
            "cell_size": None if cell_size is None else float(cell_size),
            "x_data_min": x_data_min,
            "x_data_max": x_data_max,
            "y_data_min": y_data_min,
            "y_data_max": y_data_max,
            "ax_xlim_after_set": list(ax.get_xlim()),
            "ax_ylim_after_set": list(ax.get_ylim()),
        },
    )
    # endregion agent log

    # region agent log
    _agent_debug_log(
        hypothesis_id="D",
        location="learning_viz.py:plot_particles",
        message="focus position and cell_size (cell visibility diagnostic)",
        data={
            "cell_size": None if cell_size is None else float(cell_size),
            "focus_index": int(focus_index),
            "x_focus": float(x[focus_index]),
            "y_focus": float(y[focus_index]),
        },
    )
    # endregion agent log

    # --- Deutsch ---
    # Optional: dx-basiertes Raster (Orientierung):
    # - sehr dezent (linestyle=":", geringe Alpha)
    # - dieses Raster ist nur für die Optik, keine Physik
    #
    # --- English ---
    # Optional: dx-based raster (orientation):
    # - very subtle (linestyle=":", low alpha)
    # - this raster is visuals only, not physics
    if raster_dx is not None:
        _validate_positive("raster_dx", float(raster_dx))
        draw_uniform_grid(
            ax=ax,
            x=x,
            y=y,
            cell_size=float(raster_dx),
            linestyle=":",
            alpha=0.15,
            linewidth=0.7,
            color="gray",
            zorder=0.8,
        )
        try:
            xs_pts = np.unique(np.round(np.asarray(x, dtype=np.float64), 12))
            ys_pts = np.unique(np.round(np.asarray(y, dtype=np.float64), 12))
            if xs_pts.size <= 80 and ys_pts.size <= 80:
                for xv in xs_pts:
                    ax.axvline(
                        float(xv),
                        color="#b0b0b0",
                        linewidth=0.9,
                        alpha=0.35,
                        linestyle="-",
                        zorder=0.9,
                    )
                for yv in ys_pts:
                    ax.axhline(
                        float(yv),
                        color="#b0b0b0",
                        linewidth=0.9,
                        alpha=0.35,
                        linestyle="-",
                        zorder=0.9,
                    )
        except Exception:
            pass

    # --- Deutsch ---
    # Optional: Such-Grid (cell_size) als sehr dezente Linien.
    # Wichtig: Das Grid ist eine Suchstruktur, keine Physik.
    #
    # Zusätzlich: Suchbereich als EIN Rechteck (3×3 Zellen) hervorheben.
    # Das macht sofort sichtbar:
    # "Grid reduziert Kandidaten – Radius definiert Physik."
    #
    # --- English ---
    # Optional: search grid (cell_size) as very subtle lines.
    # Important: the grid is a search structure, not physics.
    #
    # Additionally: highlight the search area as ONE rectangle (3×3 cells).
    # This makes it immediately visible:
    # "Grid reduces candidates – radius defines physics."
    if cell_size is not None:
        cell = float(cell_size)

        draw_uniform_grid(
            ax=ax,
            x=x,
            y=y,
            cell_size=cell,
            linestyle=":",
            alpha=0.25,
            linewidth=0.8,
            color="gray",
            zorder=1.0,
        )

        ix = int(np.floor(x_focus / cell))
        iy = int(np.floor(y_focus / cell))

        if candidate_indices is not None and len(candidate_indices) > 0:
            _cand = np.asarray(candidate_indices, dtype=np.int64)
            _x_c = x[_cand]
            _y_c = y[_cand]
            search_x0 = float(np.min(_x_c))
            search_y0 = float(np.min(_y_c))
            search_w = float(np.max(_x_c) - np.min(_x_c))
            search_h = float(np.max(_y_c) - np.min(_y_c))
        else:
            search_x0 = float(x_focus - 1.5 * cell)
            search_y0 = float(y_focus - 1.5 * cell)
            search_w = float(3.0 * cell)
            search_h = float(3.0 * cell)

        search_rect = Rectangle(
            (search_x0, search_y0),
            width=search_w,
            height=search_h,
            facecolor="tab:orange",
            edgecolor="tab:orange",
            linewidth=1.6,
            linestyle="--",
            alpha=0.10,
            zorder=1.2,  # über Grid-Linien, unter Partikelpunkten
        )
        ax.add_patch(search_rect)

    ax.set_title("Partikel-Layout + Fokus + Nachbarn innerhalb h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2, linestyle=":")

    # region agent log
    _agent_debug_log(
        hypothesis_id="A",
        location="learning_viz.py:plot_particles",
        message="axis/aspect before legend",
        data={
            "cell_size": None if cell_size is None else float(cell_size),
            "has_candidates": bool(candidate_indices is not None),
            "has_true_neighbors": bool(true_neighbor_indices is not None),
            "ax_xlim": list(ax.get_xlim()),
            "ax_ylim": list(ax.get_ylim()),
            "aspect": "equal",
        },
    )
    # endregion agent log

    # --- Deutsch ---
    # Legende: Matplotlib wählt mit "best" automatisch eine freie Stelle.
    # Das kann aber (je nach Daten) über dem Fokus-Partikel landen.
    # Für Debugging loggen wir nur, *dass* wir "best" benutzen.
    #
    # --- English ---
    # Legend: Matplotlib chooses a free location automatically with "best".
    # But (depending on the data) this can end up on top of the focus particle.
    # For debugging, we only log that we use "best".
    # --- Deutsch ---
    # Trick für die automatische Legendenposition ("best"):
    # Wir legen einen unsichtbaren "Platzhalter" um den Fokus.
    # Dann versucht Matplotlib, die Legende nicht genau dort zu platzieren.
    #
    # --- English ---
    # Trick for the automatic legend placement ("best"):
    # We add an invisible "placeholder" around the focus.
    # Then Matplotlib tries to avoid placing the legend right there.
    ax.scatter(
        [float(x[focus_index])],
        [float(y[focus_index])],
        s=6000,
        c="white",
        alpha=0.0,
        label="_nolegend_",
        zorder=0,
    )

    # --- Deutsch ---
    # Legende bewusst fest platzieren:
    # - "best" kann je nach Daten irgendwo landen und stört dann die Lesbarkeit.
    # - Für Lernzwecke ist eine konstante Position besser.
    #
    # Wunsch: oben links.
    #
    # Hinweis (Debug-Mode):
    # - Wir erstellen die Legende wie bisher, verschieben sie dann aber explizit.
    # - So bleiben bestehende Debug-Logs konsistent, und wir haben trotzdem das gewünschte Layout.
    #
    # --- English ---
    # Place the legend deliberately:
    # - "best" can end up anywhere depending on the data and then hurts readability.
    # - For learning purposes, a constant position is better.
    #
    # Desired: upper left.
    #
    # Note (debug mode):
    # - We create the legend as before and then explicitly move it.
    # - This keeps existing debug logs consistent while still achieving the desired layout.
    legend = ax.legend(loc="best")
    if legend is not None:
        legend.set_loc("lower left")

    # region agent log
    _agent_debug_log(
        hypothesis_id="C",
        location="learning_viz.py:plot_particles",
        message="legend moved to upper left",
        data={"legend_loc_final": "lower left"},
    )
    # endregion agent log

    # region agent log
    _agent_debug_log(
        hypothesis_id="C",
        location="learning_viz.py:plot_particles",
        message="legend created",
        data={"legend_loc": "best"},
    )
    # endregion agent log


def plot_poly6_kernel_curve(ax, h: float, num_samples: int = 200) -> None:
    """
    DE:
    Plot: Poly6-Kernelkurve W(r, h) über r.

    Was zeigen wir?
    - r von 0 bis 1.5*h (damit man auch den Bereich *außerhalb* des Supports sieht).
    - W(r,h) als Kurve.
    - Eine vertikale Linie bei r = h, um den Übergang "ab hier 0" zu markieren.

    Didaktik:
    - Innerhalb von h ist der Kernel > 0 (für r < h).
    - Außerhalb von h ist der Kernel exakt 0 ("compact support").

    EN:
    Plot: Poly6 kernel curve W(r, h) over r.

    What do we show?
    - r from 0 to 1.5*h (so you can also see the region *outside* the support).
    - W(r,h) as a curve.
    - A vertical line at r = h to mark the transition "from here on 0".

    Teaching notes:
    - Inside h, the kernel is > 0 (for r < h).
    - Outside h, the kernel is exactly 0 ("compact support").
    """

    _validate_positive("h", h)

    # --- Deutsch ---
    # r-Werte sampeln: gleichmäßig von 0 bis 1.5*h
    #
    # --- English ---
    # Sample r values: uniformly from 0 to 1.5*h
    r = np.linspace(0.0, 1.5 * h, int(num_samples), dtype=np.float64)

    # --- Deutsch ---
    # Kernelwerte berechnen (funktioniert direkt mit Arrays).
    #
    # --- English ---
    # Compute kernel values (works directly with arrays).
    W = poly6_kernel(r, h)

    ax.plot(r, W, color="tab:purple", linewidth=2.0, label="Poly6: W(r,h)")

    # --- Deutsch ---
    # Vertikale Linie bei r = h: ab hier ist der Kernel 0.
    #
    # --- English ---
    # Vertical line at r = h: beyond this point the kernel is 0.
    ax.axvline(h, color="black", linestyle="--", linewidth=1.5, label="r = h (Support-Grenze)")

    # --- Deutsch ---
    # Kleine Textbox, die die wichtigste Aussage zusammenfasst.
    #
    # --- English ---
    # Small textbox that summarizes the key statement.
    ax.text(
        0.02,
        0.95,
        "Außerhalb des Radius h gilt:\nW(r,h) = 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_title("Poly6-Kernel als Funktion der Distanz r")
    ax.set_xlabel("Distanz r")
    ax.set_ylabel("Kernelwert W(r,h)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_density(ax, x, y, rho, focus_index: int, h: float) -> None:
    """
    DE:
    Plot: Dichte pro Partikel als Farbe + Fokus-Partikel + Kreis mit Radius h.

    Was sehen wir?
    - Jeder Punkt (Partikel) wird eingefärbt nach seiner Dichte rho[i].
    - Eine Colorbar erklärt die Farbkodierung.
    - Fokus-Partikel und Kreis h werden wieder eingezeichnet,
      damit man den Zusammenhang zwischen "Nachbarschaft" und "Dichte" sieht.

    Didaktik:
    - Dichte in SPH ist eine Kernel-gewichtete Summe über Nachbarn:
      rho_i = sum_j m_j * W(r_ij, h)
    - In dichten Regionen (viele Nachbarn nahe dran) ist rho typischerweise größer.

    EN:
    Plot: density per particle as color + focus particle + circle with radius h.

    What do we see?
    - Each point (particle) is colored by its density rho[i].
    - A colorbar explains the color mapping.
    - The focus particle and circle h are drawn again
      so you can see the relationship between "neighborhood" and "density".

    Teaching notes:
    - In SPH, density is a kernel-weighted sum over neighbors:
      rho_i = sum_j m_j * W(r_ij, h)
    - In denser regions (many neighbors close by), rho is typically larger.
    """

    # --- Deutsch ---
    # Scatter mit Farbskala (c=rho)
    #
    # --- English ---
    # Scatter with a color scale (c=rho)
    sc = ax.scatter(x, y, c=rho, s=45, cmap="viridis")

    # --- Deutsch ---
    # Colorbar gehört zur ganzen Figure, aber wir hängen sie an diese Achse.
    #
    # --- English ---
    # The colorbar belongs to the whole figure, but we attach it to this axis.
    cbar = ax.figure.colorbar(sc, ax=ax)
    cbar.set_label("density rho")

    # --- Deutsch ---
    # Fokus + Kreis wieder zeichnen (wie im Partikelplot)
    #
    # --- English ---
    # Draw focus + circle again (as in the particle plot)
    ax.scatter(
        [x[focus_index]],
        [y[focus_index]],
        s=140,
        c="tab:red",
        marker="*",
        label=f"Fokus-Partikel i={focus_index}",
        zorder=5,
    )

    circle = Circle(
        (float(x[focus_index]), float(y[focus_index])),
        radius=float(h),
        fill=False,
        color="tab:red",
        linewidth=2.0,
        alpha=0.9,
        label="Radius h",
    )
    ax.add_patch(circle)

    # --- Deutsch ---
    # Textbox mit Min/Max und der zentralen Formel
    #
    # --- English ---
    # Textbox with min/max and the central formula
    rho_min = float(np.min(rho))
    rho_max = float(np.max(rho))
    text = (
        f"rho min = {rho_min:.6g}\n"
        f"rho max = {rho_max:.6g}\n\n"
        "SPH-Dichte:\n"
        "rho_i = Σ_j m_j · W(r_ij, h)"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Naive SPH-Dichte (Farbe) + Fokus + Radius h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")


def run_learning_viz(
    L: float = 1.0,
    dx: float = 0.1,
    rho0: float = 1000.0,
    mass_per_particle: float = 0.01,
    h: float = 0.15,
    focus_index: int = 0,
) -> None:
    """
    DE:
    Erzeuge Partikel, berechne Dichte und erstelle eine 3-Panel-Lernvisualisierung.

    Panels (von links nach rechts):
    1) Partikelpositionen + Fokus-Partikel + Nachbarn innerhalb h
    2) Poly6-Kernelkurve W(r,h) und Markierung des Radius h
    3) Dichte pro Partikel als Farbe (rho) + Fokus-Partikel + Radius h

    Validierung:
    - L, dx, rho0, mass_per_particle, h müssen > 0 sein.
    - focus_index muss in [0, N-1] liegen (sonst setzen wir ihn auf 0).

    Hinweis für Anfänger:
    - Wenn du dx sehr klein machst, entstehen sehr viele Partikel.
    - Die naive Dichteberechnung ist O(N^2) und wird dann langsam.

    EN:
    Create particles, compute density, and build a 3-panel learning visualization.

    Panels (from left to right):
    1) Particle positions + focus particle + neighbors within h
    2) Poly6 kernel curve W(r,h) and marking of the radius h
    3) Density per particle as color (rho) + focus particle + radius h

    Validation:
    - L, dx, rho0, mass_per_particle, h must be > 0.
    - focus_index must be in [0, N-1] (otherwise we set it to 0).

    Note for beginners:
    - If you make dx very small, you create many particles.
    - The naive density computation is O(N^2) and becomes slow.
    """

    # --- Deutsch ---
    # --- Eingaben validieren (frühe, klare Fehler) ----------------------------
    #
    # --- English ---
    # --- Validate inputs (early, clear errors) --------------------------------
    _validate_positive("L", L)
    _validate_positive("dx", dx)
    _validate_positive("rho0", rho0)
    _validate_positive("mass_per_particle", mass_per_particle)
    _validate_positive("h", h)

    # --- Deutsch ---
    # Eine sehr einfache Plausibilitätsprüfung:
    # Wenn dx >= L, entstehen in der Regel 0 oder 1 Partikel pro Achse → nicht sinnvoll.
    #
    # --- English ---
    # A very simple plausibility check:
    # If dx >= L, you typically get 0 or 1 particle per axis → not meaningful.
    if dx >= L:
        raise ValueError("dx muss kleiner als L sein (dx < L), sonst entstehen zu wenige Partikel.")

    # --- Deutsch ---
    # --- Partikel erzeugen ----------------------------------------------------
    #
    # --- English ---
    # --- Create particles -----------------------------------------------------
    particles = initialize_particles_cube(
        L=L,
        dx=dx,
        rho0=rho0,
        mass_per_particle=mass_per_particle,
    )

    N = particles.n

    # --- Deutsch ---
    # --- Fokusindex "clampen" -------------------------------------------------
    # Anfängerfreundliche Entscheidung:
    # Wenn jemand einen ungültigen Index eingibt, ist das wahrscheinlich ein Tippfehler.
    # Statt das ganze Skript abstürzen zu lassen, setzen wir auf 0 zurück.
    # (In manchen Projekten würde man hier strikt ValueError werfen – beides ist okay.
    # Für dieses Lernskript ist ein sanfter Fallback oft angenehmer.)
    #
    # --- English ---
    # --- "Clamp" the focus index ---------------------------------------------
    # Beginner-friendly decision:
    # If someone provides an invalid index, it is likely a typo.
    # Instead of letting the whole script fail, we reset it to 0.
    # (In some projects, you would strictly raise ValueError here—both are acceptable.)
    # For this learning script, a gentle fallback is often more pleasant.)
    if focus_index < 0 or focus_index >= N:
        focus_index = 0

    # --- Deutsch ---
    # Fokus-Partikel für Lernzwecke "innen" wählen:
    #
    # Problem:
    # - Der Default focus_index=0 liegt bei unserem Startgitter oft am Rand (z.B. (0,0)).
    # - Am Rand sind Nachbarschaften asymmetrisch (Rand-Effekt) und das Bild wirkt weniger "symmetrisch".
    #
    # Lösung:
    # - Wenn focus_index=0 (Default), wählen wir stattdessen das Partikel, das dem Zentrum
    #   des Gebiets am nächsten ist (ungefähr (L/2, L/2)).
    #
    # Wichtig:
    # - Wenn du explizit einen anderen focus_index setzt, respektieren wir das.
    #
    # --- English ---
    # Choose an "interior" focus particle for learning:
    #
    # Problem:
    # - The default focus_index=0 is often at the boundary for our initial grid (e.g. (0,0)).
    # - At the boundary, neighborhoods are asymmetric (boundary effect) and the figure looks less "symmetric".
    #
    # Solution:
    # - If focus_index=0 (default), we instead pick the particle closest to the domain center
    #   (approximately (L/2, L/2)).
    #
    # Important:
    # - If you explicitly set a different focus_index, we respect it.
    if focus_index == 0 and N > 0:
        x_target = 0.5 * float(L)
        y_target = 0.5 * float(L)

        dx_center = particles.x - x_target
        dy_center = particles.y - y_target
        dist2 = dx_center * dx_center + dy_center * dy_center

        focus_index = int(np.argmin(dist2))

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Uniform Grid Nachbarsuche (Suchstruktur)
    # -------------------------------------------------------------------------
    # Wir wählen hier cell_size = h.
    #
    # Warum ist das eine plausible Wahl?
    # - h ist der Einflussradius des Kernels.
    # - Wenn Zellen ungefähr so groß wie h sind, dann liegen echte Nachbarn typischerweise
    #   in der eigenen Zelle oder in den direkt angrenzenden Zellen.
    #
    # Wichtig:
    # - Das Grid ist nur eine Suchstruktur. Es ist keine Physik.
    # - Physik kommt erst bei der Distanzprüfung: r <= h.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Uniform grid neighbor search (search structure)
    # -------------------------------------------------------------------------
    # We choose cell_size = h.
    #
    # Why is this a plausible choice?
    # - h is the kernel's influence radius.
    # - If cells are roughly the size of h, true neighbors typically lie in the same cell
    #   or in directly adjacent cells.
    #
    # Important:
    # - The grid is only a search structure. It is not physics.
    # - Physics only comes in the distance check: r <= h.
    cell_size = h

    grid = build_uniform_grid(particles=particles, cell_size=cell_size)
    candidates = query_neighbor_candidates(
        i=focus_index,
        particles=particles,
        grid=grid,
        cell_size=cell_size,
    )

    # --- Deutsch ---
    # Kandidaten können wir "sauber" als Menge betrachten (ohne Duplikate).
    # (In diesem Grid-Ansatz entstehen normalerweise keine Duplikate, aber es ist robust.)
    #
    # --- English ---
    # We treat candidates cleanly as a set (no duplicates).
    # (In this grid approach, duplicates usually do not occur, but this is robust.)
    candidates_unique = sorted(set(int(j) for j in candidates))

    # --- Deutsch ---
    # Echte Nachbarn bestimmen wir über die Physik-Regel: Distanz r <= h.
    # Dafür brauchen wir Distanzen vom Fokus zu allen Partikeln.
    #
    # --- English ---
    # We determine true neighbors by the physics rule: distance r <= h.
    # For this we need distances from the focus to all particles.
    r_all = _distances_to_focus(particles.x, particles.y, focus_index=focus_index)

    true_neighbors_including_self = [j for j in candidates_unique if float(r_all[j]) <= h]

    # --- Deutsch ---
    # Für die Visualisierung ist es oft sinnvoll, den Fokus selbst separat zu behandeln.
    # Daher splitten wir:
    # - echte Nachbarn ohne Fokus
    # - Kandidaten ohne Fokus (die nicht echt sind)
    #
    # --- English ---
    # For visualization, it is often convenient to treat the focus separately.
    # Therefore we split:
    # - true neighbors excluding the focus
    # - candidates excluding the focus (that are not true)
    true_neighbors_excluding_focus = [j for j in true_neighbors_including_self if j != focus_index]

    candidates_other = [j for j in candidates_unique if j != focus_index]
    num_candidates_other = len(candidates_other)
    num_true_neighbors = len(true_neighbors_excluding_focus)

    # --- Deutsch ---
    # --- Dichte berechnen (naiv, O(N^2)) -------------------------------------
    #
    # --- English ---
    # --- Compute density (naive, O(N^2)) --------------------------------------
    rho = compute_density_naive(particles=particles, h=h)

    # --- Deutsch ---
    # --- Figure mit 3 Subplots ------------------------------------------------
    #
    # --- English ---
    # --- Figure with 3 subplots -----------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # --- Deutsch ---
    # Links: Uniform Grid Nachbarsuche sichtbar machen
    #
    # --- English ---
    # Left: make uniform grid neighbor search visible
    plot_particles(
        axs[0],
        particles.x,
        particles.y,
        focus_index=focus_index,
        h=h,
        cell_size=cell_size,
        raster_dx=dx,
        candidate_indices=candidates_unique,
        true_neighbor_indices=true_neighbors_including_self,
    )

    # --- Deutsch ---
    # Titel (didaktisch): Das Grid reduziert Kandidaten, aber die Physik entscheidet mit r<=h.
    #
    # --- English ---
    # Title (teaching): the grid reduces candidates, but physics decides with r<=h.
    axs[0].set_title("Uniform Grid Neighbor Search\nGrid reduces candidates, radius defines physics")

    # --- Deutsch ---
    # Textbox mit den wichtigsten Zahlen:
    # - N: Gesamtpartikel
    # - Anzahl Kandidaten (ohne Fokus)
    # - Anzahl echter Nachbarn (ohne Fokus)
    #
    # Kurzer Merksatz:
    # - Grid = Suchstruktur
    # - r <= h = Physik
    #
    # --- English ---
    # Textbox with the key numbers:
    # - N: total particles
    # - number of candidates (excluding focus)
    # - number of true neighbors (excluding focus)
    #
    # Short takeaway:
    # - grid = search structure
    # - r <= h = physics
    left_panel_text = axs[0].text(
        0.98,
        0.02,
        "N = {N}\n"
        "Kandidaten (Grid) = {c}\n"
        "Echte Nachbarn (r ≤ h) = {n}\n\n"
        "Grid = Suchstruktur\n"
        "r ≤ h = Physik".format(N=N, c=num_candidates_other, n=num_true_neighbors),
        transform=axs[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    # --- Deutsch ---
    # Mitte: Kernelkurve
    #
    # --- English ---
    # Middle: kernel curve
    plot_poly6_kernel_curve(axs[1], h=h, num_samples=200)

    # --- Deutsch ---
    # Rechts: Dichte als Farbe
    #
    # --- English ---
    # Right: density as color
    plot_density(axs[2], particles.x, particles.y, rho, focus_index=focus_index, h=h)

    # --- Deutsch ---
    # Großer Titel über der gesamten Figure
    #
    # --- English ---
    # Large title above the entire figure
    fig_suptitle = fig.suptitle(
        "SPH Learning Viz: Partikel-Layout, Poly6-Kernel, naive Dichte",
        fontsize=14,
        y=0.98,
    )

    # --- Deutsch ---
    # Kleine Parameter-Zusammenfassung als Untertitel (Textbox oben)
    #
    # --- English ---
    # Small parameter summary as a subtitle (textbox at the top)
    param_summary_text = fig.text(
        0.5,
        0.94,
        f"L={L}, dx={dx}, h={h}, N={N}, rho0={rho0}, m={mass_per_particle}, focus_index={focus_index}",
        ha="center",
        va="top",
        fontsize=10,
    )

    # --- Deutsch ---
    # Layout verbessern: Platz für Titel/Untertitel lassen
    #
    # --- English ---
    # Improve layout: keep space for title/subtitle
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    # region agent log
    # --- Deutsch ---
    # Debugging der Text-Überlappung:
    # Wir erzwingen ein "draw", damit Matplotlib die finalen Positionen/BBoxes berechnet.
    # Dann loggen wir die Bounding-Boxes (in Figure-Koordinaten), um Überlappungen zu sehen.
    #
    # --- English ---
    # Debugging text overlap:
    # We force a draw so Matplotlib computes final positions/bounding boxes.
    # Then we log bounding boxes (in figure coordinates) to see overlaps.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _bbox_in_fig_coords(artist) -> list[float]:
        bb = artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        return [float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1)]

    legend0 = axs[0].get_legend()

    _agent_debug_log(
        hypothesis_id="E",
        location="learning_viz.py:run_learning_viz",
        message="text/legend bounding boxes (figure coords) after tight_layout",
        data={
            "fig_size_inches": [float(v) for v in fig.get_size_inches()],
            "dpi": float(fig.dpi),
            "tight_layout_rect": [0.0, 0.0, 1.0, 0.92],
            "suptitle_bbox": _bbox_in_fig_coords(fig_suptitle) if fig_suptitle is not None else None,
            "param_summary_bbox": _bbox_in_fig_coords(param_summary_text),
            "left_panel_text_bbox": _bbox_in_fig_coords(left_panel_text),
            "legend0_bbox": _bbox_in_fig_coords(legend0) if legend0 is not None else None,
        },
    )
    # endregion agent log

    # --- Deutsch ---
    # Anzeigen (kein File-I/O, keine Prints)
    #
    # --- English ---
    # Display (no file I/O, no prints)
    plt.show()


if __name__ == "__main__":
    run_learning_viz()


