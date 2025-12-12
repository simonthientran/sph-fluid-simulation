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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation

import json
import time

from sph_sim.core.particles import ParticleSet2D, initialize_particles_cube
from sph_sim.core.kernels import poly6_kernel
from sph_sim.core.density import compute_density_naive
from sph_sim.core.pressure import compute_pressure_eos
from sph_sim.core.neighbor_search import build_uniform_grid, query_neighbor_candidates
from sph_sim.core.forces import compute_pressure_acceleration
from sph_sim.core.integration import step_semi_implicit_euler

try:
    from sph_sim.core.density_grid import compute_density_grid
except Exception:
    compute_density_grid = None


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


def _compute_shared_xy_limits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    focus_index: int,
    h: float,
    dx: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    DE:
    Berechne gemeinsame Achsen-Limits (xlim/ylim) für ALLE Partikel-Plots.

    Ziel:
    - Alle Diagramme, die Partikel in x/y zeigen, sollen denselben sichtbaren Bereich haben.
    - Dann sind die Plots direkt vergleichbar ("gleiche Werte / gleiche Skala").

    Heuristik:
    - Wir nehmen den gesamten Partikelbereich (min/max von x und y).
    - Wir geben einen kleinen Rand (dx/2), damit die Punkte nicht "am Rand kleben".
    - Wir stellen sicher, dass der Radius h um den Fokus komplett sichtbar ist.
    - Wir "snappen" die Grenzen auf ein dx-Raster (ruhiger, gleichmäßiger Look).

    EN:
    Compute shared axis limits (xlim/ylim) for ALL particle plots.

    Goal:
    - All diagrams that show particles in x/y should use the same visible area.
    - Then plots are directly comparable ("same values / same scale").

    Heuristic:
    - Use the full particle domain (min/max of x and y).
    - Add a small margin (dx/2) so points do not stick to the border.
    - Ensure the radius h around the focus is fully visible.
    - Snap bounds to a dx grid (calmer, more uniform look).
    """

    dx_f = float(dx)
    h_f = float(h)
    if dx_f <= 0.0:
        raise ValueError("dx muss > 0 sein.")
    if h_f <= 0.0:
        raise ValueError("h muss > 0 sein.")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    y_min = float(np.min(y_arr))
    y_max = float(np.max(y_arr))

    x_focus = float(x_arr[int(focus_index)])
    y_focus = float(y_arr[int(focus_index)])

    pad = 0.5 * dx_f

    x_needed_min = min(x_min - pad, x_focus - h_f)
    x_needed_max = max(x_max + pad, x_focus + h_f)
    y_needed_min = min(y_min - pad, y_focus - h_f)
    y_needed_max = max(y_max + pad, y_focus + h_f)

    # --- Deutsch ---
    # Auf dx-Raster "snappen" für ruhige, gleichmäßige Achsen.
    #
    # --- English ---
    # Snap to dx grid for calm, uniform axes.
    x0 = float(np.floor(x_needed_min / dx_f) * dx_f)
    x1 = float(np.ceil(x_needed_max / dx_f) * dx_f)
    y0 = float(np.floor(y_needed_min / dx_f) * dx_f)
    y1 = float(np.ceil(y_needed_max / dx_f) * dx_f)

    return (x0, x1), (y0, y1)


def _apply_shared_xy_limits(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """
    DE:
    Wende gemeinsame Achsen-Limits auf eine Achse an.

    EN:
    Apply shared axis limits to an axis.
    """

    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_aspect("equal", adjustable="box")


def _link_xy_axes(axes: list) -> None:
    """
    DE:
    "Verknüpfe" mehrere Matplotlib-Achsen:
    - Wenn du in EINEM Plot zoomst/pannst, übernehmen die anderen denselben x/y Ausschnitt.

    Hinweis:
    - Matplotlib unterstützt `sharex/sharey` nur sauber innerhalb einer Figure.
    - Für mehrere Figures nutzen wir Callbacks.

    EN:
    "Link" multiple Matplotlib axes:
    - If you zoom/pan in ONE plot, the others adopt the same x/y view.

    Note:
    - Matplotlib supports `sharex/sharey` cleanly only within one figure.
    - Across multiple figures we use callbacks.
    """

    axes_clean = [ax for ax in axes if ax is not None]
    if len(axes_clean) <= 1:
        return

    guard = {"busy": False}

    def _sync_from(source_ax) -> None:
        if guard["busy"]:
            return
        guard["busy"] = True
        try:
            xlim = source_ax.get_xlim()
            ylim = source_ax.get_ylim()
            for ax in axes_clean:
                if ax is source_ax:
                    continue
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.figure.canvas.draw_idle()
        finally:
            guard["busy"] = False

    for ax in axes_clean:
        ax.callbacks.connect("xlim_changed", _sync_from)
        ax.callbacks.connect("ylim_changed", _sync_from)


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


def plot_density(
    ax,
    x,
    y,
    rho,
    focus_index: int,
    h: float,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
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
    # Zusatz (für konsistente Plots):
    # - Optional können wir vmin/vmax setzen, damit mehrere Dichte-Plots
    #   exakt dieselbe Farbskala verwenden (direkt vergleichbar).
    #
    # --- English ---
    # Scatter with a color scale (c=rho)
    #
    # Extra (for consistent plots):
    # - Optionally we can set vmin/vmax so multiple density plots use
    #   exactly the same color scale (directly comparable).
    sc = ax.scatter(x, y, c=rho, s=45, cmap="viridis", vmin=vmin, vmax=vmax)

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
        "Density is computed from neighboring particles via the kernel\n\n"
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
    ax.set_title("SPH-Dichte (Farbe) + Fokus + Radius h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")


def plot_scalar_field(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    focus_index: int | None = None,
    h: float | None = None,
) -> None:
    """
    DE:
    Plot: Skalarfeld pro Partikel als Farbe (z.B. rho oder |Δrho|).

    Warum diese Extra-Funktion?
    - `plot_density(...)` ist sehr didaktisch und enthält absichtlich Textbox + Formel.
    - Für Vergleiche (naiv vs grid vs Differenz) wollen wir aber eine ruhigere Darstellung:
      gleiche Achsen, gleiche Marker, nur andere Farben.

    Optional:
    - Wenn focus_index und h gegeben sind, zeichnen wir Fokus + Kreis h als Orientierung.

    EN:
    Plot: scalar field per particle as color (e.g., rho or |Δrho|).

    Why this extra function?
    - `plot_density(...)` is very didactic and intentionally includes a textbox + formula.
    - For comparisons (naive vs grid vs difference) we want a calmer visualization:
      same axes, same markers, only different colors.

    Optional:
    - If focus_index and h are provided, we draw focus + h circle as orientation.
    """

    # --- Deutsch ---
    # Scatter: jeder Partikel bekommt eine Farbe entsprechend `values[i]`.
    #
    # Zusatz (für konsistente Plots):
    # - Optional können wir vmin/vmax oder eine Normierung (`norm`) setzen,
    #   damit mehrere Plots exakt dieselbe Farbskala verwenden.
    #
    # --- English ---
    # Scatter: each particle gets a color according to `values[i]`.
    #
    # Extra (for consistent plots):
    # - Optionally we can set vmin/vmax or a normalization (`norm`)
    #   so multiple plots use exactly the same color scale.
    if norm is None:
        sc = ax.scatter(x, y, c=values, s=45, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        sc = ax.scatter(x, y, c=values, s=45, cmap=cmap, norm=norm)

    # --- Deutsch ---
    # Colorbar: erklärt die Farbskala.
    #
    # --- English ---
    # Colorbar: explains the color scale.
    cbar = ax.figure.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)

    # --- Deutsch ---
    # Optional: Fokus + Kreis h als Orientierung (keine zusätzliche Physik).
    #
    # --- English ---
    # Optional: focus + h circle as orientation (no additional physics).
    if focus_index is not None and h is not None:
        ax.scatter(
            [x[int(focus_index)]],
            [y[int(focus_index)]],
            s=140,
            c="tab:red",
            marker="*",
            zorder=5,
        )
        circle = Circle(
            (float(x[int(focus_index)]), float(y[int(focus_index)])),
            radius=float(h),
            fill=False,
            color="tab:red",
            linewidth=1.8,
            alpha=0.75,
            zorder=4,
        )
        ax.add_patch(circle)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)


def run_learning_viz(
    L: float = 1.0,
    dx: float = 0.1,
    rho0: float = 1000.0,
    mass_per_particle: float = 0.01,
    h: float = 0.15,
    focus_index: int = 0,
    use_synthetic_pressure: bool = False,
    n_steps: int = 500,
    dt: float = 0.001,
    damping: float = 0.02,
    show_focus_trajectory: bool = True,
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

    Zusatz (Mini-Simulation, nur Demo):
    - Optional läuft am Ende ein kleiner Simulationsloop (keine Animation), der pro Schritt
      `rho → p → a → Integration` ausführt und dann Start/Ende gegenüberstellt.
    - Das ist nur eine Demonstration, kein “voller SPH-Solver”.

    Additional (mini simulation, demo only):
    - Optionally, a small simulation loop runs at the end (no animation) that performs
      `rho → p → a → integration` per step and then compares start vs end.
    - This is only a demonstration, not a “full SPH solver”.
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
    if int(n_steps) <= 0:
        raise ValueError("n_steps muss > 0 sein.")
    _validate_positive("dt", dt)
    if float(damping) < 0.0 or float(damping) >= 1.0:
        raise ValueError("damping muss in [0, 1) liegen.")

    # --- Deutsch ---
    # Demo-Defaults (warum gerade dx=0.1 und h=1.5*dx?):
    # - dx=0.1 → genügend Partikel (bei L=1.0 sind es typischerweise 10×10 = 100), also kein 2×2-Grid.
    # - h=1.5*dx (bei den Defaults: 0.15) → jeder Partikel hat mehrere Nachbarn im Einflussradius,
    #   dadurch entstehen sichtbare Dichteunterschiede (Rand vs. Innen) statt "alles gleich".
    #
    # --- English ---
    # Demo defaults (why dx=0.1 and h=1.5*dx?):
    # - dx=0.1 → enough particles (for L=1.0 this is typically 10×10 = 100), i.e. not a 2×2 grid.
    # - h=1.5*dx (for the defaults: 0.15) → each particle has multiple neighbors within the influence radius,
    #   which yields visible density differences (boundary vs interior) instead of "everything equal".

    # --- Deutsch ---
    # Demo-Defaults (didaktisch):
    # - dx klein → viele Partikel → es gibt "Innen" und "Rand" → Dichteunterschiede werden sichtbar.
    # - h etwas größer als dx (hier: h ≈ 1.5*dx) → mehr Nachbarn tragen zur Dichte bei,
    #   dadurch entstehen stärkere und glattere Dichtevariationen.
    #
    # --- English ---
    # Demo defaults (teaching):
    # - small dx → many particles → we get "interior" and "boundary" → density differences become visible.
    # - h slightly larger than dx (here: h ≈ 1.5*dx) → more neighbors contribute to density,
    #   which creates stronger and smoother density variations.

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
    # -------------------------------------------------------------------------
    # Dichte für die Visualisierung korrekt berechnen und verwenden
    # -------------------------------------------------------------------------
    # WICHTIG:
    # - `particles.rho` enthält nur Initialwerte (rho0) und ist keine berechnete SPH-Dichte.
    # - Für Plots dürfen wir daher NICHT `particles.rho` nutzen.
    # - Wir berechnen die Dichte explizit mit SPH:
    #   bevorzugt grid-basiert, sonst naiv.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Correctly compute and use density for visualization
    # -------------------------------------------------------------------------
    # IMPORTANT:
    # - `particles.rho` contains only initial values (rho0) and is not a computed SPH density.
    # - Therefore we must NOT use `particles.rho` for plots.
    # - We compute density explicitly via SPH:
    #   prefer grid-based, otherwise naive.
    if compute_density_grid is not None:
        rho_grid = compute_density_grid(particles=particles, h=h, cell_size=h)
        rho = rho_grid
    else:
        rho_grid = None
        rho = compute_density_naive(particles=particles, h=h)

    # --- Deutsch ---
    # Zusätzlich (Vergleich / Debug):
    # - Wir berechnen auch die naive Referenzdichte.
    # - Damit können wir Unterschiede visualisieren und zeigen: gleiche Physik, andere Nachbarsuche.
    #
    # --- English ---
    # Additionally (comparison / debug):
    # - We also compute the naive reference density.
    # - This lets us visualize differences and show: same physics, different neighbor search.
    rho_naive = compute_density_naive(particles=particles, h=h)
    if rho_grid is not None:
        rho_diff_abs = np.abs(rho_naive - rho_grid)
        rho_diff_abs_max = float(np.max(rho_diff_abs))
    else:
        rho_diff_abs = np.abs(rho_naive - rho_naive)
        rho_diff_abs_max = float(np.max(rho_diff_abs))

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Gemeinsame Farbskala für Dichte-Plots (rho)
    # -------------------------------------------------------------------------
    # Ziel:
    # - Wenn wir mehrere Dichtebilder vergleichen (naiv vs grid vs andere Plots),
    #   sollen gleiche Farben auch wirklich gleiche Werte bedeuten.
    #
    # Umsetzung:
    # - Wir definieren vmin/vmax einmal global aus allen relevanten rho-Arrays.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Shared color scale for density plots (rho)
    # -------------------------------------------------------------------------
    # Goal:
    # - When comparing multiple density images (naive vs grid vs other plots),
    #   the same colors should truly mean the same values.
    #
    # Implementation:
    # - We define vmin/vmax once globally from all relevant rho arrays.
    if rho_grid is not None:
        rho_vmin = float(min(float(np.min(rho_naive)), float(np.min(rho_grid))))
        rho_vmax = float(max(float(np.max(rho_naive)), float(np.max(rho_grid))))
    else:
        rho_vmin = float(np.min(rho_naive))
        rho_vmax = float(np.max(rho_naive))

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Einheitliche Referenzdichte für ALLE Diagramme (Demo/Visualisierung)
    # -------------------------------------------------------------------------
    # Ziel:
    # - Alle Plots sollen auf denselben Daten basieren und logisch zusammenhängen.
    #
    # Wir wählen deshalb:
    #   rho0_ref = mean(rho)
    #
    # Warum ist das didaktisch gut?
    # - Dann ist drho = rho - rho0_ref automatisch um 0 zentriert:
    #   einige Partikel haben drho < 0 (unter dem Mittel), andere drho > 0 (über dem Mittel).
    # - Dadurch sehen wir im Demo klar BOTH:
    #   - negative Abweichung → negativer Druck
    #   - positive Abweichung → positiver Druck
    #
    # WICHTIG:
    # - Das ist nur für Demo/Visualisierung.
    # - Eine echte Simulation nutzt später ein festes rho0 (z.B. Wasser: 1000 kg/m^3)
    #   und kalibriert Massen/Kernel/Units entsprechend.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # One shared reference density for ALL diagrams (demo/visualization)
    # -------------------------------------------------------------------------
    # Goal:
    # - All plots should be based on the same data and be logically connected.
    #
    # Therefore we choose:
    #   rho0_ref = mean(rho)
    #
    # Why is this didactically useful?
    # - Then drho = rho - rho0_ref is automatically centered around 0:
    #   some particles have drho < 0 (below the mean), others drho > 0 (above the mean).
    # - This makes it easy to clearly show BOTH in the demo:
    #   - negative deviation → negative pressure
    #   - positive deviation → positive pressure
    #
    # IMPORTANT:
    # - This is demo/visualization only.
    # - A real simulation later uses a fixed rho0 (e.g., water: 1000 kg/m^3)
    #   and calibrates masses/kernel/units accordingly.
    rho0_ref = float(np.mean(rho))

    # --- Deutsch ---
    # Dichteabweichung:
    # drho = rho - rho0_ref
    #
    # --- English ---
    # Density deviation:
    # drho = rho - rho0_ref
    drho = rho - float(rho0_ref)

    # --- Deutsch ---
    # EOS-Parameter k (Demo-Default):
    # - Relativ klein, damit die Bewegung bei vielen Steps ruhiger bleibt.
    # - Das ist nur eine Skalierung im Demo (keine Änderung an der Kernphysik).
    #
    # --- English ---
    # EOS parameter k (demo default):
    # - Relatively small so motion stays calmer over many steps.
    # - This is only a demo scaling choice (no change to core physics).
    k = 25.0

    # --- Deutsch ---
    # Druck aus der EOS, ohne Clamping:
    #
    # p = k * (rho - rho0_ref) = k * drho
    #
    # Wichtig (Didaktik):
    # - p ist DIREKT proportional zu drho.
    # - Deshalb müssen drho und p die gleiche "Struktur" zeigen (nur skaliert).
    #
    # --- English ---
    # Pressure from EOS, without clamping:
    #
    # p = k * (rho - rho0_ref) = k * drho
    #
    # Important (teaching):
    # - p is DIRECTLY proportional to drho.
    # - Therefore drho and p must show the same "structure" (just scaled).
    p = compute_pressure_eos(rho=rho, rho0=float(rho0_ref), k=float(k), clamp_negative=False)

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Debug-Option: synthetischer Druck für "garantiert sichtbare" Druckkräfte
    # -------------------------------------------------------------------------
    # Motivation:
    # - Manchmal ist die echte EOS-Druckverteilung (aus rho) für ein perfektes Gitter
    #   sehr "ruhig" oder symmetrisch, sodass die Kräfte im Plot klein wirken.
    #
    # Debug-Idee:
    # - Wir definieren künstlich einen Druck, der radial nach außen abnimmt:
    #
    #     p_synth = -r
    #
    #   wobei r der Abstand zum Zentrum (L/2, L/2) ist.
    #
    # Warum ist der Druck in der Mitte "am höchsten"?
    # - Bei r = 0 (Zentrum) ist p = 0.
    # - Für r > 0 ist p negativ.
    # - Also ist p im Zentrum maximal (am wenigsten negativ).
    #
    # Erwartung (Didaktik):
    # - Druck ist in der Mitte höher als außen.
    # - Druckkräfte treiben (tendenziell) von hohem Druck zu niedrigem Druck.
    # - Daher sollten die Beschleunigungs-Pfeile im Plot nach außen zeigen.
    #
    # WICHTIG:
    # - Das ist NUR Debug/Visualisierung.
    # - Das überschreibt die EOS-Pipeline (rho → drho → p) bewusst.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Debug option: synthetic pressure for "guaranteed visible" pressure forces
    # -------------------------------------------------------------------------
    # Motivation:
    # - Sometimes the real EOS pressure field (from rho) for a perfect grid
    #   is very "calm" or symmetric, so forces look small in the plot.
    #
    # Debug idea:
    # - We define an artificial pressure that decreases radially outward:
    #
    #     p_synth = -r
    #
    #   where r is the distance to the domain center (L/2, L/2).
    #
    # Why is pressure "highest" in the center?
    # - At r = 0 (center), p = 0.
    # - For r > 0, p is negative.
    # - Therefore the center has the maximum p (least negative).
    #
    # Expectation (teaching):
    # - Pressure is higher in the center than outside.
    # - Pressure forces (tend to) push from high pressure to low pressure.
    # - Therefore the acceleration arrows in the plot should point outward.
    #
    # IMPORTANT:
    # - This is debug/visualization only.
    # - It intentionally overrides the EOS pipeline (rho → drho → p).
    p_for_force = p
    if bool(use_synthetic_pressure):
        x_center = 0.5 * float(L)
        y_center = 0.5 * float(L)
        r_center = np.sqrt((particles.x - x_center) ** 2 + (particles.y - y_center) ** 2)
        p_for_force = -np.asarray(r_center, dtype=np.float64)

    # --- Deutsch ---
    # Für Diagnose/Plot: min/max von rho, drho, p als Zahlen.
    #
    # --- English ---
    # For diagnostics/plots: numeric min/max of rho, drho, p.
    rho_min = float(np.min(rho)) if rho.size > 0 else 0.0
    rho_max = float(np.max(rho)) if rho.size > 0 else 0.0
    drho_min = float(np.min(drho)) if drho.size > 0 else 0.0
    drho_max = float(np.max(drho)) if drho.size > 0 else 0.0
    p_min = float(np.min(p)) if p.size > 0 else 0.0
    p_max = float(np.max(p)) if p.size > 0 else 0.0

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Druckbeschleunigung (Kräfte) berechnen und später als Pfeile visualisieren
    # -------------------------------------------------------------------------
    # Jetzt kommt der nächste Schritt: aus dem Druck p werden Kräfte/Accelerations.
    #
    # WICHTIG (Didaktik):
    # - Wir nutzen hier explizit:
    #   - rho = unsere (grid-basierte) SPH-Dichte
    #   - p = Druck aus der EOS (ungeclamped, kann negativ sein)
    # - Das ist didaktisch wichtig, weil so negative/positive Druckwerte sichtbar bleiben.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Compute pressure acceleration (forces) and later visualize as arrows
    # -------------------------------------------------------------------------
    # Now we take the next step: from pressure p we compute forces/accelerations.
    #
    # IMPORTANT (teaching):
    # - We explicitly use:
    #   - rho = our (grid-based) SPH density
    #   - p = pressure from EOS (unclamped, can be negative)
    # - This is didactically important because negative/positive pressures stay visible.
    ax_pressure, ay_pressure = compute_pressure_acceleration(
        particles=particles,
        rho=rho,
        p=p_for_force,
        h=float(h),
        cell_size=float(cell_size),
    )

    # --- Deutsch ---
    # Pfeil-Skalierung (arrow_scale):
    #
    # Problem:
    # - (ax, ay) haben physikalische Einheiten (Beschleunigung).
    # - Unsere Plot-Achsen sind Positions-Einheiten (x/y im Quadrat [0, L)).
    # - Wenn wir Beschleunigungen "1:1" als Pfeile zeichnen, sind sie meistens
    #   viel zu klein oder viel zu groß, je nach Parametern.
    #
    # Lösung:
    # - Wir skalieren die Pfeillängen mit einem Faktor arrow_scale, rein für Lesbarkeit.
    # - Das ändert keine Physik – es ist nur eine Visualisierungs-Hilfe.
    #
    # --- English ---
    # Arrow scaling (arrow_scale):
    #
    # Problem:
    # - (ax, ay) have physical units (acceleration).
    # - Our plot axes are position units (x/y in the square [0, L)).
    # - If we draw acceleration vectors "1:1", arrows are usually
    #   far too small or far too large depending on parameters.
    #
    # Solution:
    # - We scale arrow lengths with a factor arrow_scale purely for readability.
    # - This changes no physics – it is only a visualization aid.
    # --- Deutsch ---
    # Auto-Skalierung (praktischer als ein fixer Wert):
    # - Wir wählen eine Ziel-Pfeillänge in "Plot-Einheiten" (also ungefähr in x/y-Einheiten).
    # - Dann skalieren wir so, dass der größte (gezeichnete) Beschleunigungsvektor ungefähr
    #   diese Ziel-Länge bekommt.
    #
    # Vorteil:
    # - Egal ob deine Parameter "kleine" oder "große" Beschleunigungen erzeugen:
    #   du siehst im Plot immer etwas.
    #
    # --- English ---
    # Auto scaling (more practical than a fixed value):
    # - We choose a target arrow length in "plot units" (roughly x/y units).
    # - Then we scale so that the largest (drawn) acceleration vector gets roughly
    #   that target length.
    #
    # Advantage:
    # - No matter whether your parameters produce "small" or "large" accelerations:
    #   you will always see something in the plot.
    a_mag = np.sqrt(ax_pressure * ax_pressure + ay_pressure * ay_pressure)

    # --- Deutsch ---
    # Pfeil-Skalierung, die garantiert "sichtbar" ist:
    #
    # Idee:
    # - Wir möchten, dass der größte Pfeil ungefähr eine feste Länge im Plot hat (z.B. 0.15).
    # - Dafür skalieren wir alle Pfeile mit:
    #
    #     arrow_mult = 0.15 / max(|a|)
    #
    # Wichtig:
    # - Das ist nur Visualisierung (Einheiten/Lesbarkeit), KEINE Physikänderung.
    #
    # --- English ---
    # Arrow scaling that makes arrows reliably "visible":
    #
    # Idea:
    # - We want the largest arrow to have roughly a fixed length in plot units (e.g. 0.15).
    # - Therefore we scale all arrows with:
    #
    #     arrow_mult = 0.15 / max(|a|)
    #
    # Important:
    # - This is visualization only (units/readability), NOT a physics change.
    a_mag_max = float(np.max(a_mag)) if a_mag.size > 0 else 0.0
    arrow_mult = 1.0
    if a_mag_max > 0.0:
        arrow_mult = 0.15 / a_mag_max
    else:
        arrow_mult = 0.0

    # --- Deutsch ---
    # Wir verwenden die Variable arrow_scale später beim Quiver-Plot.
    # (Kompatibel mit dem bestehenden Code – wir setzen sie auf unseren arrow_mult.)
    #
    # --- English ---
    # We use the variable arrow_scale later in the quiver plot.
    # (Compatible with the existing code – we set it to our arrow_mult.)
    arrow_scale = float(arrow_mult)

    # --- Deutsch ---
    # --- Deutsch ---
    # Hinweis:
    # - Wir haben p bereits als ungeclamp-ten Druck berechnet.
    # - Dieser Druck enthält negative und positive Werte und ist die Grundlage für die Kraft-Demo.
    #
    # --- English ---
    # Note:
    # - We already computed p as an unclamped pressure.
    # - This pressure contains negative and positive values and is the basis for the force demo.

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
    plot_density(
        axs[2],
        particles.x,
        particles.y,
        rho,
        focus_index=focus_index,
        h=h,
        vmin=rho_vmin,
        vmax=rho_vmax,
    )

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
    # -------------------------------------------------------------------------
    # Zusätzliche Vergleichs-Figure: naive vs grid vs |Δrho|
    # -------------------------------------------------------------------------
    # Wir entfernen nichts am bestehenden Lern-Plot.
    # Stattdessen erstellen wir eine zweite Figure, die nur den Dichtevergleich zeigt.
    #
    # Plot A: naive rho
    # Plot B: grid rho
    # Plot C: |rho_naive − rho_grid|
    #
    # In Plot C zeigen wir eine Textbox:
    # - max(|Δρ|)
    # - kurzer Merksatz: "Same physics, fewer neighbor checks"
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Additional comparison figure: naive vs grid vs |Δrho|
    # -------------------------------------------------------------------------
    # We do not remove anything from the existing learning plot.
    # Instead we create a second figure that focuses only on the density comparison.
    #
    # Plot A: naive rho
    # Plot B: grid rho
    # Plot C: |rho_naive − rho_grid|
    #
    # In plot C we show a textbox:
    # - max(|Δρ|)
    # - short takeaway: "Same physics, fewer neighbor checks"
    fig_cmp, axs_cmp = plt.subplots(1, 3, figsize=(18, 6))

    plot_scalar_field(
        axs_cmp[0],
        particles.x,
        particles.y,
        rho_naive,
        title="Plot A: naive rho",
        cbar_label="rho",
        cmap="viridis",
        vmin=rho_vmin,
        vmax=rho_vmax,
        focus_index=focus_index,
        h=h,
    )
    # --- Deutsch ---
    # Zusätzliche Zahlenangabe: min/max der Dichte (naiv) direkt als Textbox.
    #
    # --- English ---
    # Additional numeric annotation: min/max density (naive) as a textbox.
    axs_cmp[0].text(
        0.02,
        0.98,
        "min(rho) = {mn}\nmax(rho) = {mx}".format(
            mn=f"{float(np.min(rho_naive)):.12g}",
            mx=f"{float(np.max(rho_naive)):.12g}",
        ),
        transform=axs_cmp[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    plot_scalar_field(
        axs_cmp[1],
        particles.x,
        particles.y,
        rho_grid if rho_grid is not None else rho_naive,
        title="Plot B: grid rho",
        cbar_label="rho",
        cmap="viridis",
        vmin=rho_vmin,
        vmax=rho_vmax,
        focus_index=focus_index,
        h=h,
    )
    # --- Deutsch ---
    # Zusätzliche Zahlenangabe: min/max der Dichte (grid) direkt als Textbox.
    #
    # --- English ---
    # Additional numeric annotation: min/max density (grid) as a textbox.
    axs_cmp[1].text(
        0.02,
        0.98,
        "min(rho) = {mn}\nmax(rho) = {mx}".format(
            mn=f"{float(np.min(rho_grid if rho_grid is not None else rho_naive)):.12g}",
            mx=f"{float(np.max(rho_grid if rho_grid is not None else rho_naive)):.12g}",
        ),
        transform=axs_cmp[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    plot_scalar_field(
        axs_cmp[2],
        particles.x,
        particles.y,
        rho_diff_abs,
        title="Plot C: |rho_naive − rho_grid|",
        cbar_label="|Δρ|",
        cmap="magma",
        vmin=0.0,
        vmax=float(rho_diff_abs_max),
        focus_index=focus_index,
        h=h,
    )

    axs_cmp[2].text(
        0.02,
        0.98,
        "max(|Δρ|) = {v}\nSame physics, fewer neighbor checks".format(v=f"{rho_diff_abs_max:.6g}"),
        transform=axs_cmp[2].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    fig_cmp.suptitle("SPH Density Comparison: naive vs grid", fontsize=14, y=0.98)
    fig_cmp.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Zusätzliche Demo-Figure: rho, p und drho als Farben (anschaulich)
    # -------------------------------------------------------------------------
    # Ziel:
    # - Ein Plot zeigt Dichte rho als Farbe.
    # - Ein Plot zeigt Druck p als Farbe (kann negativ/positiv sein).
    # - Ein Plot zeigt drho = rho - mean(rho) als Farbe (positiv/negativ).
    #
    # Wichtig:
    # - Das ist nur Darstellung/Didaktik. Keine neue Physik.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Additional demo figure: rho, p and drho as colors (intuitive)
    # -------------------------------------------------------------------------
    # Goal:
    # - One plot shows density rho as color.
    # - One plot shows pressure p as color (can be negative/positive).
    # - One plot shows drho = rho - mean(rho) as color (positive/negative).
    #
    # Important:
    # - This is visualization/teaching only. No new physics.
    fig_demo, axs_demo = plt.subplots(1, 3, figsize=(18, 6))

    plot_scalar_field(
        axs_demo[0],
        particles.x,
        particles.y,
        rho,
        title="Density rho (color)",
        cbar_label="density rho",
        cmap="viridis",
        vmin=rho_vmin,
        vmax=rho_vmax,
        focus_index=focus_index,
        h=h,
    )
    axs_demo[0].text(
        0.02,
        0.98,
        "min(rho) = {mn}\nmax(rho) = {mx}\n\nDensity is computed from neighboring particles via the kernel\n\nrho0_ref = {r0}\nk = {k}".format(
            mn=f"{float(np.min(rho)):.12g}",
            mx=f"{float(np.max(rho)):.12g}",
            r0=f"{float(rho0_ref):.12g}",
            k=f"{float(k):.12g}",
        ),
        transform=axs_demo[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    # --- Deutsch ---
    # p kann positiv/negativ sein → divergierende Farbskala, zentriert bei 0.
    #
    # --- English ---
    # p can be positive/negative → diverging colormap centered at 0.
    p_abs_max = float(np.max(np.abs(p))) if p.size > 0 else 0.0

    # --- Deutsch ---
    # Gemeinsame Normierung für alle Plots, die p zeigen:
    # - Wenn p_abs_max == 0 ist (alles 0), vermeiden wir TwoSlopeNorm-Probleme.
    #
    # --- English ---
    # Shared normalization for all plots that show p:
    # - If p_abs_max == 0 (all zeros), we avoid TwoSlopeNorm issues.
    if p_abs_max > 0.0:
        norm_p_shared = TwoSlopeNorm(vcenter=0.0, vmin=-p_abs_max, vmax=p_abs_max)
        sc_p = axs_demo[1].scatter(particles.x, particles.y, c=p, s=45, cmap="coolwarm", norm=norm_p_shared)
    else:
        norm_p_shared = None
        sc_p = axs_demo[1].scatter(particles.x, particles.y, c=p, s=45, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar_p = fig_demo.colorbar(sc_p, ax=axs_demo[1])
    cbar_p.set_label("pressure p (unclamped, can be negative)")
    axs_demo[1].set_aspect("equal", adjustable="box")
    axs_demo[1].set_title("Pressure p (color, unclamped)")
    axs_demo[1].set_xlabel("x")
    axs_demo[1].set_ylabel("y")
    axs_demo[1].grid(True, alpha=0.3)
    axs_demo[1].scatter(
        [particles.x[int(focus_index)]],
        [particles.y[int(focus_index)]],
        s=140,
        c="tab:red",
        marker="*",
        zorder=5,
    )
    circle_p = Circle(
        (float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])),
        radius=float(h),
        fill=False,
        color="tab:red",
        linewidth=1.8,
        alpha=0.75,
        zorder=4,
    )
    axs_demo[1].add_patch(circle_p)
    axs_demo[1].text(
        0.02,
        0.98,
        "min(p) = {mn}\nmax(p) = {mx}\n\nPressure is proportional to drho: p = k * drho\n\nrho0_ref = {r0}\nk = {k}".format(
            mn=f"{float(np.min(p)):.12g}",
            mx=f"{float(np.max(p)):.12g}",
            r0=f"{float(rho0_ref):.12g}",
            k=f"{float(k):.12g}",
        ),
        transform=axs_demo[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    # --- Deutsch ---
    # drho ist positiv und negativ → wir nehmen eine divergierende Farbskala und zentrieren bei 0.
    #
    # --- English ---
    # drho can be positive/negative → we use a diverging colormap centered at 0.
    drho_abs_max = float(np.max(np.abs(drho))) if drho.size > 0 else 0.0
    if drho_abs_max > 0.0:
        norm_drho_shared = TwoSlopeNorm(vcenter=0.0, vmin=-drho_abs_max, vmax=drho_abs_max)
        sc_drho = axs_demo[2].scatter(particles.x, particles.y, c=drho, s=45, cmap="coolwarm", norm=norm_drho_shared)
    else:
        norm_drho_shared = None
        sc_drho = axs_demo[2].scatter(particles.x, particles.y, c=drho, s=45, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar_drho = fig_demo.colorbar(sc_drho, ax=axs_demo[2])
    cbar_drho.set_label("density deviation drho = rho - mean(rho)")
    axs_demo[2].set_aspect("equal", adjustable="box")
    axs_demo[2].set_title("Density deviation drho (color)")
    axs_demo[2].set_xlabel("x")
    axs_demo[2].set_ylabel("y")
    axs_demo[2].grid(True, alpha=0.3)
    axs_demo[2].scatter(
        [particles.x[int(focus_index)]],
        [particles.y[int(focus_index)]],
        s=140,
        c="tab:red",
        marker="*",
        zorder=5,
    )
    circle_drho = Circle(
        (float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])),
        radius=float(h),
        fill=False,
        color="tab:red",
        linewidth=1.8,
        alpha=0.75,
        zorder=4,
    )
    axs_demo[2].add_patch(circle_drho)
    axs_demo[2].text(
        0.02,
        0.98,
        "min(drho) = {mn}\nmax(drho) = {mx}\n\nrho0_ref = {r0}\nk = {k}".format(
            mn=f"{float(np.min(drho)):.12g}",
            mx=f"{float(np.max(drho)):.12g}",
            r0=f"{float(rho0_ref):.12g}",
            k=f"{float(k):.12g}",
        ),
        transform=axs_demo[2].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    # --- Deutsch ---
    # Fokus-Info (kleine, direkte Zahlenwerte) als zusätzliche Textbox "beim Fokus-Partikel":
    # - rho_focus
    # - p_focus
    # - drho_focus
    #
    # --- English ---
    # Focus info (small, direct numeric values) as an additional textbox "near the focus particle":
    # - rho_focus
    # - p_focus
    # - drho_focus
    rho_focus = float(rho[int(focus_index)])
    p_focus = float(p[int(focus_index)])
    drho_focus = float(drho[int(focus_index)])
    axs_demo[0].annotate(
        "rho_focus = {r}\n"
        "p_focus = {p}\n"
        "drho_focus = {d}".format(r=f"{rho_focus:.12g}", p=f"{p_focus:.12g}", d=f"{drho_focus:.12g}"),
        xy=(float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])),
        xytext=(12, 12),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        arrowprops={"arrowstyle": "->", "color": "tab:red", "alpha": 0.6, "linewidth": 1.0},
    )

    fig_demo.suptitle("SPH Demo: density, pressure, density deviation", fontsize=14, y=0.98)
    fig_demo.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Zusätzliche Pressure-Figure: Druck sichtbar machen + EOS-Beziehung zeigen
    # -------------------------------------------------------------------------
    # Panel 1: Druck p als Farbe auf den Partikeln (Colorbar "pressure p").
    # Panel 2: Zusammenhang rho → p als Scatter (x=rho, y=p).
    #
    # Textbox:
    # - verwendete Parameter rho0 und k
    # - kurzer Satz: "EOS maps density error to pressure"
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Additional pressure figure: visualize pressure + show EOS relation
    # -------------------------------------------------------------------------
    # Panel 1: pressure p as color on particles (colorbar "pressure p").
    # Panel 2: relation rho → p as a scatter (x=rho, y=p).
    #
    # Textbox:
    # - used parameters rho0 and k
    # - short sentence: "EOS maps density error to pressure"
    fig_p, axs_p = plt.subplots(1, 2, figsize=(14, 6))

    # --- Deutsch ---
    # Für die EOS-Plot-Figure zeigen wir den un-geclamp-ten Demo-Druck,
    # damit negative Werte (bei niedriger Dichte) sichtbar sind.
    #
    # --- English ---
    # For the EOS plot figure we show the unclamped demo pressure
    # so negative values (at low density) become visible.
    # --- Deutsch ---
    # Wir nutzen dieselbe Normierung wie im vorherigen p-Plot,
    # damit "rot/blau" in allen Figuren exakt dieselbe Bedeutung hat.
    #
    # --- English ---
    # We reuse the same normalization as in the previous p plot
    # so that "red/blue" means the exact same values across all figures.
    if "norm_p_shared" in locals() and norm_p_shared is not None:
        sc_p2 = axs_p[0].scatter(particles.x, particles.y, c=p, s=45, cmap="coolwarm", norm=norm_p_shared)
    else:
        sc_p2 = axs_p[0].scatter(particles.x, particles.y, c=p, s=45, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar_p2 = fig_p.colorbar(sc_p2, ax=axs_p[0])
    cbar_p2.set_label("pressure p (demo, unclamped)")
    axs_p[0].set_aspect("equal", adjustable="box")
    axs_p[0].set_title("Pressure p (color, demo, unclamped)")
    axs_p[0].set_xlabel("x")
    axs_p[0].set_ylabel("y")
    axs_p[0].grid(True, alpha=0.3)
    axs_p[0].scatter(
        [particles.x[int(focus_index)]],
        [particles.y[int(focus_index)]],
        s=140,
        c="tab:red",
        marker="*",
        zorder=5,
    )
    circle_p2 = Circle(
        (float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])),
        radius=float(h),
        fill=False,
        color="tab:red",
        linewidth=1.8,
        alpha=0.75,
        zorder=4,
    )
    axs_p[0].add_patch(circle_p2)
    axs_p[0].text(
        0.02,
        0.98,
        "min(p) = {mn}\nmax(p) = {mx}\n\nPressure is proportional to drho: p = k * drho".format(
            mn=f"{float(np.min(p)):.12g}",
            mx=f"{float(np.max(p)):.12g}",
        ),
        transform=axs_p[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    # --- Deutsch ---
    # Scatter: jedes Partikel ist ein Punkt (rho_i, p_i).
    #
    # --- English ---
    # Scatter: each particle is one point (rho_i, p_i).
    axs_p[1].scatter(rho, p, s=35, alpha=0.85, color="tab:blue")
    axs_p[1].set_title("EOS: rho → p")
    axs_p[1].set_xlabel("rho")
    axs_p[1].set_ylabel("pressure p")
    axs_p[1].grid(True, alpha=0.3)

    axs_p[1].text(
        0.02,
        0.98,
        "rho0_ref = {rho0}\n"
        "k = {k}\n\n"
        "EOS maps density error to pressure".format(rho0=f"{float(rho0_ref):.12g}", k=f"{float(k):.12g}"),
        transform=axs_p[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    fig_p.suptitle("SPH Pressure via EOS", fontsize=14, y=0.98)
    fig_p.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Pipeline-Figure (als letzte Figure): rho → drho → p → a (Pfeile)
    # -------------------------------------------------------------------------
    # Diese Figure ist bewusst so gebaut, dass Anfänger die Pipeline direkt sehen:
    #
    # Panel 1: drho = rho - rho0_ref (Farben, +/-)
    # Panel 2: p = k * drho (Farben, +/-)
    # Panel 3: Beschleunigung a aus Druckunterschieden (Pfeile)
    #
    # WICHTIG:
    # - Alle Panels nutzen exakt dieselben Daten (rho0_ref, drho, p, ax/ay).
    # - Negative drho → negativer Druck (Unterdruck im Modell).
    # - Positive drho → positiver Druck (Überdruck im Modell).
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Pipeline figure (as the last figure): rho → drho → p → a (arrows)
    # -------------------------------------------------------------------------
    # This figure is intentionally built so beginners can see the pipeline directly:
    #
    # Panel 1: drho = rho - rho0_ref (colors, +/-)
    # Panel 2: p = k * drho (colors, +/-)
    # Panel 3: acceleration a from pressure differences (arrows)
    #
    # IMPORTANT:
    # - All panels use the exact same data (rho0_ref, drho, p, ax/ay).
    # - Negative drho → negative pressure (suction in this model).
    # - Positive drho → positive pressure (over-pressure in this model).
    fig_pipe, axs_pipe = plt.subplots(1, 3, figsize=(18, 6))

    # --- Deutsch ---
    # Fokuswerte (für alle Textboxen konsistent):
    #
    # --- English ---
    # Focus values (consistent for all textboxes):
    rho_focus = float(rho[int(focus_index)])
    drho_focus = float(drho[int(focus_index)])
    p_focus = float(p[int(focus_index)])
    a_focus = float(np.sqrt(float(ax_pressure[int(focus_index)]) ** 2 + float(ay_pressure[int(focus_index)]) ** 2))

    # --- Deutsch ---
    # Normierungen (diverging colormap, Zentrum bei 0):
    #
    # --- English ---
    # Normalizations (diverging colormap, centered at 0):
    if drho_abs_max > 0.0:
        norm_drho_pipe = norm_drho_shared
    else:
        norm_drho_pipe = None
    if p_abs_max > 0.0:
        norm_p_pipe = norm_p_shared
    else:
        norm_p_pipe = None

    # --- Deutsch ---
    # Panel 1: drho
    #
    # --- English ---
    # Panel 1: drho
    if norm_drho_pipe is not None:
        sc1 = axs_pipe[0].scatter(particles.x, particles.y, c=drho, s=40, cmap="coolwarm", norm=norm_drho_pipe)
    else:
        sc1 = axs_pipe[0].scatter(particles.x, particles.y, c=drho, s=40, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar1 = fig_pipe.colorbar(sc1, ax=axs_pipe[0])
    cbar1.set_label("drho = rho - rho0_ref")
    axs_pipe[0].scatter([float(particles.x[int(focus_index)])], [float(particles.y[int(focus_index)])], s=150, c="tab:red", zorder=5)
    axs_pipe[0].add_patch(
        Circle((float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])), radius=float(h), fill=False, color="tab:red", linewidth=1.6, alpha=0.7, zorder=4)
    )
    axs_pipe[0].set_title("drho on particles")
    axs_pipe[0].set_xlabel("x")
    axs_pipe[0].set_ylabel("y")
    axs_pipe[0].grid(True, alpha=0.25)
    axs_pipe[0].text(
        0.02,
        0.98,
        "min/max drho = {mn} / {mx}\n"
        "rho_focus = {rf}\n"
        "drho_focus = {df}\n"
        "p_focus = {pf}\n"
        "|a_focus| = {af}".format(
            mn=f"{float(drho_min):.12g}",
            mx=f"{float(drho_max):.12g}",
            rf=f"{float(rho_focus):.12g}",
            df=f"{float(drho_focus):.12g}",
            pf=f"{float(p_focus):.12g}",
            af=f"{float(a_focus):.12g}",
        ),
        transform=axs_pipe[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        zorder=10,
    )

    # --- Deutsch ---
    # Panel 2: p
    #
    # --- English ---
    # Panel 2: p
    if norm_p_pipe is not None:
        sc2 = axs_pipe[1].scatter(particles.x, particles.y, c=p, s=40, cmap="coolwarm", norm=norm_p_pipe)
    else:
        sc2 = axs_pipe[1].scatter(particles.x, particles.y, c=p, s=40, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar2 = fig_pipe.colorbar(sc2, ax=axs_pipe[1])
    cbar2.set_label("pressure p = k * drho (can be negative)")
    axs_pipe[1].scatter([float(particles.x[int(focus_index)])], [float(particles.y[int(focus_index)])], s=150, c="tab:red", zorder=5)
    axs_pipe[1].add_patch(
        Circle((float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])), radius=float(h), fill=False, color="tab:red", linewidth=1.6, alpha=0.7, zorder=4)
    )
    axs_pipe[1].set_title("pressure p on particles")
    axs_pipe[1].set_xlabel("x")
    axs_pipe[1].set_ylabel("y")
    axs_pipe[1].grid(True, alpha=0.25)
    axs_pipe[1].text(
        0.02,
        0.98,
        "min/max p = {mn} / {mx}\n"
        "k = {k}\n"
        "rho0_ref = {r0}\n\n"
        "rho_focus = {rf}\n"
        "drho_focus = {df}\n"
        "p_focus = {pf}\n"
        "|a_focus| = {af}".format(
            mn=f"{float(p_min):.12g}",
            mx=f"{float(p_max):.12g}",
            k=f"{float(k):.6g}",
            r0=f"{float(rho0_ref):.12g}",
            rf=f"{float(rho_focus):.12g}",
            df=f"{float(drho_focus):.12g}",
            pf=f"{float(p_focus):.12g}",
            af=f"{float(a_focus):.12g}",
        ),
        transform=axs_pipe[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        zorder=10,
    )

    # --- Deutsch ---
    # Panel 3: Pfeile (Beschleunigung)
    #
    # Regeln:
    # - Wenn N <= 400: Pfeile für alle Partikel.
    # - Sonst: nur Fokus + Nachbarn (damit es nicht zu voll wird).
    #
    # Pfeile sichtbar machen:
    # - Wir skalieren rein visuell, damit der größte Pfeil ungefähr Länge 0.25 hat.
    #
    # --- English ---
    # Panel 3: arrows (acceleration)
    #
    # Rules:
    # - If N <= 400: arrows for all particles.
    # - Otherwise: focus + neighbors only (to avoid clutter).
    #
    # Make arrows visible:
    # - We scale purely visually so the largest arrow has roughly length 0.25.
    if a_mag_max > 0.0:
        arrow_mult_pipe = 0.25 / float(a_mag_max)
    else:
        arrow_mult_pipe = 1.0

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # TEIL A: Pfeillängen normieren (Richtung bleibt physikalisch korrekt)
    # -------------------------------------------------------------------------
    # Problem (für Anfänger verwirrend):
    # - Wenn wir Pfeile proportional zu |a| zeichnen, können sie am Rand sehr lang wirken
    #   (Randartefakte) und sich stark überlagern.
    #
    # Lösung (nur Visualisierung, keine Physikänderung):
    # - Wir berechnen zuerst den Betrag:
    #
    #     a_mag = sqrt(ax^2 + ay^2)
    #
    # - Dann normieren wir die Pfeile relativ zum Maximum:
    #
    #     ax_plot = ax / (a_mag.max() + 1e-12)
    #     ay_plot = ay / (a_mag.max() + 1e-12)
    #
    # Wichtig:
    # - Die Richtung der Pfeile ist physikalisch relevant (zeigt Beschleunigungsrichtung).
    # - Die Länge ist hier NUR skaliert/normiert, um das Bild ruhig und lesbar zu machen.
    # - Den absoluten Betrag |a| zeigen wir separat als Zahl (Textbox).
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # PART A: normalize arrow lengths (direction stays physically correct)
    # -------------------------------------------------------------------------
    # Problem (confusing for beginners):
    # - If we draw arrows proportional to |a|, they can look very long near boundaries
    #   (boundary artifacts) and heavily overlap.
    #
    # Solution (visualization only, no physics change):
    # - First compute the magnitude:
    #
    #     a_mag = sqrt(ax^2 + ay^2)
    #
    # - Then normalize arrows relative to the maximum:
    #
    #     ax_plot = ax / (a_mag.max() + 1e-12)
    #     ay_plot = ay / (a_mag.max() + 1e-12)
    #
    # Important:
    # - Arrow direction is physically relevant (shows acceleration direction).
    # - Arrow length is scaled/normalized here ONLY to keep the figure calm and readable.
    # - We show the absolute magnitude |a| separately as a number (textbox).
    a_mag_pipe = np.sqrt(ax_pressure * ax_pressure + ay_pressure * ay_pressure)
    a_mag_max_pipe = float(np.max(a_mag_pipe)) if a_mag_pipe.size > 0 else 0.0
    denom_pipe = float(a_mag_max_pipe) + 1e-12
    ax_plot = ax_pressure / denom_pipe
    ay_plot = ay_pressure / denom_pipe

    # --- Deutsch ---
    # Wir setzen eine moderate Ziellänge für den größten Pfeil (nur Optik).
    #
    # --- English ---
    # We set a moderate target length for the largest arrow (visual only).
    arrow_len = 0.20

    if N <= 400:
        idx_draw = np.arange(N, dtype=np.int64)
    else:
        idx_draw = np.asarray(sorted(set(int(j) for j in true_neighbors_including_self)), dtype=np.int64)
        if idx_draw.size == 0:
            idx_draw = np.asarray([int(focus_index)], dtype=np.int64)

    axs_pipe[2].scatter(particles.x, particles.y, s=18, c="lightgray", zorder=1)
    axs_pipe[2].quiver(
        particles.x[idx_draw],
        particles.y[idx_draw],
        ax_plot[idx_draw] * float(arrow_len),
        ay_plot[idx_draw] * float(arrow_len),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:blue",
        alpha=0.9,
        width=0.0045,
        headwidth=5.5,
        headlength=7.5,
        headaxislength=6.5,
        zorder=2,
    )
    axs_pipe[2].quiver(
        [float(particles.x[int(focus_index)])],
        [float(particles.y[int(focus_index)])],
        [float(ax_plot[int(focus_index)]) * float(arrow_len)],
        [float(ay_plot[int(focus_index)]) * float(arrow_len)],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:purple",
        alpha=1.0,
        width=0.010,
        headwidth=7.0,
        headlength=9.0,
        headaxislength=8.0,
        zorder=3,
    )
    axs_pipe[2].scatter([float(particles.x[int(focus_index)])], [float(particles.y[int(focus_index)])], s=150, c="tab:red", zorder=5)
    axs_pipe[2].add_patch(
        Circle((float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])), radius=float(h), fill=False, color="tab:red", linewidth=1.6, alpha=0.7, zorder=4)
    )
    axs_pipe[2].set_title("pressure acceleration (arrows)", pad=16)
    axs_pipe[2].set_xlabel("x")
    axs_pipe[2].set_ylabel("y")
    axs_pipe[2].grid(True, alpha=0.25)
    accel_box_text = axs_pipe[2].text(
        0.02,
        0.98,
        "max(|a|) = {amax}\n"
        "arrows drawn = {n}\n\n"
        "rho_focus = {rf}\n"
        "drho_focus = {df}\n"
        "p_focus = {pf}\n"
        "|a_focus| = {af}\n\n"
        "DE:\n"
        "|a| = Betrag der Beschleunigung\n"
        "Pfeillänge ist skaliert/normiert (nicht absolut)\n"
        "max(|a|) wird oben numerisch angegeben\n\n"
        "EN:\n"
        "|a| = magnitude of acceleration\n"
        "arrow length is scaled/normalized (not absolute)\n"
        "max(|a|) is shown numerically above".format(
            amax=f"{float(a_mag_max):.12g}",
            n=int(idx_draw.size),
            rf=f"{float(rho_focus):.12g}",
            df=f"{float(drho_focus):.12g}",
            pf=f"{float(p_focus):.12g}",
            af=f"{float(a_focus):.12g}",
        ),
        transform=axs_pipe[2].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        zorder=10,
    )

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # TEIL C + D: Rand-Disclaimer + didaktische Klarstellung (gut sichtbar)
    # -------------------------------------------------------------------------
    # Warum sieht es am Rand "komisch" aus?
    # - Dieses Demo nutzt KEINE Randbedingungen (keine Wände, keine Ghost-Particles).
    # - Am Rand fehlen Nachbarn → Summen/Gradienten werden einseitig → Randartefakte sind erwartbar.
    #
    # Was zeigen die Pfeile?
    # - Pfeile zeigen die resultierende Beschleunigung a an jedem Partikel.
    # - Sie zeigen NICHT einzelne Nachbarbeiträge.
    # - Viele Nachbarn zusammen können zu scheinbar kontra-intuitiven Richtungen führen.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # PART C + D: boundary disclaimer + didactic clarification (clearly visible)
    # -------------------------------------------------------------------------
    # Why can it look "weird" near boundaries?
    # - This demo uses NO boundary conditions (no walls, no ghost particles).
    # - Near boundaries neighbors are missing → sums/gradients become one-sided → boundary artifacts are expected.
    #
    # What do the arrows show?
    # - Arrows show the resulting acceleration a at each particle.
    # - They do NOT represent individual neighbor force contributions.
    # - Many neighbors together can create counter-intuitive directions.
    fig_pipe.text(
        0.015,
        0.985,
        "DE:\n"
        "Hinweis:\n"
        "Dieses Demo verwendet keine Randbedingungen (keine Wände, keine Ghost-Particles).\n"
        "Die gezeigten Pfeile an Rändern und Ecken entstehen durch fehlende Nachbarn\n"
        "und sind erwartete SPH-Randartefakte.\n"
        "Randbedingungen werden in einem späteren Schritt ergänzt.\n\n"
        "Pfeile zeigen die resultierende Beschleunigung.\n"
        "Sie zeigen nicht einzelne Nachbarbeiträge.\n"
        "Mehrere Nachbarn können zu scheinbar widersprüchlichen Richtungen führen.\n\n"
        "EN:\n"
        "Note:\n"
        "This demo uses no boundary conditions (no walls, no ghost particles).\n"
        "Arrows near boundaries and corners result from missing neighbors\n"
        "and are expected SPH boundary artifacts.\n"
        "Boundary conditions will be added in a later step.\n\n"
        "Arrows show the resulting acceleration.\n"
        "They do not represent individual neighbor forces.\n"
        "Multiple neighbors can create counter-intuitive directions.",
        ha="left",
        va="top",
        fontsize=9.0,
        wrap=True,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
        zorder=100,
    )

    fig_pipe.suptitle(
        "DE: Pipeline: rho → drho = rho - rho0_ref → p = k*drho → a(p, rho, ∇W)\n"
        "DE: Negative drho → negativer Druck (Unterdruck im Modell). Positive drho → positiver Druck.\n\n"
        "EN: Pipeline: rho → drho = rho - rho0_ref → p = k*drho → a(p, rho, ∇W)\n"
        "EN: Negative drho produces negative pressure (in this model). Positive drho produces positive pressure.",
        fontsize=12,
        x=0.67,
        y=0.985,
    )
    # --- Deutsch ---
    # Wir reservieren oben Platz für die Disclaimer-Textbox (sonst kann sie sich mit dem Titel überlappen).
    #
    # --- English ---
    # We reserve space at the top for the disclaimer textbox (otherwise it can overlap with the title).
    fig_pipe.tight_layout(rect=[0.0, 0.0, 1.0, 0.78])

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Textbox für "pressure acceleration" außerhalb des Diagramms platzieren
    # -------------------------------------------------------------------------
    # Ziel:
    # - Die Box soll NICHT über den Pfeilen oder dem Fokus liegen.
    # - Sie soll oberhalb der Achse stehen, direkt über der Überschrift/Title des Panels.
    #
    # Technischer Trick:
    # - Wir erzeugen die Textbox zuerst "normal" (wie die anderen Panels).
    # - Danach, NACH tight_layout (wenn die Achsenposition final ist), verschieben wir sie
    #   in Figure-Koordinaten knapp oberhalb der Achse.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Place the "pressure acceleration" textbox outside the diagram
    # -------------------------------------------------------------------------
    # Goal:
    # - The box should NOT cover arrows or the focus particle.
    # - It should sit above the axis, directly above the panel title.
    #
    # Technical trick:
    # - We first create the textbox "normally" (like other panels).
    # - Then, AFTER tight_layout (when the axis position is final), we move it into
    #   figure coordinates slightly above the axis.
    ax2_pos = axs_pipe[2].get_position()
    accel_box_text.set_transform(fig_pipe.transFigure)
    accel_box_text.set_position((float(ax2_pos.x0) + 0.5 * float(ax2_pos.width), float(ax2_pos.y1) + 0.045))
    accel_box_text.set_ha("center")
    accel_box_text.set_va("bottom")
    accel_box_text.set_clip_on(False)
    accel_box_text.set_zorder(200)

    # --- Deutsch ---
    # =========================================================================
    # Mini-Simulation (nur Vorher/Nachher, keine Echtzeit-Animation)
    # =========================================================================
    # Ziel:
    # - Wir wollen den kompletten "Loop" einmal didaktisch sichtbar machen:
    #
    #     rho → p → a → Integration
    #
    # - Danach zeigen wir nur:
    #   (a) Startpositionen (hell) und
    #   (b) Endpositionen (kräftig)
    #   Optional: Trajektorie des Fokus-Partikels.
    #
    # WICHTIGER DISCLAIMER (für Anfänger):
    # - Keine Randbedingungen (keine Wände, keine Ghost-Particles) → Randartefakte sind normal.
    # - Keine Viskosität → es fehlt “Dämpfung”, daher kann Bewegung “hart” wirken.
    # - Nur Demonstration: keine garantierte Stabilität oder physikalische Vollständigkeit.
    #
    # --- English ---
    # =========================================================================
    # Mini simulation (before/after only, no real-time animation)
    # =========================================================================
    # Goal:
    # - We want to show the full "loop" once in a didactic way:
    #
    #     rho → p → a → integration
    #
    # - After that we show only:
    #   (a) start positions (faint) and
    #   (b) end positions (strong)
    #   Optional: trajectory of the focus particle.
    #
    # IMPORTANT DISCLAIMER (for beginners):
    # - No boundary conditions (no walls, no ghost particles) → boundary artifacts are normal.
    # - No viscosity → there is no “damping”, so motion can feel “harsh”.
    # - Demonstration only: no guarantee of stability or full physical completeness.
    N = particles.n
    x_start = particles.x.copy()
    y_start = particles.y.copy()

    # --- Deutsch ---
    # Wir simulieren auf einer eigenen Kopie, damit die bisherigen Plots unverändert bleiben.
    #
    # --- English ---
    # We simulate on our own copy so the existing plots remain unchanged.
    sim_particles = ParticleSet2D(
        x=x_start.copy(),
        y=y_start.copy(),
        vx=particles.vx.copy(),
        vy=particles.vy.copy(),
        rho=np.zeros(N, dtype=np.float64),
        p=np.zeros(N, dtype=np.float64),
        m=particles.m.copy(),
    )

    focus_x_traj: list[float] = [float(sim_particles.x[int(focus_index)])]
    focus_y_traj: list[float] = [float(sim_particles.y[int(focus_index)])]
    traj_stride = max(1, int(int(n_steps) // 200))

    # --- Deutsch ---
    # Für “sichtbaren” Druck wählen wir einen Referenzwert rho0_ref_sim aus dem Startzustand:
    # - Wenn das Gitter sehr symmetrisch ist, kann rho fast konstant sein.
    # - Dann ist drho = rho - rho0_ref_sim fast 0 → p ~ 0 → a ~ 0 → keine Bewegung.
    # - Für die Demo nehmen wir rho0_ref_sim = mean(rho_init).
    #
    # WICHTIG:
    # - Wir nutzen im Simulationsloop absichtlich **signed pressure** (ohne clamp),
    #   damit man auch Unterdruck/Überdruck sehen kann.
    #
    # --- English ---
    # For “visible” pressure we choose a reference rho0_ref_sim from the initial state:
    # - If the grid is very symmetric, rho can be almost constant.
    # - Then drho = rho - rho0_ref_sim is almost 0 → p ~ 0 → a ~ 0 → no motion.
    # - For the demo we use rho0_ref_sim = mean(rho_init).
    #
    # IMPORTANT:
    # - In the simulation loop we intentionally use **signed pressure** (no clamp),
    #   so you can see negative/positive pressure.
    if compute_density_grid is not None:
        rho_init = compute_density_grid(particles=sim_particles, h=float(h), cell_size=float(h))
    else:
        rho_init = compute_density_naive(particles=sim_particles, h=float(h))
    rho0_ref_sim = float(np.mean(rho_init))

    # --- Deutsch ---
    # Kleine, kontrollierte Asymmetrie (NUR fürs Demo):
    # - Wir verschieben genau EIN Partikel minimal.
    # - Das bricht perfekte Symmetrie, damit sich Kräfte nicht komplett ausgleichen.
    #
    # Hinweis:
    # - Wir verändern die echten Partikel nicht, nur `sim_particles` (Demo-Kopie).
    #
    # --- English ---
    # Small, controlled asymmetry (DEMO ONLY):
    # - We shift exactly ONE particle by a tiny amount.
    # - This breaks perfect symmetry so forces do not cancel out completely.
    #
    # Note:
    # - We do not modify the real particles, only `sim_particles` (demo copy).
    sim_particles.x[0] += 0.02 * float(dx)

    ax_last = np.zeros(N, dtype=np.float64)
    ay_last = np.zeros(N, dtype=np.float64)
    p_last = np.zeros(N, dtype=np.float64)
    a_mag_max_first = 0.0

    for _step in range(int(n_steps)):
        if compute_density_grid is not None:
            rho_step = compute_density_grid(particles=sim_particles, h=float(h), cell_size=float(h))
        else:
            rho_step = compute_density_naive(particles=sim_particles, h=float(h))

        # --- Deutsch ---
        # drho und p konsistent aus demselben Referenzwert rho0_ref_sim:
        # - drho = rho - rho0_ref_sim
        # - p = k * drho   (signed, ohne clamp)
        #
        # Warum ohne clamp?
        # - Für die Demo wollen wir Unterdruck (negativ) und Überdruck (positiv) sehen.
        #
        # --- English ---
        # drho and p consistent from the same reference value rho0_ref_sim:
        # - drho = rho - rho0_ref_sim
        # - p = k * drho   (signed, no clamp)
        #
        # Why no clamp?
        # - For the demo we want to see negative pressure (under-pressure) and positive pressure.
        drho_step = np.asarray(rho_step, dtype=np.float64) - float(rho0_ref_sim)
        p_step = compute_pressure_eos(rho=rho_step, rho0=float(rho0_ref_sim), k=float(k), clamp_negative=False)

        ax_step, ay_step = compute_pressure_acceleration(
            particles=sim_particles,
            rho=rho_step,
            p=p_step,
            h=float(h),
            cell_size=float(h),
        )

        if int(_step) == 0:
            a_mag = np.sqrt(np.asarray(ax_step, dtype=np.float64) ** 2 + np.asarray(ay_step, dtype=np.float64) ** 2)
            a_mag_max_first = float(np.max(a_mag)) if a_mag.size > 0 else 0.0

        # --- Deutsch ---
        # Integration: v += a*dt, dann x += v*dt (Semi-Implicit Euler).
        #
        # --- English ---
        # Integration: v += a*dt, then x += v*dt (semi-implicit Euler).
        step_semi_implicit_euler(particles=sim_particles, ax=ax_step, ay=ay_step, dt=float(dt))

        # --- Deutsch ---
        # Optionale, sehr einfache Dämpfung (nur Demo):
        # - Wir reduzieren v leicht pro Schritt.
        # - Das ist "Demo stabilization; not physical viscosity".
        #
        # --- English ---
        # Optional, very simple damping (demo only):
        # - We reduce v slightly per step.
        # - This is "demo stabilization; not physical viscosity".
        if float(damping) > 0.0:
            sim_particles.vx *= (1.0 - float(damping))
            sim_particles.vy *= (1.0 - float(damping))

        # --- Deutsch ---
        # Trajektorie nur fürs Fokus-Partikel, und maximal ~200 Punkte:
        # - Wir speichern nur jeden `traj_stride`-ten Schritt.
        #
        # --- English ---
        # Trajectory only for the focus particle, and max ~200 points:
        # - We store only every `traj_stride`-th step.
        if bool(show_focus_trajectory) and (int(_step) % int(traj_stride) == 0):
            focus_x_traj.append(float(sim_particles.x[int(focus_index)]))
            focus_y_traj.append(float(sim_particles.y[int(focus_index)]))

        ax_last = np.asarray(ax_step, dtype=np.float64)
        ay_last = np.asarray(ay_step, dtype=np.float64)
        p_last = np.asarray(p_step, dtype=np.float64)

    x_end = sim_particles.x.copy()
    y_end = sim_particles.y.copy()

    # --- Deutsch ---
    # Debug-Metriken (pro Run):
    # - max(|a|) im ersten Schritt: wenn das ~0 ist, ist keine Bewegung möglich.
    # - max(|v|) am Ende
    # - max(|dx|) am Ende (maximale Positionsänderung)
    #
    # --- English ---
    # Debug metrics (per run):
    # - max(|a|) in the first step: if that is ~0, no motion is possible.
    # - max(|v|) at the end
    # - max(|dx|) at the end (maximum position change)
    v_mag_end = np.sqrt(sim_particles.vx * sim_particles.vx + sim_particles.vy * sim_particles.vy)
    v_mag_max_end = float(np.max(v_mag_end)) if v_mag_end.size > 0 else 0.0
    disp = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
    disp_max = float(np.max(disp)) if disp.size > 0 else 0.0

    # --- Deutsch ---
    # Neue finale Figure: Start vs End (keine Animation).
    #
    # --- English ---
    # New final figure: start vs end (no animation).
    fig_sim, ax_sim = plt.subplots(1, 1, figsize=(8.0, 6.5))
    ax_sim.set_title("mini simulation: start vs end (rho → p → a → integration)")
    ax_sim.set_xlabel("x")
    ax_sim.set_ylabel("y")
    ax_sim.set_aspect("equal", adjustable="box")
    ax_sim.grid(True, alpha=0.25)

    # --- Deutsch ---
    # Startpositionen (klar getrennt von "Ende"):
    # - Wir zeichnen Start als graue, hohle Marker.
    # - Dadurch sieht man sofort: "Hohl = Start".
    #
    # --- English ---
    # Start positions (clearly separated from "end"):
    # - We draw start as gray, hollow markers.
    # - This makes it immediately clear: "hollow = start".
    start_sc = ax_sim.scatter(
        x_start,
        y_start,
        s=26,
        facecolors="none",
        edgecolors="gray",
        linewidths=0.9,
        alpha=0.35,
        label="start",
        zorder=1,
    )

    # --- Deutsch ---
    # Optional: Bewegungsvektoren (nur wenn es nicht überlädt).
    # - Wenn N <= 200: wir zeichnen für alle Partikel eine dünne Linie von Start → Ende.
    # - Wenn N groß ist: nur Fokus + (falls vorhanden) seine Nachbarn.
    #
    # --- English ---
    # Optional: motion vectors (only if it does not clutter).
    # - If N <= 200: draw a thin line for all particles from start → end.
    # - If N is large: only focus + (if available) its neighbors.
    idx_motion: np.ndarray
    if int(N) <= 200:
        idx_motion = np.arange(int(N), dtype=np.int64)
    else:
        if "true_neighbors_including_self" in locals():
            idx_motion = np.asarray(sorted(set(int(j) for j in true_neighbors_including_self)), dtype=np.int64)
            if idx_motion.size == 0:
                idx_motion = np.asarray([int(focus_index)], dtype=np.int64)
        else:
            idx_motion = np.asarray([int(focus_index)], dtype=np.int64)

    for i in idx_motion:
        ii = int(i)
        ax_sim.plot(
            [float(x_start[ii]), float(x_end[ii])],
            [float(y_start[ii]), float(y_end[ii])],
            color="gray",
            alpha=0.15,
            linewidth=0.8,
            zorder=2,
        )

    # --- Deutsch ---
    # Endpositionen: kräftiger (hier farbig nach p, damit man “Druckfeld” sieht).
    #
    # --- English ---
    # End positions: stronger (colored by p here so you can see a “pressure field”).
    # --- Deutsch ---
    # Endpositionen: kräftiger (farbig nach p, damit man die Druckverteilung erkennt).
    # Wir nutzen eine Diverging-Farbskala um 0, weil p hier signed ist.
    #
    # --- English ---
    # End positions: stronger (colored by p so you can see the pressure field).
    # We use a diverging color scale around 0 because p is signed here.
    p_min_end = float(np.min(p_last)) if p_last.size > 0 else 0.0
    p_max_end = float(np.max(p_last)) if p_last.size > 0 else 0.0
    if p_min_end < 0.0 < p_max_end:
        norm_p_end = TwoSlopeNorm(vmin=p_min_end, vcenter=0.0, vmax=p_max_end)
    else:
        norm_p_end = None

    if norm_p_end is not None:
        sc_end = ax_sim.scatter(
            x_end,
            y_end,
            s=30,
            c=p_last,
            cmap="coolwarm",
            norm=norm_p_end,
            alpha=0.85,
            label="end (colored by pressure)",
            zorder=4,
        )
    else:
        sc_end = ax_sim.scatter(
            x_end,
            y_end,
            s=30,
            c=p_last,
            cmap="coolwarm",
            alpha=0.85,
            label="end (colored by pressure)",
            zorder=4,
        )
    cbar_end = fig_sim.colorbar(sc_end, ax=ax_sim)
    cbar_end.set_label("pressure p (demo)")

    # --- Deutsch ---
    # Fokus-Partikel: Bewegung explizit sichtbar machen
    # - Start: kleines schwarzes "x"
    # - Ende: großer roter Stern
    # - Pfeil: zeigt integrierte Bewegung Start → Ende über n_steps
    #
    # --- English ---
    # Focus particle: make motion explicitly visible
    # - Start: small black "x"
    # - End: large red star
    # - Arrow: shows integrated motion start → end over n_steps
    x_focus_start = float(x_start[int(focus_index)])
    y_focus_start = float(y_start[int(focus_index)])
    x_focus_end = float(x_end[int(focus_index)])
    y_focus_end = float(y_end[int(focus_index)])
    dx_focus = float(np.sqrt((x_focus_end - x_focus_start) ** 2 + (y_focus_end - y_focus_start) ** 2))

    ax_sim.scatter(
        [x_focus_start],
        [y_focus_start],
        s=70,
        c="black",
        marker="x",
        linewidths=2.0,
        alpha=0.9,
        label="_nolegend_",
        zorder=5,
    )
    ax_sim.scatter(
        [x_focus_end],
        [y_focus_end],
        s=160,
        c="tab:red",
        marker="*",
        alpha=1.0,
        label="_nolegend_",
        zorder=7,
    )
    ax_sim.annotate(
        "",
        xy=(x_focus_end, y_focus_end),
        xytext=(x_focus_start, y_focus_start),
        arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 2.0, "alpha": 0.9},
        zorder=6,
    )

    # --- Deutsch ---
    # Optional: Trajektorie des Fokus-Partikels als Linie (falls aktiviert).
    #
    # --- English ---
    # Optional: focus particle trajectory as a line (if enabled).
    if bool(show_focus_trajectory):
        ax_sim.plot(focus_x_traj, focus_y_traj, color="tab:red", linewidth=1.6, alpha=0.75, label="_nolegend_", zorder=3)

    # --- Deutsch ---
    # Disclaimer-Textbox (kurz, klar).
    #
    # --- English ---
    # Disclaimer textbox (short, clear).
    ax_sim.text(
        0.02,
        0.98,
        "DE:\n"
        f"dt = {float(dt):.6g}\n"
        f"n_steps = {int(n_steps)}\n"
        f"k = {float(k):.6g}\n"
        f"damping = {float(damping):.6g}\n"
        f"max(|dx|) Ende = {disp_max:.6g}\n\n"
        "Start = hohl grau, Ende = farbig (Druck)\n\n"
        "Hinweis:\n"
        "- keine Randbedingungen\n"
        "- keine Viskosität\n"
        "- keine Stabilisierung (außer demo damping)\n"
        "- Bewegung dient nur der Demonstration der Pipeline\n\n"
        "EN:\n"
        f"dt = {float(dt):.6g}\n"
        f"n_steps = {int(n_steps)}\n"
        f"k = {float(k):.6g}\n"
        f"damping = {float(damping):.6g}\n"
        f"max(|dx|) end = {disp_max:.6g}\n\n"
        "Start = hollow gray, End = colored (pressure)\n\n"
        "Note:\n"
        "- no boundary conditions\n"
        "- no viscosity\n"
        "- no stabilization (except demo damping)\n"
        "- motion is only to demonstrate the pipeline",
        transform=ax_sim.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
        zorder=10,
    )

    # --- Deutsch ---
    # Fokus-Textbox: dx_focus macht die integrierte Bewegung über n_steps direkt als Zahl sichtbar.
    #
    # --- English ---
    # Focus textbox: dx_focus makes the integrated motion over n_steps visible as a number.
    ax_sim.text(
        0.98,
        0.02,
        "DE:\n"
        f"dx_focus = {dx_focus:.6g}\n"
        "Das zeigt die integrierte Bewegung über n_steps.\n\n"
        "EN:\n"
        f"dx_focus = {dx_focus:.6g}\n"
        "This shows the integrated motion over n_steps.",
        transform=ax_sim.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
        zorder=10,
    )

    # --- Deutsch ---
    # Debug-Textbox: erklärt, warum ggf. keine Bewegung sichtbar ist.
    #
    # --- English ---
    # Debug textbox: explains why motion may not be visible.
    ax_sim.text(
        0.02,
        0.02,
        "DE:\n"
        f"max(|a|) Schritt 0 = {a_mag_max_first:.6g}\n"
        f"max(|v|) Ende     = {v_mag_max_end:.6g}\n"
        f"max(|dx|) Ende    = {disp_max:.6g}\n"
        "Wenn |a|≈0, dann kann es keine Bewegung geben.\n\n"
        "EN:\n"
        f"max(|a|) step 0   = {a_mag_max_first:.6g}\n"
        f"max(|v|) end      = {v_mag_max_end:.6g}\n"
        f"max(|dx|) end     = {disp_max:.6g}\n"
        "If |a|≈0, there cannot be any motion.",
        transform=ax_sim.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
        zorder=10,
    )

    # --- Deutsch ---
    # Kleine Legende, die die Start/Ende-Kodierung erklärt (DE/EN blockweise).
    #
    # --- English ---
    # Small legend that explains the start/end encoding (DE/EN block-wise).
    leg = ax_sim.legend(
        handles=[start_sc, sc_end],
        loc="lower left",
        fontsize=9,
        framealpha=0.9,
        title="DE:\nstart = hohl grau\nende = farbig (Druck)\n\nEN:\nstart = hollow gray\nend = colored (pressure)",
        title_fontsize=9,
    )

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Alle Diagramme "verknüpfen": gleiche Achsen + gemeinsame Zoom/Pan-Ansicht
    # -------------------------------------------------------------------------
    # Ziel:
    # - Alle Plots, die Partikel in x/y zeigen, sollen:
    #   (1) dieselben Achsen-Limits haben (direkt vergleichbar)
    #   (2) beim Zoomen/Pannen synchron bleiben (miteinander verknüpft)
    #
    # Umsetzung:
    # - Wir berechnen gemeinsame xlim/ylim aus denselben Daten (particles.x/y, Fokus, h, dx).
    # - Wir setzen diese Limits auf alle Partikel-Achsen.
    # - Wir verbinden die Achsen über Callbacks (funktioniert auch über mehrere Figures).
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # "Link" all diagrams: same axes + shared zoom/pan view
    # -------------------------------------------------------------------------
    # Goal:
    # - All plots that show particles in x/y should:
    #   (1) use the same axis limits (directly comparable)
    #   (2) stay synchronized when zooming/panning (linked together)
    #
    # Implementation:
    # - Compute shared xlim/ylim from the same data (particles.x/y, focus, h, dx).
    # - Apply those limits to all particle axes.
    # - Link axes via callbacks (works across multiple figures).
    xlim_shared, ylim_shared = _compute_shared_xy_limits(
        particles.x,
        particles.y,
        focus_index=int(focus_index),
        h=float(h),
        dx=float(dx),
    )

    axes_to_link = [
        # Haupt-Plot: Partikel + Dichte
        axs[0],
        axs[2],
        # Vergleich: naive/grid/diff
        axs_cmp[0],
        axs_cmp[1],
        axs_cmp[2],
        # Demo: rho/p/drho als x/y Scatter
        axs_demo[0],
        axs_demo[1],
        axs_demo[2],
        # Pressure-Figure: Druckkarte (x/y), EOS-Scatter NICHT (rho->p)
        axs_p[0],
        # Pipeline-Figure: drho / p / Pfeile
        axs_pipe[0],
        axs_pipe[1],
        axs_pipe[2],
        # Mini-Simulation: Start vs End (x/y)
        ax_sim,
    ]

    for ax_link in axes_to_link:
        _apply_shared_xy_limits(ax_link, xlim=xlim_shared, ylim=ylim_shared)

    _link_xy_axes(axes_to_link)


    # --- Deutsch ---
    # Anzeigen (kein File-I/O, keine Prints)
    #
    # --- English ---
    # Display (no file I/O, no prints)
    plt.show()


def run_learning_viz_animation(
    L: float = 1.0,
    dx: float = 0.1,
    h: float | None = None,
    k: float = 25.0,
    dt: float = 0.001,
    steps_per_frame: int = 5,
    n_frames: int = 400,
    damping: float = 0.02,
    focus_index: int = 0,
) -> None:
    """
    DE:
    Live-Demo-Animation: Partikel + Druckbeschleunigungs-Pfeile werden pro Frame aktualisiert.

    Ziel (Didaktik):
    --------------
    Anfänger sollen sofort sehen, was passiert:
    - Partikel bewegen sich, weil wir pro Zeitschritt integrieren.
    - Pfeile zeigen die Beschleunigungsrichtung (aus Druckkräften).
    - Die Pfeillänge ist für die Darstellung skaliert, nicht absolut.

    Pipeline pro Simulationsschritt:
    -------------------------------
    1) rho berechnen (grid preferred, fallback naive)
    2) rho0_ref = mean(rho)  (Demo-Referenz)
    3) drho = rho - rho0_ref
    4) p = k * drho (signed, clamp_negative=False)
    5) (ax, ay) aus Druckkräften berechnen
    6) integrieren (Semi-Implicit Euler)
    7) Demo-Dämpfung auf v anwenden (Stabilisierung nur fürs Demo)

    Wichtige Disclaimer:
    --------------------
    - Keine Randbedingungen (keine Wände, keine Ghost-Particles) → Randartefakte sind normal.
    - Keine echte Viskosität → wir fügen keine physikalische Dämpfungskraft hinzu.
    - Die optionale Dämpfung hier ist NUR “Demo stabilization; not physical viscosity”.

    EN:
    Live demo animation: particles + pressure-acceleration arrows update each frame.

    Goal (teaching):
    Beginners should immediately see what is happening:
    - Particles move because we integrate each time step.
    - Arrows show acceleration direction (from pressure forces).
    - Arrow length is scaled for visualization, not absolute.

    Pipeline per simulation step:
    1) compute rho (grid preferred, fallback naive)
    2) rho0_ref = mean(rho)  (demo reference)
    3) drho = rho - rho0_ref
    4) p = k * drho (signed, clamp_negative=False)
    5) compute (ax, ay) from pressure forces
    6) integrate (semi-implicit Euler)
    7) apply demo damping to v (stabilization for demo only)

    Important disclaimers:
    - No boundary conditions (no walls, no ghost particles) → boundary artifacts are normal.
    - No real viscosity → we do not add a physical damping force.
    - The optional damping here is ONLY “demo stabilization; not physical viscosity”.
    """

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Eingaben validieren (frühe, klare Fehler)
    # -------------------------------------------------------------------------
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Validate inputs (early, clear errors)
    # -------------------------------------------------------------------------
    _validate_positive("L", float(L))
    _validate_positive("dx", float(dx))
    _validate_positive("k", float(k))
    _validate_positive("dt", float(dt))
    if int(steps_per_frame) <= 0:
        raise ValueError("steps_per_frame muss > 0 sein.")
    if int(n_frames) <= 0:
        raise ValueError("n_frames muss > 0 sein.")
    if float(damping) < 0.0 or float(damping) >= 1.0:
        raise ValueError("damping muss in [0, 1) liegen.")

    # --- Deutsch ---
    # Wenn h nicht gegeben ist: didaktischer Default h = 1.5*dx.
    #
    # --- English ---
    # If h is not provided: didactic default h = 1.5*dx.
    if h is None:
        h_used = 1.5 * float(dx)
    else:
        h_used = float(h)
    _validate_positive("h", float(h_used))

    # --- Deutsch ---
    # Partikel erzeugen (Demo-Parameter für rho0 und Masse):
    # - rho0=1000.0 (Wasser als typischer Referenzwert)
    #
    # Demo-Massenabschätzung (2D):
    # - In 2D ist eine einfache, plausible Masse pro Partikel:
    #
    #     m = rho0 * dx * dx
    #
    #   Interpretation:
    #   - Jede Zelle im Gitter hat Fläche dx*dx.
    #   - rho0 ist eine Referenzdichte.
    #   - m ist dann eine grobe "Masse pro Partikel" für sinnvollere Skalen.
    #
    # WICHTIG:
    # - Das ist nur Demo-Skalierung.
    # - Es ist keine perfekte physikalische Kalibrierung.
    #
    # --- English ---
    # Create particles (demo parameters for rho0 and mass):
    # - rho0=1000.0 (water as a typical reference value)
    #
    # Demo mass estimate (2D):
    # - In 2D, a simple, plausible mass per particle is:
    #
    #     m = rho0 * dx * dx
    #
    # Interpretation:
    # - Each grid cell has area dx*dx.
    # - rho0 is a reference density.
    # - m is then a rough "mass per particle" for more reasonable scales.
    #
    # IMPORTANT:
    # - This is demo scaling only.
    # - It is not a perfect physical calibration.
    rho0_demo = 1000.0
    m_demo = float(rho0_demo) * float(dx) * float(dx)
    particles = initialize_particles_cube(L=float(L), dx=float(dx), rho0=float(rho0_demo), mass_per_particle=float(m_demo))
    N = particles.n

    # --- Deutsch ---
    # Fokus-Index robust machen (anfängerfreundlich).
    #
    # --- English ---
    # Make focus index robust (beginner-friendly).
    if int(focus_index) < 0 or int(focus_index) >= int(N):
        focus_index = 0

    # --- Deutsch ---
    # Startpositionen merken (für max(|dx|) seit Start).
    #
    # --- English ---
    # Store start positions (for max(|dx|) since start).
    x0 = particles.x.copy()
    y0 = particles.y.copy()

    # --- Deutsch ---
    # Wir halten die "letzten" a-Werte für die Visualisierung.
    #
    # --- English ---
    # We keep the "latest" a values for visualization.
    ax_last = np.zeros(N, dtype=np.float64)
    ay_last = np.zeros(N, dtype=np.float64)

    # --- Deutsch ---
    # Figure/Axes (ein Diagramm):
    # - Scatter: Partikel
    # - Quiver: Beschleunigungsrichtung (skaliert)
    #
    # --- English ---
    # Figure/axes (single plot):
    # - Scatter: particles
    # - Quiver: acceleration direction (scaled)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SPH demo animation: particles + pressure acceleration (scaled arrows)")
    ax.grid(True, alpha=0.25)

    pad = 0.05 * float(L)
    ax.set_xlim(-pad, float(L) + pad)
    ax.set_ylim(-pad, float(L) + pad)

    # --- Deutsch ---
    # Scatter: wir erstellen ihn einmal und updaten nur die Offsets.
    #
    # --- English ---
    # Scatter: create once and only update offsets.
    scat = ax.scatter(particles.x, particles.y, s=24, c="tab:blue", alpha=0.85, zorder=2)

    # --- Deutsch ---
    # Fokus-Partikel markieren (extra Marker).
    #
    # --- English ---
    # Mark focus particle (extra marker).
    focus_scat = ax.scatter(
        [float(particles.x[int(focus_index)])],
        [float(particles.y[int(focus_index)])],
        s=140,
        c="tab:red",
        marker="*",
        zorder=5,
        label="focus",
    )

    # --- Deutsch ---
    # Quiver: initial (noch keine Beschleunigung, also 0).
    #
    # WICHTIG (Didaktik):
    # - Wir normalisieren Pfeile später pro Frame:
    #   ax_plot = ax/(max(|a|)+eps), damit Richtung klar sichtbar bleibt.
    #
    # --- English ---
    # Quiver: initial (no acceleration yet, so 0).
    #
    # IMPORTANT (teaching):
    # - We normalize arrows per frame:
    #   ax_plot = ax/(max(|a|)+eps), so direction stays clearly visible.
    arrow_len = 0.25 * float(dx)
    quiv = ax.quiver(
        particles.x,
        particles.y,
        np.zeros(N, dtype=np.float64),
        np.zeros(N, dtype=np.float64),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:purple",
        alpha=0.75,
        width=0.0035,
        zorder=3,
    )

    # --- Deutsch ---
    # Statische Disclaimer-Textbox (bleibt gleich).
    #
    # --- English ---
    # Static disclaimer textbox (does not change).
    disclaimer_text = ax.text(
        0.02,
        0.98,
        "DE:\n"
        "Demo-Animation: Druckkräfte aus EOS + Spiky-Gradient.\n"
        "Hinweis: Keine Randbedingungen (keine Wände/Ghost-Particles), keine echte Viskosität, nur Demo.\n"
        "Pfeile zeigen Beschleunigungsrichtung; Länge ist skaliert.\n\n"
        "EN:\n"
        "Demo animation: pressure forces from EOS + spiky gradient.\n"
        "Note: no boundary conditions (no walls/ghost particles), no real viscosity, demo only.\n"
        "Arrows show acceleration direction; length is scaled.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
        zorder=10,
    )

    # --- Deutsch ---
    # Debug-Textbox (wird pro Frame aktualisiert).
    #
    # --- English ---
    # Debug textbox (updated each frame).
    debug_text = ax.text(
        0.02,
        0.02,
        "DE:\nframe = 0\nmax(|a|)=0\nmax(|v|)=0\nmax(|dx|)=0\n\nEN:\nframe = 0\nmax(|a|)=0\nmax(|v|)=0\nmax(|dx|)=0",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
        zorder=10,
    )

    def _one_sim_step() -> tuple[np.ndarray, np.ndarray]:
        """
        DE:
        Führe genau EINEN Simulationsschritt aus und gib (ax, ay) zurück.

        EN:
        Perform exactly ONE simulation step and return (ax, ay).
        """

        # --- Deutsch ---
        # 1) Dichte rho berechnen (Grid bevorzugt).
        #
        # --- English ---
        # 1) Compute density rho (prefer grid).
        if compute_density_grid is not None:
            rho = compute_density_grid(particles=particles, h=float(h_used), cell_size=float(h_used))
        else:
            rho = compute_density_naive(particles=particles, h=float(h_used))

        # --- Deutsch ---
        # 2) Demo-Referenz rho0_ref: Mittelwert.
        #
        # --- English ---
        # 2) Demo reference rho0_ref: mean.
        rho0_ref = float(np.mean(rho)) if rho.size > 0 else 0.0

        # --- Deutsch ---
        # 3) drho und 4) Druck p (signed, ohne clamp).
        #
        # --- English ---
        # 3) drho and 4) pressure p (signed, no clamp).
        drho = rho - rho0_ref
        p = compute_pressure_eos(rho=rho, rho0=float(rho0_ref), k=float(k), clamp_negative=False)

        # --- Deutsch ---
        # 5) Druckbeschleunigung berechnen.
        #
        # --- English ---
        # 5) Compute pressure acceleration.
        ax_step, ay_step = compute_pressure_acceleration(
            particles=particles,
            rho=rho,
            p=p,
            h=float(h_used),
            cell_size=float(h_used),
        )

        # --- Deutsch ---
        # 6) Integration.
        #
        # --- English ---
        # 6) Integration.
        step_semi_implicit_euler(particles=particles, ax=ax_step, ay=ay_step, dt=float(dt))

        # --- Deutsch ---
        # 7) Demo-Dämpfung (Stabilisierung nur fürs Demo, KEINE physikalische Viskosität).
        #
        # --- English ---
        # 7) Demo damping (demo stabilization only, NOT physical viscosity).
        if float(damping) > 0.0:
            particles.vx *= (1.0 - float(damping))
            particles.vy *= (1.0 - float(damping))

        return np.asarray(ax_step, dtype=np.float64), np.asarray(ay_step, dtype=np.float64)

    def update(frame: int):
        """
        DE:
        Matplotlib-Callback pro Frame.
        Wir führen `steps_per_frame` Simulationsschritte aus und aktualisieren dann die Artists.

        EN:
        Matplotlib callback per frame.
        We perform `steps_per_frame` simulation steps and then update the artists.
        """

        nonlocal ax_last, ay_last

        for _ in range(int(steps_per_frame)):
            ax_last, ay_last = _one_sim_step()

        # --- Deutsch ---
        # Scatter Offsets updaten.
        #
        # --- English ---
        # Update scatter offsets.
        scat.set_offsets(np.column_stack([particles.x, particles.y]))
        focus_scat.set_offsets(np.array([[float(particles.x[int(focus_index)]), float(particles.y[int(focus_index)])]]))

        # --- Deutsch ---
        # --- Deutsch ---
        # ---------------------------------------------------------------------
        # Soft-Scaling für Pfeile: Richtung + "etwas" Stärke, aber ruhig lesbar
        # ---------------------------------------------------------------------
        # Idee:
        # - Die Richtung ist physikalisch: sie ist die Beschleunigungsrichtung.
        # - Die Länge ist für Lesbarkeit skaliert, enthält aber jetzt auch Information über die Stärke.
        #
        # Umsetzung:
        # 1) a_mag = |a|
        # 2) a_unit = a / (|a| + eps)   → Einheitsvektor (nur Richtung)
        # 3) a_rel = sqrt(|a| / max(|a|))   → 0..1, aber "weicher"
        #
        # Warum sqrt?
        # - Sehr große Unterschiede (Ausreißer) werden abgemildert.
        # - Das reduziert “Pfeil-Chaos”, bleibt aber anschaulich.
        #
        # --- English ---
        # ---------------------------------------------------------------------
        # Soft scaling for arrows: direction + "some" strength, but calmly readable
        # ---------------------------------------------------------------------
        # Idea:
        # - Direction is physical: it is the acceleration direction.
        # - Length is scaled for readability, but now also carries some information about magnitude.
        #
        # Implementation:
        # 1) a_mag = |a|
        # 2) a_unit = a / (|a| + eps)       → unit vector (direction only)
        # 3) a_rel = sqrt(|a| / max(|a|))   → 0..1, but "softer"
        #
        # Why sqrt?
        # - Extreme differences (outliers) are damped.
        # - This reduces “arrow chaos” while staying intuitive.
        a_mag = np.sqrt(ax_last * ax_last + ay_last * ay_last)
        a_mag_max = float(np.max(a_mag)) if a_mag.size > 0 else 0.0
        a_unit_x = ax_last / (a_mag + 1e-12)
        a_unit_y = ay_last / (a_mag + 1e-12)
        a_rel = np.sqrt(a_mag / (float(a_mag_max) + 1e-12))

        quiv.set_offsets(np.column_stack([particles.x, particles.y]))
        quiv.set_UVC(a_unit_x * float(a_rel) * float(arrow_len), a_unit_y * float(a_rel) * float(arrow_len))

        # --- Deutsch ---
        # Debug-Metriken:
        # - max(|a|): aus den aktuellen Beschleunigungen
        # - max(|v|): aus den aktuellen Geschwindigkeiten
        # - max(|dx|): maximale Verschiebung seit Start
        #
        # --- English ---
        # Debug metrics:
        # - max(|a|): from current accelerations
        # - max(|v|): from current velocities
        # - max(|dx|): max displacement since start
        v_mag = np.sqrt(particles.vx * particles.vx + particles.vy * particles.vy)
        v_mag_max = float(np.max(v_mag)) if v_mag.size > 0 else 0.0
        disp = np.sqrt((particles.x - x0) ** 2 + (particles.y - y0) ** 2)
        disp_max = float(np.max(disp)) if disp.size > 0 else 0.0

        debug_text.set_text(
            "DE:\n"
            f"frame = {int(frame)}\n"
            f"max(|a|) = {a_mag_max:.6g}\n"
            f"max(|v|) = {v_mag_max:.6g}\n"
            f"max(|dx|) = {disp_max:.6g}\n\n"
            "EN:\n"
            f"frame = {int(frame)}\n"
            f"max(|a|) = {a_mag_max:.6g}\n"
            f"max(|v|) = {v_mag_max:.6g}\n"
            f"max(|dx|) = {disp_max:.6g}"
        )

        # blit=False → Rückgabewert ist optional; wir geben die wichtigsten Artists zurück.
        return scat, focus_scat, quiv, disclaimer_text, debug_text

    # --- Deutsch ---
    # Animation starten (looped).
    #
    # --- English ---
    # Start animation (looped).
    _anim = FuncAnimation(
        fig,
        update,
        frames=int(n_frames),
        interval=30,
        blit=False,
        repeat=True,
    )

    ax.legend(loc="lower left")
    plt.show()


def run_pressure_force_demo(
    L: float = 1.0,
    dx: float = 0.1,
    h: float | None = None,
    rho0: float = 1000.0,
    k: float = 500.0,
    dt: float = 0.002,
    num_steps: int = 30,
    focus_index: int = 0,
) -> None:
    """
    DE:
    Optionaler Zusatz-Demo-Modus: Druckkräfte (Pressure Forces) wirklich sichtbar machen.

    WICHTIG:
    - Diese Funktion ist komplett separat von `run_learning_viz()`.
    - Sie ändert keine bestehenden Plots/Funktionen.
    - Sie ist absichtlich sehr linear/verständlich geschrieben (Tutorial-Stil).

    Problem-Hintergrund:
    - Ein perfektes, symmetrisches Partikelgitter kann fast konstante Dichte/Druck erzeugen.
    - Dann ist der Druckgradient klein und die resultierende Kraft ~0.
    - Um "Force" zu demonstrieren, brauchen wir:
      (a) eine kleine Asymmetrie (Perturbation) und
      (b) mehrere Integrationsschritte, damit Bewegung sichtbar wird.

    Was wir hier machen:
    1) Wir erzeugen ein Partikelgitter.
    2) Wir arbeiten auf lokalen Kopien (x,y,vx,vy) → `particles` wird nicht dauerhaft verändert.
    3) Wir verschieben ein Partikel minimal (Perturbation), um einen Druckgradienten zu erzeugen.
    4) Wir iterieren `num_steps`:
       - rho berechnen (Grid bevorzugt, sonst naiv)
       - p_demo über EOS (geclamped)
       - Druckbeschleunigung (ax, ay)
       - explizites Euler: v += a*dt, x += v*dt
    5) Am Ende zeichnen wir eine ruhige 2-Panel-Figure:
       - Links: Before vs After (vorher grau, nachher farbig nach p_demo)
       - Rechts: Pfeile (quiver) für Druckbeschleunigung (letzter Schritt), visuell skaliert

    EN:
    Optional add-on demo mode: make pressure forces (pressure forces) truly visible.

    IMPORTANT:
    - This function is completely separate from `run_learning_viz()`.
    - It does not modify existing plots/functions.
    - It is intentionally written in a very linear/beginner-friendly style (tutorial style).

    Background problem:
    - A perfectly symmetric particle grid can produce nearly constant density/pressure.
    - Then the pressure gradient is tiny and the resulting force is ~0.
    - To demonstrate "force", we need:
      (a) a small asymmetry (perturbation) and
      (b) multiple integration steps so motion becomes visible.

    What we do here:
    1) Create a particle grid.
    2) Work on local copies (x,y,vx,vy) → `particles` is not permanently modified.
    3) Move one particle slightly (perturbation) to create a pressure gradient.
    4) Iterate `num_steps`:
       - compute rho (prefer grid, otherwise naive)
       - compute p_demo via EOS (clamped)
       - compute pressure acceleration (ax, ay)
       - explicit Euler: v += a*dt, x += v*dt
    5) Finally draw a calm 2-panel figure:
       - Left: before vs after (before gray, after colored by p_demo)
       - Right: arrows (quiver) for pressure acceleration (last step), visually scaled
    """

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Eingaben validieren (frühe, klare Fehler)
    # -------------------------------------------------------------------------
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Validate inputs (early, clear errors)
    # -------------------------------------------------------------------------
    _validate_positive("L", float(L))
    _validate_positive("dx", float(dx))
    _validate_positive("rho0", float(rho0))
    _validate_positive("k", float(k))
    _validate_positive("dt", float(dt))
    if int(num_steps) <= 0:
        raise ValueError("num_steps muss > 0 sein.")
    if float(dx) >= float(L):
        raise ValueError("dx muss < L sein, sonst entstehen zu wenige Partikel.")

    # --- Deutsch ---
    # Wenn h nicht gegeben ist, wählen wir einen didaktischen Default:
    # h = 1.5 * dx (typisch: genügend Nachbarn im Einflussradius).
    #
    # --- English ---
    # If h is not provided, we choose a didactic default:
    # h = 1.5 * dx (typical: enough neighbors within the influence radius).
    if h is None:
        h_used = 1.5 * float(dx)
    else:
        h_used = float(h)
    _validate_positive("h", float(h_used))

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Partikelgitter erzeugen
    # -------------------------------------------------------------------------
    # Hinweis:
    # - `initialize_particles_cube` benötigt auch die Masse pro Partikel.
    # - Für dieses Demo wählen wir eine feste, kleine Masse.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Create particle grid
    # -------------------------------------------------------------------------
    # Note:
    # - `initialize_particles_cube` also requires the mass per particle.
    # - For this demo we choose a fixed, small mass.
    mass_per_particle = 0.01
    particles = initialize_particles_cube(
        L=float(L),
        dx=float(dx),
        rho0=float(rho0),
        mass_per_particle=float(mass_per_particle),
    )
    N = particles.n
    if N <= 0:
        raise ValueError("Es wurden keine Partikel erzeugt (N == 0).")

    # --- Deutsch ---
    # Fokusindex clampen (anfängerfreundlich):
    #
    # --- English ---
    # Clamp focus index (beginner-friendly):
    if int(focus_index) < 0 or int(focus_index) >= int(N):
        focus_index = 0

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Lokale Kopien anlegen (wir verändern `particles` nicht dauerhaft)
    # -------------------------------------------------------------------------
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Create local copies (we do not permanently modify `particles`)
    # -------------------------------------------------------------------------
    x = np.asarray(particles.x, dtype=np.float64).copy()
    y = np.asarray(particles.y, dtype=np.float64).copy()
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    m = np.asarray(particles.m, dtype=np.float64).copy()

    # --- Deutsch ---
    # "Before"-Zustand speichern:
    #
    # --- English ---
    # Save the "before" state:
    x0 = x.copy()
    y0 = y.copy()

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Sehr kleine Asymmetrie (Perturbation) nur für das Demo
    # -------------------------------------------------------------------------
    # Idee:
    # - Wir wählen ein Partikel nahe der Mitte (hier: j = N//2 als sehr einfache Wahl).
    # - Wir verschieben es minimal nach rechts: x[j] += 0.25*dx
    # - Dadurch entsteht ein kleiner Dichte-/Druckunterschied → Druckgradient → Kraft.
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Very small asymmetry (perturbation) for the demo only
    # -------------------------------------------------------------------------
    # Idea:
    # - We pick a particle near the middle (here: j = N//2 as a very simple choice).
    # - We shift it slightly to the right: x[j] += 0.25*dx
    # - This creates a small density/pressure difference → pressure gradient → force.
    j = int(N // 2)
    x[j] = float(x[j]) + 0.25 * float(dx)

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Mini-Simulation: mehrere Schritte, damit Bewegung sichtbar wird
    # -------------------------------------------------------------------------
    # Wir nutzen explicit Euler (nur Demo):
    # - v += a * dt
    # - x += v * dt
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Mini simulation: multiple steps so motion becomes visible
    # -------------------------------------------------------------------------
    # We use explicit Euler (demo only):
    # - v += a * dt
    # - x += v * dt
    rho = np.zeros(N, dtype=np.float64)
    p_demo = np.zeros(N, dtype=np.float64)
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)

    for _step in range(int(num_steps)):
        # --- Deutsch ---
        # Temporäres ParticleSet2D bauen:
        # - Wichtig: Wir geben unsere lokalen Arrays hinein.
        # - rho/p in diesem Objekt sind "Dummy" und werden NICHT verwendet.
        #
        # --- English ---
        # Build a temporary ParticleSet2D:
        # - Important: we pass in our local arrays.
        # - rho/p in this object are "dummy" and are NOT used.
        temp_particles = ParticleSet2D(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            rho=np.zeros(N, dtype=np.float64),
            p=np.zeros(N, dtype=np.float64),
            m=m,
        )

        # --- Deutsch ---
        # Dichte berechnen (Grid bevorzugt, sonst naiv):
        #
        # --- English ---
        # Compute density (prefer grid, otherwise naive):
        if compute_density_grid is not None:
            rho = compute_density_grid(particles=temp_particles, h=float(h_used), cell_size=float(h_used))
        else:
            rho = compute_density_naive(particles=temp_particles, h=float(h_used))

        # --- Deutsch ---
        # Demo-Referenzdichte:
        # - Wir setzen rho0_demo leicht unter mean(rho), damit p_demo (geclamped) nicht überall 0 ist.
        #
        # --- English ---
        # Demo reference density:
        # - We set rho0_demo slightly below mean(rho) so clamped p_demo is not all zeros.
        rho0_demo = float(np.mean(rho) * 0.95) if rho.size > 0 else 0.0

        # --- Deutsch ---
        # Druck über EOS (geclamped):
        #
        # --- English ---
        # Pressure via EOS (clamped):
        p_demo = compute_pressure_eos(rho=rho, rho0=float(rho0_demo), k=float(k), clamp_negative=True)

        # --- Deutsch ---
        # Druckbeschleunigung (Kräfte) berechnen:
        #
        # --- English ---
        # Compute pressure acceleration (forces):
        ax, ay = compute_pressure_acceleration(
            particles=temp_particles,
            rho=rho,
            p=p_demo,
            h=float(h_used),
            cell_size=float(h_used),
        )

        # --- Deutsch ---
        # Euler-Integration:
        #
        # --- English ---
        # Euler integration:
        vx = vx + ax * float(dt)
        vy = vy + ay * float(dt)
        x = x + vx * float(dt)
        y = y + vy * float(dt)

    # --- Deutsch ---
    # Diagnosewerte für die Textbox:
    #
    # --- English ---
    # Diagnostic values for the textbox:
    rho_min = float(np.min(rho)) if rho.size > 0 else 0.0
    rho_max = float(np.max(rho)) if rho.size > 0 else 0.0
    p_min = float(np.min(p_demo)) if p_demo.size > 0 else 0.0
    p_max = float(np.max(p_demo)) if p_demo.size > 0 else 0.0

    a_mag = np.sqrt(ax * ax + ay * ay)
    a_mag_max = float(np.max(a_mag)) if a_mag.size > 0 else 0.0

    disp = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    disp_max = float(np.max(disp)) if disp.size > 0 else 0.0

    # --- Deutsch ---
    # Pfeile sichtbar machen (nur Visualisierung):
    # - Wir skalieren so, dass der größte Pfeil ungefähr Länge 0.15 im Plot hat.
    #
    # --- English ---
    # Make arrows visible (visualization only):
    # - We scale so the largest arrow has roughly length 0.15 in plot units.
    if a_mag_max > 0.0:
        arrow_mult = 0.15 / float(a_mag_max)
    else:
        arrow_mult = 1.0

    # --- Deutsch ---
    # -------------------------------------------------------------------------
    # Plot: eine ruhige 2-Panel-Figure
    # -------------------------------------------------------------------------
    #
    # --- English ---
    # -------------------------------------------------------------------------
    # Plot: one calm 2-panel figure
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Deutsch ---
    # Links: Before vs After
    #
    # --- English ---
    # Left: before vs after
    axs[0].scatter(x0, y0, s=22, c="lightgray", label="before", zorder=1)
    sc_after = axs[0].scatter(x, y, s=30, c=p_demo, cmap="viridis", label="after (colored by p_demo)", zorder=2)
    cbar = fig.colorbar(sc_after, ax=axs[0])
    cbar.set_label("demo pressure p_demo (clamped)")
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_title("Pressure force demo: before vs after")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="best")

    # --- Deutsch ---
    # Rechts: Pfeile (quiver) für den letzten Schritt
    #
    # --- English ---
    # Right: arrows (quiver) for the last step
    axs[1].scatter(x, y, s=22, c="lightgray", label="particles", zorder=1)

    if N <= 600:
        axs[1].quiver(
            x,
            y,
            ax * float(arrow_mult),
            ay * float(arrow_mult),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="tab:blue",
            alpha=0.35,
            width=0.0028,
            zorder=2,
        )

    # --- Deutsch ---
    # Fokus-Pfeil immer zeichnen (Mindestanforderung).
    #
    # --- English ---
    # Always draw the focus arrow (minimum requirement).
    axs[1].quiver(
        [float(x[int(focus_index)])],
        [float(y[int(focus_index)])],
        [float(ax[int(focus_index)]) * float(arrow_mult)],
        [float(ay[int(focus_index)]) * float(arrow_mult)],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:purple",
        alpha=1.0,
        width=0.006,
        zorder=3,
        label="focus force (scaled)",
    )

    axs[1].set_aspect("equal", adjustable="box")
    axs[1].set_title("Pressure acceleration (arrows, last step)")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="best")

    # --- Deutsch ---
    # Große Diagnose-Textbox (kein Print):
    #
    # --- English ---
    # Large diagnostics textbox (no prints):
    fig.text(
        0.02,
        0.02,
        "Diagnostics (pressure force demo)\n"
        "N = {N}\n"
        "dx = {dx}\n"
        "h = {h}\n"
        "dt = {dt}\n"
        "num_steps = {ns}\n"
        "rho0 = {rho0}\n"
        "k = {k}\n\n"
        "rho min/max = {rmin} / {rmax}\n"
        "p_demo min/max = {pmin} / {pmax}\n"
        "max(|a|) = {amax}\n"
        "max displacement = {dmax}".format(
            N=int(N),
            dx=f"{float(dx):.6g}",
            h=f"{float(h_used):.6g}",
            dt=f"{float(dt):.6g}",
            ns=int(num_steps),
            rho0=f"{float(rho0):.6g}",
            k=f"{float(k):.6g}",
            rmin=f"{float(rho_min):.12g}",
            rmax=f"{float(rho_max):.12g}",
            pmin=f"{float(p_min):.12g}",
            pmax=f"{float(p_max):.12g}",
            amax=f"{float(a_mag_max):.12g}",
            dmax=f"{float(disp_max):.12g}",
        ),
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )

    fig.suptitle("SPH pressure force demo (optional add-on)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.93])
    plt.show()


if __name__ == "__main__":
    run_learning_viz_animation()
    # --- Deutsch ---
    # Optional: statische Lern-Visualisierung (ohne Animation).
    #
    # --- English ---
    # Optional: static learning visualization (no animation).
    # run_learning_viz()
    # --- Deutsch ---
    # Optional: separater Zusatz-Demo-Modus für Druckkräfte (ändert `run_learning_viz()` nicht).
    #
    # --- English ---
    # Optional: separate add-on demo mode for pressure forces (does not change `run_learning_viz()`).
    # run_pressure_force_demo()


