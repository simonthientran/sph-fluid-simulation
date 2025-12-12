"""
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
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sph_sim.core.particles import initialize_particles_cube
from sph_sim.core.kernels import poly6_kernel
from sph_sim.core.density import compute_density_naive


def _validate_positive(name: str, value: float) -> None:
    """
    Kleine Hilfsfunktion: Prüfe, dass ein Parameter > 0 ist.

    Warum eine Hilfsfunktion?
    - Wir brauchen diese Prüfung an mehreren Stellen.
    - Eine zentrale Funktion hält die Fehlermeldungen konsistent und den Code lesbar.
    """

    if value <= 0.0:
        raise ValueError(f"{name} muss > 0 sein.")


def _distances_to_focus(x: np.ndarray, y: np.ndarray, focus_index: int) -> np.ndarray:
    """
    Berechne die Distanz jedes Partikels zum Fokus-Partikel.

    Rückgabe:
    - r: 1D-Array (Länge N), r[k] ist die Distanz vom Partikel k zum Fokus-Partikel.

    Hinweis:
    - Wir nutzen hier NumPy-Vektorisierung, weil es sehr lesbar ist:
      dx = x - x_focus ist ein Array-Array-Operator.
    """

    x_focus = float(x[focus_index])
    y_focus = float(y[focus_index])

    dx = x - x_focus
    dy = y - y_focus

    # Euklidische Distanz r = sqrt(dx^2 + dy^2) für alle Partikel gleichzeitig
    return np.sqrt(dx * dx + dy * dy)


def plot_particles(ax, x, y, focus_index: int, h: float) -> None:
    """
    Plot: Partikel als Punkte + Fokus-Partikel + Kreis mit Radius h + Nachbarn.

    Was zeigen wir hier?
    - Alle Partikelpositionen als Scatter-Plot (Punkte).
    - Ein Fokus-Partikel (z.B. Partikel 0) deutlich hervorgehoben.
    - Einen Kreis mit Radius h um den Fokus: das ist der Einflussradius des Kernels.
    - Optional: alle Nachbarn innerhalb des Radius (r <= h) anders einfärben.

    Warum ist das didaktisch wichtig?
    - In SPH hat jedes Partikel nur "Nachbarn" innerhalb einer bestimmten Reichweite (h).
    - Diese Visualisierung macht den Begriff "compact support" sofort sichtbar.
    """

    # Distanz jedes Partikels zum Fokus-Partikel berechnen.
    r = _distances_to_focus(x, y, focus_index)

    # Nachbarn sind alle Partikel mit r <= h.
    # (Wir zählen auch das Fokus-Partikel selbst dazu, weil r=0 <= h ist.)
    inside_mask = r <= h

    # Alle Partikel (leicht grau) – das ist der "Hintergrund".
    ax.scatter(x, y, s=25, c="lightgray", label="alle Partikel")

    # Nachbarn (innerhalb h) – auffälliger, damit man sieht "wer zählt".
    ax.scatter(
        x[inside_mask],
        y[inside_mask],
        s=40,
        c="tab:blue",
        label=f"Nachbarn (r ≤ h), Anzahl={int(np.sum(inside_mask))}",
    )

    # Fokus-Partikel (stark hervorgehoben).
    ax.scatter(
        [x[focus_index]],
        [y[focus_index]],
        s=120,
        c="tab:red",
        marker="*",
        label=f"Fokus-Partikel i={focus_index}",
        zorder=5,  # über den anderen Punkten zeichnen
    )

    # Kreis um das Fokus-Partikel:
    # - Das ist die geometrische Darstellung des Einflussradius h.
    circle = Circle(
        (float(x[focus_index]), float(y[focus_index])),
        radius=float(h),
        fill=False,
        color="tab:red",
        linewidth=2.0,
        alpha=0.9,
        label="Radius h (Einflussbereich)",
    )
    ax.add_patch(circle)

    # Achsen/Look & Feel
    ax.set_aspect("equal", adjustable="box")  # gleiche Skalierung in x und y
    ax.set_title("Partikel-Layout + Fokus + Nachbarn innerhalb h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_poly6_kernel_curve(ax, h: float, num_samples: int = 200) -> None:
    """
    Plot: Poly6-Kernelkurve W(r, h) über r.

    Was zeigen wir?
    - r von 0 bis 1.5*h (damit man auch den Bereich *außerhalb* des Supports sieht).
    - W(r,h) als Kurve.
    - Eine vertikale Linie bei r = h, um den Übergang "ab hier 0" zu markieren.

    Didaktik:
    - Innerhalb von h ist der Kernel > 0 (für r < h).
    - Außerhalb von h ist der Kernel exakt 0 ("compact support").
    """

    _validate_positive("h", h)

    # r-Werte sampeln: gleichmäßig von 0 bis 1.5*h
    r = np.linspace(0.0, 1.5 * h, int(num_samples), dtype=np.float64)

    # Kernelwerte berechnen (funktioniert direkt mit Arrays).
    W = poly6_kernel(r, h)

    ax.plot(r, W, color="tab:purple", linewidth=2.0, label="Poly6: W(r,h)")

    # Vertikale Linie bei r = h: ab hier ist der Kernel 0.
    ax.axvline(h, color="black", linestyle="--", linewidth=1.5, label="r = h (Support-Grenze)")

    # Kleine Textbox, die die wichtigste Aussage zusammenfasst.
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
    """

    # Scatter mit Farbskala (c=rho)
    sc = ax.scatter(x, y, c=rho, s=45, cmap="viridis")

    # Colorbar gehört zur ganzen Figure, aber wir hängen sie an diese Achse.
    cbar = ax.figure.colorbar(sc, ax=ax)
    cbar.set_label("density rho")

    # Fokus + Kreis wieder zeichnen (wie im Partikelplot)
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

    # Textbox mit Min/Max und der zentralen Formel
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
    """

    # --- Eingaben validieren (frühe, klare Fehler) ----------------------------
    _validate_positive("L", L)
    _validate_positive("dx", dx)
    _validate_positive("rho0", rho0)
    _validate_positive("mass_per_particle", mass_per_particle)
    _validate_positive("h", h)

    # Eine sehr einfache Plausibilitätsprüfung:
    # Wenn dx >= L, entstehen in der Regel 0 oder 1 Partikel pro Achse → nicht sinnvoll.
    if dx >= L:
        raise ValueError("dx muss kleiner als L sein (dx < L), sonst entstehen zu wenige Partikel.")

    # --- Partikel erzeugen ----------------------------------------------------
    particles = initialize_particles_cube(
        L=L,
        dx=dx,
        rho0=rho0,
        mass_per_particle=mass_per_particle,
    )

    N = particles.n

    # --- Fokusindex "clampen" -------------------------------------------------
    # Anfängerfreundliche Entscheidung:
    # Wenn jemand einen ungültigen Index eingibt, ist das wahrscheinlich ein Tippfehler.
    # Statt das ganze Skript abstürzen zu lassen, setzen wir auf 0 zurück.
    # (In manchen Projekten würde man hier strikt ValueError werfen – beides ist okay.
    # Für dieses Lernskript ist ein sanfter Fallback oft angenehmer.)
    if focus_index < 0 or focus_index >= N:
        focus_index = 0

    # --- Dichte berechnen (naiv, O(N^2)) -------------------------------------
    rho = compute_density_naive(particles=particles, h=h)

    # --- Figure mit 3 Subplots ------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Links: Partikel + Fokus + Nachbarn
    plot_particles(axs[0], particles.x, particles.y, focus_index=focus_index, h=h)

    # Mitte: Kernelkurve
    plot_poly6_kernel_curve(axs[1], h=h, num_samples=200)

    # Rechts: Dichte als Farbe
    plot_density(axs[2], particles.x, particles.y, rho, focus_index=focus_index, h=h)

    # Großer Titel über der gesamten Figure
    fig.suptitle(
        "SPH Learning Viz: Partikel-Layout, Poly6-Kernel, naive Dichte",
        fontsize=14,
        y=0.98,
    )

    # Kleine Parameter-Zusammenfassung als Untertitel (Textbox oben)
    fig.text(
        0.5,
        0.94,
        f"L={L}, dx={dx}, h={h}, N={N}, rho0={rho0}, m={mass_per_particle}, focus_index={focus_index}",
        ha="center",
        va="top",
        fontsize=10,
    )

    # Layout verbessern: Platz für Titel/Untertitel lassen
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    # Anzeigen (kein File-I/O, keine Prints)
    plt.show()


if __name__ == "__main__":
    run_learning_viz()


