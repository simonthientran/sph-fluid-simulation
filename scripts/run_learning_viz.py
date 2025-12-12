"""
DE:
Runner-Skript für die Learning-Visualisierung.

Warum gibt es dieses Skript überhaupt?
-------------------------------------
Als Anfänger ist es angenehm, ein einziges "Start"-Skript zu haben, das man ausführen kann,
ohne sich Gedanken über Python-Import-Pfade oder Paket-Installation zu machen.

Dieses Skript macht genau das:
1) Es sorgt dafür, dass Python unser `src/`-Verzeichnis als Import-Pfad kennt.
2) Es importiert `run_learning_viz` aus `sph_sim.vis.learning_viz`.
3) Es startet die Visualisierung.

Wichtig:
- Keine Datei-I/O
- Keine Prints
- Alles passiert über die Matplotlib-Fensteranzeige (`plt.show()` im Visualisierungsmodul).

EN:
Runner script for the learning visualization.

Why does this script exist at all?
As a beginner, it is convenient to have a single "start" script that you can run
without having to think about Python import paths or package installation.

This script does exactly that:
1) It ensures that Python knows our `src/` directory as an import path.
2) It imports `run_learning_viz` from `sph_sim.vis.learning_viz`.
3) It starts the visualization.

Important:
- No file I/O
- No prints
- Everything happens via the Matplotlib window display (`plt.show()` in the visualization module).
"""

from pathlib import Path
import sys

# --- Deutsch ---
# -----------------------------------------------------------------------------
# Was ist `sys.path`?
# -----------------------------------------------------------------------------
# `sys.path` ist eine Liste von Ordnern, in denen Python nach Modulen/Paketen sucht,
# wenn du `import ...` schreibst.
#
# Beispiel:
# - Wenn du `from sph_sim.vis.learning_viz import run_learning_viz` importierst,
#   muss Python irgendwo ein Paket `sph_sim` finden.
#
# In "normal" installierten Paketen (pip install ...) ist das sauber geregelt,
# weil das Paket im Python-Environment registriert ist.
#
# In diesem Projekt nutzen wir aber eine `src/`-Struktur:
# - Der Code liegt unter `src/sph_sim/...`
# - Wenn du das Projekt NICHT als Paket installiert hast, kennt Python `src/` nicht automatisch.
# - Ergebnis: Import-Fehler ("ModuleNotFoundError: No module named 'sph_sim'")
#
# Lösung für Lern-/Entwicklungszwecke:
# - Wir fügen `src/` einmalig vorne in `sys.path` ein.
# - Dann findet Python das Paket `sph_sim` direkt im Quellcode.
#
# Später (professioneller Weg):
# - Man nutzt Packaging (z.B. `pyproject.toml`) und installiert das Projekt
#   z.B. mit `pip install -e .`
# - Dann braucht man diesen sys.path-Block nicht mehr.
#
# Projekt-Root bestimmen:
# - Dieses Skript liegt in `<projekt>/scripts/`.
# - `parents[1]` ist dann `<projekt>/`.
#
# --- English ---
# -----------------------------------------------------------------------------
# What is `sys.path`?
# -----------------------------------------------------------------------------
# `sys.path` is a list of directories in which Python searches for modules/packages
# when you write `import ...`.
#
# Example:
# - If you import `from sph_sim.vis.learning_viz import run_learning_viz`,
#   Python must be able to find a package `sph_sim` somewhere.
#
# In "normally" installed packages (pip install ...), this is handled cleanly
# because the package is registered in the Python environment.
#
# In this project, however, we use a `src/` structure:
# - The code lives under `src/sph_sim/...`
# - If you have NOT installed the project as a package, Python does not automatically know `src/`.
# - Result: import error ("ModuleNotFoundError: No module named 'sph_sim'")
#
# Solution for learning/development purposes:
# - We insert `src/` once at the front of `sys.path`.
# - Then Python finds the `sph_sim` package directly in the source code.
#
# Later (professional approach):
# - You use packaging (e.g., `pyproject.toml`) and install the project
#   e.g., with `pip install -e .`
# - Then you no longer need this sys.path block.
#
# Determine the project root:
# - This script lives in `<project>/scripts/`.
# - `parents[1]` is then `<project>/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Deutsch ---
# Pfad zu `src/` (dort liegt unser importierbarer Code).
#
# --- English ---
# Path to `src/` (this is where our importable code lives).
SRC_PATH = PROJECT_ROOT / "src"

# --- Deutsch ---
# `sys.path` enthält Strings, keine Path-Objekte → daher `str(SRC_PATH)`.
#
# --- English ---
# `sys.path` contains strings, not Path objects → therefore `str(SRC_PATH)`.
if str(SRC_PATH) not in sys.path:
    # --- Deutsch ---
    # `insert(0, ...)` setzt den Pfad ganz vorne.
    # Vorteil: Python sucht zuerst im Projekt-Quellcode, bevor es anderswo sucht.
    #
    # --- English ---
    # `insert(0, ...)` puts the path at the very front.
    # Benefit: Python searches in the project source code first before searching elsewhere.
    sys.path.insert(0, str(SRC_PATH))

# --- Deutsch ---
# -----------------------------------------------------------------------------
# Jetzt können wir sauber aus unserem Projekt importieren.
# -----------------------------------------------------------------------------
#
# --- English ---
# -----------------------------------------------------------------------------
# Now we can import cleanly from our project.
# -----------------------------------------------------------------------------
from sph_sim.vis.learning_viz import run_learning_viz


def main() -> None:
    """
    DE:
    Starte die Learning-Visualisierung.

    Warum eine `main()`-Funktion?
    - Das ist ein übliches Python-Muster.
    - Es macht das Skript übersichtlicher und besser erweiterbar.
    - Später könnten wir hier z.B. Argumente (L, dx, h, ...) verarbeiten.

    EN:
    Start the learning visualization.

    Why a `main()` function?
    - This is a common Python pattern.
    - It makes the script more readable and easier to extend.
    - Later, we could process arguments (L, dx, h, ...) here, for example.
    """

    # --- Deutsch ---
    # Wir starten die Visualisierung mit Default-Parametern.
    # (Default-Parameter sind im Modul `sph_sim.vis.learning_viz` dokumentiert.)
    #
    # --- English ---
    # We start the visualization with default parameters.
    # (Default parameters are documented in the module `sph_sim.vis.learning_viz`.)
    run_learning_viz()


if __name__ == "__main__":
    # --- Deutsch ---
    # Dieser Block sorgt dafür, dass `main()` nur ausgeführt wird,
    # wenn du diese Datei direkt startest:
    #
    #   python scripts/run_learning_viz.py
    #
    # Wenn man die Datei importieren würde, passiert hier nichts automatisch.
    #
    # --- English ---
    # This block ensures that `main()` is only executed
    # when you run this file directly:
    #
    #   python scripts/run_learning_viz.py
    #
    # If you were to import the file, nothing happens automatically here.
    main()


