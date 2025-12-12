"""
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
"""

from pathlib import Path
import sys

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

# Projekt-Root bestimmen:
# - Dieses Skript liegt in `<projekt>/scripts/`.
# - `parents[1]` ist dann `<projekt>/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pfad zu `src/` (dort liegt unser importierbarer Code).
SRC_PATH = PROJECT_ROOT / "src"

# `sys.path` enthält Strings, keine Path-Objekte → daher `str(SRC_PATH)`.
if str(SRC_PATH) not in sys.path:
    # `insert(0, ...)` setzt den Pfad ganz vorne.
    # Vorteil: Python sucht zuerst im Projekt-Quellcode, bevor es anderswo sucht.
    sys.path.insert(0, str(SRC_PATH))

# -----------------------------------------------------------------------------
# Jetzt können wir sauber aus unserem Projekt importieren.
# -----------------------------------------------------------------------------
from sph_sim.vis.learning_viz import run_learning_viz


def main() -> None:
    """
    Starte die Learning-Visualisierung.

    Warum eine `main()`-Funktion?
    - Das ist ein übliches Python-Muster.
    - Es macht das Skript übersichtlicher und besser erweiterbar.
    - Später könnten wir hier z.B. Argumente (L, dx, h, ...) verarbeiten.
    """

    # Wir starten die Visualisierung mit Default-Parametern.
    # (Default-Parameter sind im Modul `sph_sim.vis.learning_viz` dokumentiert.)
    run_learning_viz()


if __name__ == "__main__":
    # Dieser Block sorgt dafür, dass `main()` nur ausgeführt wird,
    # wenn du diese Datei direkt startest:
    #
    #   python scripts/run_learning_viz.py
    #
    # Wenn man die Datei importieren würde, passiert hier nichts automatisch.
    main()


