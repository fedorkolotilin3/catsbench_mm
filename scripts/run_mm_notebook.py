"""Execute and save the fixed comparison using this Python environment."""

import os
os.environ["LSE_BACKEND"] = "cpu"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "notebooks" / "dlight_sb_mm.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=900,
                            resources={"metadata": {"path": str(ROOT)}})

    def cell_started(cell, cell_index):
        print(f"Cell {cell_index}: {cell.cell_type}", flush=True)

    client.on_cell_start = cell_started
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    assert all(o.output_type != "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
    print(f"Executed and validated {path}", flush=True)


if __name__ == "__main__":
    main()
