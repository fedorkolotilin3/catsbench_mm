import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import nbformat
from IPython.core.inputtransformer2 import TransformerManager


class NotebookLaunchTest(unittest.TestCase):
    def test_schema_syntax_and_commands_without_training(self):
        root = Path(__file__).resolve().parents[1]
        for suffix in ('local', 'colab'):
            path = root / 'notebooks' / f'run_dlight_sb_pairs_{suffix}.ipynb'
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            sources = [c.source for c in notebook.cells if c.cell_type == 'code']
            for source in sources:
                compile(TransformerManager().transform_cell(source), str(path), 'exec')
            namespace = dict(Path=Path, os=os, sys=sys, subprocess=subprocess,
                             PROJECT_DIR=root, RUNS_DIR=root / 'logs/runs',
                             RUNS={}, SEED=42,
                             perf_counter=lambda: 0.)
            runner = next(s for s in sources if 'def experiment_name(' in s)
            exec(runner, namespace)
            commands = []
            namespace['run_streaming'] = lambda command, *args: commands.append(command)
            namespace['latest_completed_run'] = lambda experiment: root / 'logs' / experiment
            for dimension in (2, 16, 64):
                for kind in ('baseline', 'mm', 'mm_regularized'):
                    commands.clear()
                    namespace['run_experiment'](kind, dimension)
                    joined = json.dumps(commands)
                    if kind == 'mm_regularized':
                        self.assertIn(f'mm_regularized/benchmark_hd/d{dimension}_g002', joined)
                    if suffix == 'local':
                        self.assertEqual(len(commands), 2)
                for action in ('run_experiment', 'export_run'):
                    self.assertTrue(any(f'{action}("mm_regularized", {dimension})' in s
                                        for s in sources))
            exporter = next(s for s in sources if 'def export_run(' in s)
            tree = ast.parse(exporter)
            allowed = [n for n in ast.walk(tree) if isinstance(n, ast.Set)]
            self.assertTrue(any('mm_regularized' in [getattr(e, 'value', None) for e in n.elts]
                                for n in allowed))


if __name__ == '__main__':
    unittest.main()
