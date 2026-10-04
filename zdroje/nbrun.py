# -*- coding: utf-8 -*-
"""Spusti notebook a nahlasi prvni chybu. Vystupy neuklada zpet do souboru."""
import io, os, sys, json
os.environ.setdefault('MPLBACKEND', 'Agg')
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(rel, timeout=600):
    path = os.path.join(REPO, rel)
    nb = nbformat.read(io.open(path, encoding='utf-8'), as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name='python3',
                            allow_errors=False,
                            resources={'metadata': {'path': REPO}})
    try:
        client.execute()
    except CellExecutionError as e:
        txt = str(e)
        print('CHYBA v %s' % rel)
        print(txt[-1800:])
        return False
    n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
    print('OK   %-52s (%d kodovych bunek)' % (rel, n_code))
    return True

if __name__ == '__main__':
    ok = all(run(a) for a in sys.argv[1:])
    sys.exit(0 if ok else 1)
