# -*- coding: utf-8 -*-
"""Generator cvicebnich notebooku pro KMI/UMIN.

Z jednoho zdroje vyrobi dve varianty:
  * studentskou  (ukoly jako `# TODO` s NotImplementedError),
  * resenou      (ukoly vyplnene).

Pouziti ve zdrojovem souboru cvNN_*.py:

    from nbbuild import md, code, task, build
    CELLS = [ md("..."), code("..."), task(stub="...", solution="..."), ... ]
    build(__file__, cislo=1, nazev="Perceptron ...", cells=CELLS)
"""
import io, json, os, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = 'https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main'


class Cell(object):
    def __init__(self, kind, src):
        self.kind = kind
        self.src = src.strip('\n')


class Task(object):
    """Ukol: ve studentske verzi stub, v resene solution."""
    def __init__(self, stub, solution):
        self.stub = stub.strip('\n')
        self.solution = solution.strip('\n')


def md(src):
    return Cell('markdown', src)


def code(src):
    return Cell('code', src)


def task(stub, solution):
    return Task(stub, solution)


def _cell(kind, src, idx=0):
    lines = src.split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    c = {'cell_type': kind, 'id': 'cell-%03d' % idx, 'metadata': {},
         'source': source}
    if kind == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    return c


def _notebook(cells):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10'},
            'colab': {'provenance': [], 'toc_visible': True},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


BADGE = ('[![Open In Colab](https://colab.research.google.com/assets/'
         'colab-badge.svg)](%s/%s)')


def build(srcfile, cislo, nazev, cells, podnadpis=None, predpona='cv',
          druh='Cvičení'):
    # cislo je int (cviceni 1..13) nebo retezec ('04b' u prakticke prednasky)
    cislo = '%02d' % cislo if isinstance(cislo, int) else cislo
    base = 'UMIN_DL_%s%s_%s' % (predpona, cislo, _slug(srcfile))
    stu_path = 'notebooks/%s.ipynb' % base
    sol_path = 'reseni/%s_reseni.ipynb' % base

    for path, resene in ((stu_path, False), (sol_path, True)):
        head = '# %s %s — %s%s\n\n%s' % (
            druh, cislo, nazev,
            '\n\n*Řešení.*' if resene else '',
            BADGE % (GH, path))
        out, n = [_cell('markdown', head, 0)], 0
        if podnadpis:
            out.append(_cell('markdown', podnadpis, len(out)))
        for c in cells:
            if isinstance(c, Task):
                n += 1
                body = c.solution if resene else c.stub
                out.append(_cell('code', body, len(out)))
            else:
                out.append(_cell(c.kind, c.src, len(out)))
        full = os.path.join(REPO, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with io.open(full, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(_notebook(out), f, ensure_ascii=False, indent=1)
            f.write('\n')
        print('%-58s bunek=%d ukolu=%d' % (path, len(out), n))
    return stu_path, sol_path


def _slug(srcfile):
    name = os.path.splitext(os.path.basename(srcfile))[0]
    return re.sub(r'^(cv\d+|pr\d+[a-z]?)_', '', name)
