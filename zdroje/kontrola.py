# -*- coding: utf-8 -*-
"""Kontrola vsech notebooku: platnost, pocet ukolu, zadne unikle reseni."""
import io, os, glob
import nbformat

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print('%-46s bunek  kod  TODO  Colab' % 'soubor')
chyby = []
for slozka in ('notebooks', 'reseni'):
    for path in sorted(glob.glob(os.path.join(REPO, slozka, '*.ipynb'))):
        jmeno = os.path.basename(path)
        nb = nbformat.read(io.open(path, encoding='utf-8'), as_version=4)
        try:
            nbformat.validate(nb)
        except Exception as e:
            chyby.append('%s: nevalidni (%s)' % (jmeno, str(e)[:60]))

        kod = [c for c in nb.cells if c.cell_type == 'code']
        zdroj_kod = '\n'.join(''.join(c.source) for c in kod)
        vse = '\n'.join(''.join(c.source) for c in nb.cells)

        todo = zdroj_kod.count('# TODO')
        badge = 'colab.research.google.com' in vse
        print('%-46s %5d %4d %5d  %s'
              % (os.path.join(slozka, jmeno), len(nb.cells), len(kod), todo,
                 'ano' if badge else 'NE'))

        if slozka == 'notebooks' and todo == 0:
            chyby.append('%s: zadne TODO v kodu' % jmeno)
        if slozka == 'reseni' and todo > 0:
            chyby.append('%s: v reseni zbylo TODO' % jmeno)
        if slozka == 'reseni' and 'NotImplementedError' in zdroj_kod:
            chyby.append('%s: v reseni zbyl NotImplementedError' % jmeno)
        if not badge:
            chyby.append('%s: chybi Colab odkaz' % jmeno)
        elif ('%s/%s' % (slozka, jmeno)) not in vse:
            chyby.append('%s: Colab odkaz nevede na sebe' % jmeno)

print()
print('PROBLEMY:' if chyby else 'vsechny notebooky v poradku')
for c in chyby:
    print('  -', c)
