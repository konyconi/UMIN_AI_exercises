# -*- coding: utf-8 -*-
"""Vygeneruje README.md jen pro ta cviceni, ktera jsou uz zverejnena.

Duvod: na GitHubu nema stat tabulka s trinacti odkazy, z nichz jedenact vede
na soubory, ktere tam jeste nejsou. Pri zverejneni dalsiho cviceni se zmeni
jedine cislo.

Pouziti:
    python readme_gen.py 2          .. README pro cviceni 1 az 2
    python readme_gen.py 1 2 5      .. README pro vyjmenovana cviceni

Krome README.md zapise jeste zdroje/publikovat.txt -- seznam cest, ktere ma
upload_cviceni.bat poslat na GitHub. Ten seznam je tedy na jednom miste
a nemusi se udrzovat zvlast.
"""
import io
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = 'https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main'
BADGE = ('[![Colab](https://colab.research.google.com/assets/'
         'colab-badge.svg)]')

NAZVY = {
    1: 'Perceptron, lineární separabilita, baseline',
    2: 'Dopředné sítě a backpropagation',
    3: 'Úlohy, reprezentace dat a ztrátové funkce',
    4: 'Optimalizace učení',
    5: 'Generalizace a regularizace',
    6: 'Praktické trénování hlubokých sítí',
    7: 'Konvoluční neuronové sítě',
    8: 'Přenos učení a Vision Transformer',
    9: 'Sekvenční modely a vektory slov',
    10: 'Transformer',
    11: 'Jazykové modely a LLM',
    12: 'Robustnost, nejistota a změna distribuce',
    13: 'Volitelné trasy a semestrální projekt',
}

# notebooky k praktickym prednaskam (UMIN_DL_prNNx_*.ipynb); zverejni se
# vzdy, kdyz lezi v notebooks/, a v tabulce stoji za cvicenim s nizsim cislem
PRAKTICKE = {
    '05b': ('První síť od dat po test (praktická přednáška)',
            '+ `torch`, `torchvision`', 'MNIST (12 MB, stáhne se sám)',
            'asi 7 minut'),
}

# notebooky k predmetu AKTI (akti/AKTI_*.ipynb, generuji se ze zdroje/akti_*.py);
# v README se objevi jen ty, ktere v akti/ opravdu lezi
AKTI = [
    ('AKTI_MLP', 'akti_mlp', 'Vícevrstvý perceptron (MLP) na MNIST'),
    ('AKTI_zapocet_sablona', 'akti_zapocet',
     'Šablona zápočtového úkolu (studie na Fashion-MNIST)'),
]

# skupiny pro tabulku naroku; emituji se jen ty, ktere se zverejnenymi
# cvicenimi prekryvaji
NAROKY = [
    (set(range(1, 7)), '`numpy`, `matplotlib`, `scikit-learn`',
     'žádná (generují se)', 'do 1 minuty'),
    ({7, 12}, '+ `torch`, `torchvision`',
     'MNIST (12 MB, stáhne se sám)', '2–4 minuty'),
    ({8}, '+ `torch`, `torchvision`',
     'CIFAR-10 (170 MB) a ResNet-18 (45 MB)', '5–8 minut'),
    ({9, 10, 11, 13}, '`numpy`, `scikit-learn`, `torch`',
     'žádná', 'do 2 minut'),
]


def slugy():
    """{cislo: slug} podle souboru, ktere skutecne lezi v notebooks/."""
    out = {}
    for jm in os.listdir(os.path.join(REPO, 'notebooks')):
        m = re.match(r'UMIN_DL_cv(\d\d)_(.+)\.ipynb$', jm)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def prakticke():
    """{'05b': slug} podle souboru UMIN_DL_prNNx_*.ipynb v notebooks/."""
    out = {}
    for jm in os.listdir(os.path.join(REPO, 'notebooks')):
        m = re.match(r'UMIN_DL_pr(\d\d[a-z])_(.+)\.ipynb$', jm)
        if m and m.group(1) in PRAKTICKE:
            out[m.group(1)] = m.group(2)
    return out


def rozsah(cisla):
    """[1,2] -> '01–02';  [7,12] -> '07, 12';  [1,2,3,5] -> '01–03, 05'."""
    cisla = sorted(cisla)
    kusy, i = [], 0
    while i < len(cisla):
        j = i
        while j + 1 < len(cisla) and cisla[j + 1] == cisla[j] + 1:
            j += 1
        if j - i >= 1:
            kusy.append('%02d–%02d' % (cisla[i], cisla[j]))
        else:
            kusy.append('%02d' % cisla[i])
        i = j + 1
    return ', '.join(kusy)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    if len(argv) == 1:
        publik = list(range(1, int(argv[0]) + 1))
    else:
        publik = sorted(int(a) for a in argv)

    sl = slugy()
    chybi = [c for c in publik if c not in sl]
    if chybi:
        print('CHYBA: pro cviceni %s neni notebook v notebooks/'
              % ', '.join(str(c) for c in chybi))
        return 1

    # --------------------------------------------------------- tabulka
    pr = prakticke()
    polozky = [('%02d' % c, NAZVY[c], 'UMIN_DL_cv%02d_%s' % (c, sl[c]))
               for c in publik]
    polozky += [(k, PRAKTICKE[k][0], 'UMIN_DL_pr%s_%s' % (k, pr[k])) for k in pr]
    radky = []
    for cis, nazev, base in sorted(polozky):
        zad = 'notebooks/%s.ipynb' % base
        res = 'reseni/%s_reseni.ipynb' % base
        radky.append('| %s | %s | %s(%s/%s) | [řešení](%s/%s) |'
                     % (cis, nazev, BADGE, GH, zad, GH, res))

    if max(publik) < 13:
        pozn = ('\nDalší cvičení se tu budou objevovat průběžně, vždy v týdnu'
                '\npříslušné přednášky. Zatím jsou hotová cvičení %s'
                ' z třinácti.\n' % rozsah(publik))
    else:
        pozn = ''

    naroky = []
    for cisla, knih, data, cas in NAROKY:
        prekryv = cisla & set(publik)
        if prekryv:
            naroky.append('| %s | %s | %s | %s |'
                          % (rozsah(prekryv), knih, data, cas))
    for k in sorted(pr):
        naroky.append('| %s | %s | %s | %s |' % ((k,) + PRAKTICKE[k][1:]))

    akti = [(nb, zdroj, nazev) for nb, zdroj, nazev in AKTI
            if os.path.exists(os.path.join(REPO, 'akti', nb + '.ipynb'))]
    if akti:
        akti_sekce = ('## AKTI\n\n| Notebook | Otevřít |\n|---|---|\n'
                      + '\n'.join('| %s | %s(%s/akti/%s.ipynb) |'
                                  % (nazev, BADGE, GH, nb)
                                  for nb, _, nazev in akti)
                      + '\n\n')
    else:
        akti_sekce = ''

    torch_pozn = ''
    if not (set(publik) & {7, 8, 9, 10, 11, 12, 13}) and not pr:
        torch_pozn = ('\n\n`requirements.txt` uvádí i `torch`; ten je potřeba'
                      '\nteprve od sedmého cvičení.')

    readme = """# Cvičení ke KMI/UMIN — Deep Learning

Notebooky ke svazku II (*Deep Learning*). Číslování odpovídá přednáškám:
cvičení *N* navazuje na přednášku *N* a na příslušnou kapitolu skript.

Každý notebook obsahuje výklad, spustitelný kód a několik míst označených
`# TODO`, která jsou na studentech. Vyřešené verze jsou ve složce
[`reseni/`](reseni) — jsou spustitelné odshora dolů a slouží ke kontrole,
ne k opisování.

Vše běží v Google Colabu bez nastavování. GPU není potřeba; kde se hodí,
notebook si ji vezme sám.

| # | Téma | Zadání | Řešení |
|---|---|---|---|
%s
%s
%s## Jak s notebooky pracovat

- Otevřete notebook tlačítkem výše a hned dejte **Kopírovat na Disk**
  (*Copy to Drive*). Bez toho je jen ke čtení a vaše změny se neuloží.
- Buňky spouštějte postupně shora dolů (**Shift+Enter**).
- Místa označená `# TODO` jsou zadání. Dokud je nedoplníte, další buňky spadnou
  na `NotImplementedError` — to je záměr.
- Řada úloh končí **otázkou k zamyšlení**. Odpověď se nikam neodevzdává, ale
  stojí za to si ji formulovat.

## Co která cvičení potřebují

| | knihovny | data | přibližný čas na CPU |
|---|---|---|---|
%s

V Colabu je všechno předinstalované. Lokálně stačí `pip install -r
requirements.txt`.%s

## Údržba

Notebooky se generují ze skriptů ve složce [`zdroje/`](zdroje) — zadání
i řešení z jednoho zdroje, aby se nemohly rozejít. Jak generátor funguje, je
v [zdroje/README.md](zdroje/README.md). Tenhle soubor se generuje taky,
skriptem `zdroje/readme_gen.py`, aby tabulka odpovídala tomu, co je
opravdu zveřejněné — ruční úpravy v něm se při dalším zveřejnění ztratí.

## Rok 2024

Materiály z předchozího běhu předmětu, který měl jiný sylabus, jsou ve složce
[`2024/`](2024). Nejsou udržované a nenavazují na aktuální přednášky.
""" % ('\n'.join(radky), pozn, akti_sekce, '\n'.join(naroky), torch_pozn)

    cesta = os.path.join(REPO, 'README.md')
    io.open(cesta, 'w', encoding='utf-8', newline='\r\n').write(readme)
    print('README.md    zapsan pro cviceni %s' % rozsah(publik))

    # --------------------------------------------- seznam pro upload
    soubory = ['.gitignore', 'README.md', 'requirements.txt',
               'zdroje/README.md', 'zdroje/nbbuild.py', 'zdroje/kontrola.py',
               'zdroje/nbrun.py', 'zdroje/readme_gen.py']
    for c in publik:
        soubory += ['notebooks/UMIN_DL_cv%02d_%s.ipynb' % (c, sl[c]),
                    'reseni/UMIN_DL_cv%02d_%s_reseni.ipynb' % (c, sl[c]),
                    'zdroje/cv%02d_%s.py' % (c, sl[c])]
    for k in sorted(pr):
        soubory += ['notebooks/UMIN_DL_pr%s_%s.ipynb' % (k, pr[k]),
                    'reseni/UMIN_DL_pr%s_%s_reseni.ipynb' % (k, pr[k]),
                    'zdroje/pr%s_%s.py' % (k, pr[k])]
    for nb, zdroj, _ in akti:
        soubory += ['akti/%s.ipynb' % nb, 'zdroje/%s.py' % zdroj]
    chybne = [f for f in soubory
              if not os.path.exists(os.path.join(REPO, f))]
    if chybne:
        print('CHYBA: tyto soubory neexistuji: %s' % ', '.join(chybne))
        return 1
    # CRLF, at to 'for /f' v cmd precte spolehlive
    sez = os.path.join(REPO, 'zdroje', 'publikovat.txt')
    io.open(sez, 'w', encoding='ascii', newline='\r\n').write(
        '\n'.join(soubory) + '\n')
    print('publikovat.txt  %d souboru' % len(soubory))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
