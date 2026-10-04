# Zdroje notebooků

Notebooky se **negenerují ručně**. Zadání i řešení vznikají z jednoho zdroje,
aby se nemohly rozejít: kdykoli se změní kód úlohy, změní se obě verze naráz.

## Jak to funguje

Každý soubor `cvNN_*.py` je seznam buněk:

```python
from nbbuild import md, code, task, build

CELLS = [
    md("markdownový text"),
    code("kód, který je v obou verzích stejný"),
    task(stub="...  # TODO ...", solution="... hotové řešení ..."),
]

build(__file__, cislo=1, nazev="Název cvičení", cells=CELLS)
```

Spuštění `python cv01_perceptron.py` zapíše dva soubory:

- `../notebooks/UMIN_DL_cv01_perceptron.ipynb` — se `stub` verzemi úloh,
- `../reseni/UMIN_DL_cv01_perceptron_reseni.ipynb` — se `solution` verzemi.

Colab odkaz na začátku notebooku se doplní automaticky a vede sám na sebe.

## Přegenerování všech notebooků

```
python cv01_perceptron.py
python cv02_backprop.py
...
```

nebo v jednom kroku (PowerShell):

```powershell
Get-ChildItem cv*.py | ForEach-Object { python $_.Name }
```

## Kontrola

- `python kontrola.py` — ověří, že jsou notebooky validní, že v zadáních jsou
  `# TODO`, že v řešeních žádné nezbyly a že Colab odkazy vedou na správné
  soubory.
- `python nbrun.py reseni/UMIN_DL_cv01_perceptron_reseni.ipynb` — spustí
  řešený notebook odshora dolů a nahlásí první chybu. Před každou změnou,
  která se dotkne kódu, se hodí pustit všechna řešení.

Ke spuštění kontrol je potřeba `nbformat` a `nbclient` (nejsou v
`requirements.txt`, protože je studenti nepotřebují):

```
pip install nbformat nbclient ipykernel
```
