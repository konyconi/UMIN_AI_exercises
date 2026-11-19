# Cvičení ke KMI/UMIN — Deep Learning

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
| 01 | Perceptron, lineární separabilita, baseline | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/notebooks/UMIN_DL_cv01_perceptron.ipynb) | [řešení](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/reseni/UMIN_DL_cv01_perceptron_reseni.ipynb) |
| 02 | Dopředné sítě a backpropagation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/notebooks/UMIN_DL_cv02_backprop.ipynb) | [řešení](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/reseni/UMIN_DL_cv02_backprop_reseni.ipynb) |
| 05b | První síť od dat po test (praktická přednáška) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/notebooks/UMIN_DL_pr05b_prvni_sit.ipynb) | [řešení](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/reseni/UMIN_DL_pr05b_prvni_sit_reseni.ipynb) |

Další cvičení se tu budou objevovat průběžně, vždy v týdnu
příslušné přednášky. Zatím jsou hotová cvičení 01–02 z třinácti.

## AKTI

| Notebook | Otevřít |
|---|---|
| Vícevrstvý perceptron (MLP) na MNIST | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/akti/AKTI_MLP.ipynb) |
| Šablona zápočtového úkolu (studie na Fashion-MNIST) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/blob/main/akti/AKTI_zapocet_sablona.ipynb) |

## Jak s notebooky pracovat

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
| 01–02 | `numpy`, `matplotlib`, `scikit-learn` | žádná (generují se) | do 1 minuty |
| 05b | + `torch`, `torchvision` | MNIST (12 MB, stáhne se sám) | asi 7 minut |

V Colabu je všechno předinstalované. Lokálně stačí `pip install -r
requirements.txt`.

## Údržba

Notebooky se generují ze skriptů ve složce [`zdroje/`](zdroje) — zadání
i řešení z jednoho zdroje, aby se nemohly rozejít. Jak generátor funguje, je
v [zdroje/README.md](zdroje/README.md). Tenhle soubor se generuje taky,
skriptem `zdroje/readme_gen.py`, aby tabulka odpovídala tomu, co je
opravdu zveřejněné — ruční úpravy v něm se při dalším zveřejnění ztratí.

## Rok 2024

Materiály z předchozího běhu předmětu, který měl jiný sylabus, jsou ve složce
[`2024/`](2024). Nejsou udržované a nenavazují na aktuální přednášky.
