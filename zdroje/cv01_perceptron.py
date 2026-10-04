# -*- coding: utf-8 -*-
"""Cviceni 01 - perceptron, linearni separabilita, baseline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbbuild import md, code, task, build

CELLS = [

md("""
**Cíle cvičení**

- Naimplementovat perceptron v `numpy` a natrénovat jej na logických funkcích.
- Zreprodukovat běh učení na funkci AND krok za krokem tak, jak je ve skriptech.
- Ukázat, že na XOR perceptron selže, a vysvětlit proč.
- Změřit, jak počet oprav závisí na kódování vstupů.
- Porovnat model s **baseline** na reálných datech a říct, co z toho plyne.

Navazuje na **přednášku 1** a kapitoly 0–1 skript.
"""),

md("""
## 0) Jak s notebookem pracovat

- Buňku spustíte **Shift+Enter**, buňky procházejte shora dolů.
- Místa označená `# TODO` jsou na vás. Dokud je nedoplníte, další buňky spadnou
  na `NotImplementedError` — to je záměr, ne chyba.
- Řešení je v samostatném notebooku ve složce `reseni/`. Podívejte se do něj
  až poté, co úlohu zkusíte.
"""),

md("""
## 1) Knihovny a data

Vystačíme si s `numpy` a `matplotlib`. Pevný seed nastavujeme proto, aby byl
běh zopakovatelný — vrátíme se k tomu v šestém cvičení.
"""),

code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
np.set_printoptions(suppress=True, linewidth=120)
"""),

md("""
Logické funkce zapíšeme v **bipolárním kódování**, tedy $x_i \\in \\{-1, +1\\}$
místo $\\{0, 1\\}$. Ke každému vstupu přidáme konstantní jedničku, aby se práh
choval jako další váha; tomu se říká **rozšířený vstup**
$\\overline{\\bm x} = \\langle 1, x_1, x_2 \\rangle$.
"""),

code("""
# ctyri body v {-1,+1}^2, prvni sloupec je konstantni jednicka (bias)
X = np.array([[1,  1,  1],
              [1,  1, -1],
              [1, -1,  1],
              [1, -1, -1]], dtype=float)

y_and = np.array([ 1, -1, -1, -1], dtype=float)
y_or  = np.array([ 1,  1,  1, -1], dtype=float)
y_xor = np.array([-1,  1,  1, -1], dtype=float)

def cele(v):
    \"\"\"Pomocna: pole -> n-tice bezneho typu int (hezci tisk).\"\"\"
    return tuple(int(a) for a in v)

print("vstup        AND  OR  XOR")
for i, radek in enumerate(X):
    print("%-12s %+3d %+3d %+4d" % (cele(radek[1:]), y_and[i], y_or[i], y_xor[i]))
"""),

md("""
## 2) Dopředný výpočet

Perceptron počítá $y = \\operatorname{sign}(\\bm w \\cdot \\overline{\\bm x})$.
Nulu budeme počítat jako $+1$ (na hranici se musíme nějak rozhodnout).

### Úkol 1
Doplňte funkci `predict`. Má vrátit pole $\\pm 1$ pro všechny řádky `X` naráz —
tedy bez `for` cyklu, jedním maticovým součinem.
"""),

task(
stub="""
def predict(w, X):
    \"\"\"Vrati pole +-1 pro kazdy radek X.\"\"\"
    # TODO: spocitejte skalarni souciny w . x pro vsechny radky
    #       a prevedte je na +-1 (nulu berte jako +1)
    raise NotImplementedError("doplnte predict")
""",
solution="""
def predict(w, X):
    \"\"\"Vrati pole +-1 pro kazdy radek X.\"\"\"
    s = X @ w
    return np.where(s >= 0, 1.0, -1.0)
"""),

code("""
# rychla kontrola: vahy (-1, 1, 1) maji podle skript pocitat AND
w_kniha = np.array([-1.0, 1.0, 1.0])
print("predikce:", predict(w_kniha, X).astype(int))
print("ocekavano:", y_and.astype(int))
assert np.array_equal(predict(w_kniha, X), y_and), "jeste to nesedi"
print("OK")
"""),

md("""
## 3) Učicí pravidlo

Trénovací množinu tvoří kladné vzory $P$ (cíl $+1$) a záporné $N$ (cíl $-1$).
Hledáme váhy, které je oddělí *ostře*: pro $\\overline{\\bm x} \\in P$ chceme
$\\bm w \\cdot \\overline{\\bm x} > 0$ a pro $\\overline{\\bm x} \\in N$
chceme $\\bm w \\cdot \\overline{\\bm x} < 0$. Odtud plyne pravidlo

$$\\bm w \\leftarrow \\bm w + \\overline{\\bm x}
\\quad \\text{pokud } \\overline{\\bm x}\\in P \\text{ a } \\bm w \\cdot \\overline{\\bm x} \\le 0,$$

$$\\bm w \\leftarrow \\bm w - \\overline{\\bm x}
\\quad \\text{pokud } \\overline{\\bm x}\\in N \\text{ a } \\bm w \\cdot \\overline{\\bm x} \\ge 0.$$

### Úkol 2
Doplňte `fit`. Vzory procházejte **cyklicky** v pořadí, v jakém jsou v `X`
(ne náhodně — chceme reprodukovatelnou tabulku). Funkce vrací výsledné váhy
a záznam o každém kroku.
"""),

task(
stub="""
def fit(X, y, max_kroku=50):
    \"\"\"Cyklicky prochazi vzory a opravuje vahy.

    Vraci (w, log), kde log je seznam radku
    (krok, x, cil, skalarni soucin, akce, nove w).
    \"\"\"
    w = np.zeros(X.shape[1])
    log = []
    for krok in range(1, max_kroku + 1):
        i = (krok - 1) % len(X)
        x, cil = X[i], y[i]
        s = float(w @ x)
        # TODO: podle znamenka s a cile rozhodnete, zda a jak opravit w.
        #       Do promenne akce ulozte "w+x", "w-x" nebo "---".
        raise NotImplementedError("doplnte opravu vah")
        log.append((krok, x.astype(int), int(cil), s, akce, w.astype(int).copy()))
        # zastavime, kdyz cely prubeh datasetem probehl bez opravy
        if len(log) >= len(X) and all(r[4] == "---" for r in log[-len(X):]):
            break
    return w, log
""",
solution="""
def fit(X, y, max_kroku=50):
    \"\"\"Cyklicky prochazi vzory a opravuje vahy.

    Vraci (w, log), kde log je seznam radku
    (krok, x, cil, skalarni soucin, akce, nove w).
    \"\"\"
    w = np.zeros(X.shape[1])
    log = []
    for krok in range(1, max_kroku + 1):
        i = (krok - 1) % len(X)
        x, cil = X[i], y[i]
        s = float(w @ x)
        if cil > 0 and s <= 0:
            w = w + x
            akce = "w+x"
        elif cil < 0 and s >= 0:
            w = w - x
            akce = "w-x"
        else:
            akce = "---"
        log.append((krok, x.astype(int), int(cil), s, akce, w.astype(int).copy()))
        # zastavime, kdyz cely prubeh datasetem probehl bez opravy
        if len(log) >= len(X) and all(r[4] == "---" for r in log[-len(X):]):
            break
    return w, log
"""),

md("""
Spustíme učení na AND. Tabulka by měla vyjít přesně tak, jak je ve skriptech
(kapitola 1, část *Ukázka celého běhu*): tři opravy a pak celý průchod
bez jediné změny vah.
"""),

code("""
w_and, log_and = fit(X, y_and)

print("krok  x             trida   w.x   akce   nove w")
for krok, x, cil, s, akce, w in log_and:
    print("%3d   %-13s %+2d   %5.1f  %-5s %s"
          % (krok, cele(x), cil, s, akce, cele(w)))
print()
print("naucene vahy:", cele(w_and))
"""),

md("""
Váhy $\\bm w = (-1, 1, 1)$ odpovídají prahu $\\theta = 1$ a rozhodovací funkci
$y = \\operatorname{sign}(x_1 + x_2 - 1)$, tedy hranici $x_1 + x_2 = 1$.
"""),

md("""
## 4) Jak vypadá hranice

Váhový vektor má **tři** složky, $\\bm w = (w_0, w_1, w_2)$, a hranice je
množina bodů, kde $w_0 + w_1x_1 + w_2x_2 = 0$. Dvojice $(w_1, w_2)$ je
**normála** — určuje jen směr hranice a to, na které straně je kladný výstup;
složka $w_0$ hranicí posouvá, aniž by změnila její směr.

Modrá šipka na obrázcích kreslí právě tu normálu. Dva stavy se proto mohou
lišit, i když šipka míří stejně — všimněte si prvního a třetího panelu.
"""),

code("""
def nakresli(w, X, y, ax, titul=""):
    ax.axhline(0, color="0.85", lw=.8)
    ax.axvline(0, color="0.85", lw=.8)
    for xi, yi in zip(X, y):
        ax.plot(xi[1], xi[2], "o", ms=8,
                color="tab:red" if yi > 0 else "tab:blue")
    b, w1, w2 = w
    t = np.linspace(-2, 2, 10)
    if abs(w2) > 1e-9:                     # w1*x + w2*y + b = 0
        ax.plot(t, -(w1 * t + b) / w2, color="tab:green", lw=2)
    elif abs(w1) > 1e-9:
        ax.axvline(-b / w1, color="tab:green", lw=2)
    n = np.hypot(w1, w2)
    if n > 1e-9:
        ax.arrow(0, 0, w1 / n * .6, w2 / n * .6, color="tab:blue",
                 head_width=.09, length_includes_head=True)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.set_title(titul, fontsize=9)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])

opravy = [r for r in log_and if r[4] != "---"]
fig, axes = plt.subplots(1, len(opravy), figsize=(3.2 * len(opravy), 3.2))
for ax, (krok, x, cil, s, akce, w) in zip(np.atleast_1d(axes), opravy):
    nakresli(w.astype(float), X, y_and, ax,
             "po %d. oprave, vzor %s\\nw = %s" % (krok, cele(x[1:]), cele(w)))
plt.tight_layout(); plt.show()
"""),

md("""
Každá oprava otočí normálu k chybně zařazenému vzoru (nebo od něj) a posune
hranici. Po třetí opravě leží červený bod jako jediný na kladné straně.
"""),

md("""
## 5) XOR

Teď totéž pro XOR. Nechte běžet dost kroků a podívejte se, co dělá váhový
vektor.
"""),

code("""
w_xor, log_xor = fit(X, y_xor, max_kroku=40)

pocet_oprav = sum(1 for r in log_xor if r[4] != "---")
print("kroku: %d, z toho oprav: %d" % (len(log_xor), pocet_oprav))
print()

stavy = [cele(r[5]) for r in log_xor]
print("vahy po jednotlivych krocich:")
for i in range(0, 12):
    print("  krok %2d -> %s" % (i + 1, stavy[i]))

# najdeme nejblizsi predchozi krok se stejnym stavem vah
predchozi = max(i for i in range(11) if stavy[i] == stavy[11])
print()
print("stav po kroku 12 nastal naposledy uz po kroku %d -> perioda %d kroku"
      % (predchozi + 1, 11 - predchozi))
"""),

md("""
Algoritmus se nezastaví. Váhy se po několika krocích vrátí do stavu, ve kterém
už byly, a celý cyklus se opakuje — žádná přímka čtyři body XOR neoddělí. Věta
o konvergenci tu mlčí, protože její předpoklad (lineární separabilita)
neplatí.

Podívejme se na dva různé okamžiky téhož běhu: hranice se hýbe, ale nikdy
neoddělí červené body od modrých.
"""),

code("""
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
for ax, krok in zip(axes, (2, 4)):
    w = log_xor[krok - 1][5].astype(float)
    nakresli(w, X, y_xor, ax, "XOR po %d. kroku, w = %s" % (krok, cele(w)))
plt.tight_layout(); plt.show()
"""),

md("""
> **Otázka k zamyšlení.** Kdybyste algoritmus nechali běžet milion kroků,
> poznali byste z průběhu vah, že úloha nemá řešení? Co konkrétně byste museli
> sledovat, abyste to poznali automaticky?
"""),

md("""
## 6) Na kódování záleží

Ve skriptech je poznámka, že táž úloha v **binárním** kódování
($x_i \\in \\{0,1\\}$) potřebuje výrazně víc oprav než v bipolárním.

### Úkol 3
Sestavte binární variantu dat pro AND, spusťte na ní `fit` a porovnejte počet
oprav s bipolární variantou. Zamyslete se, proč je rozdíl tak velký.
"""),

task(
stub="""
# TODO: sestavte X_bin (prvni sloupec je stale jednicka, dalsi dva jsou 0/1)
#       a cile y_and_bin v kodovani +-1, pak spustte fit a spocitejte opravy.
raise NotImplementedError("doplnte binarni variantu")
""",
solution="""
X_bin = np.array([[1, 1, 1],
                  [1, 1, 0],
                  [1, 0, 1],
                  [1, 0, 0]], dtype=float)
y_and_bin = np.array([1, -1, -1, -1], dtype=float)

w_bin, log_bin = fit(X_bin, y_and_bin, max_kroku=200)
opravy_bip = sum(1 for r in log_and if r[4] != "---")
opravy_bin = sum(1 for r in log_bin if r[4] != "---")

print("bipolarni kodovani: %2d oprav, vahy %s" % (opravy_bip, cele(w_and)))
print("binarni kodovani:   %2d oprav, vahy %s" % (opravy_bin, cele(w_bin)))
print("predikce binarni:  ", cele(predict(w_bin, X_bin)))
"""),

md("""
Vzor $(0,0)$ nenese v prvních dvou souřadnicích žádnou informaci, takže oprava
provedená tímto vzorem mění pouze práh. Bipolární kódování dává každému vzoru
plnou váhu — proto konverguje rychleji. Je to první ukázka toho, že
**reprezentace vstupu je součást modelu**; systematicky se jí věnuje třetí
přednáška.

Rozdíl není náhoda jednoho pořadí: projdete-li všech 24 pořadí čtyř vzorů,
binární kódování potřebuje 18 až 22 oprav, bipolární nejvýše tři.
"""),

md("""
## 7) Baseline: proti čemu se model měří

Přesnost sama o sobě nic neříká, dokud nevíme, co dá triviální model. Vezmeme
reálná data a porovnáme perceptron s **majoritní baseline**, která vždy hádá
nejčastější třídu.

### Úkol 4
Doplňte výpočet obou přesností na testovací části. Perceptron trénujte na
standardizovaných příznacích (proč, uvidíme v šestém cvičení).
"""),

code("""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
Xr, yr = data.data, np.where(data.target == 1, 1.0, -1.0)
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3,
                                      random_state=0, stratify=yr)

# standardizace: prumer a smerodatnou odchylku bereme JEN z trenovaci casti
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
Xtr_s = np.hstack([np.ones((len(Xtr), 1)), (Xtr - mu) / sd])
Xte_s = np.hstack([np.ones((len(Xte), 1)), (Xte - mu) / sd])

print("trenovaci:", Xtr_s.shape, " testovaci:", Xte_s.shape)
print("zastoupeni trid v trenovaci casti:",
      {int(t): int((ytr == t).sum()) for t in (-1, 1)})
"""),

task(
stub="""
w_real, _ = fit(Xtr_s, ytr, max_kroku=20 * len(Xtr_s))

# TODO: spocitejte
#   acc_model    - podil spravnych predikci perceptronu na testovaci casti
#   trida_vetsi  - cetnejsi trida v TRENOVACI casti (+1 nebo -1)
#   acc_baseline - presnost modelu, ktery vzdy hada trida_vetsi
raise NotImplementedError("doplnte vypocet presnosti")

print("perceptron: %.3f" % acc_model)
print("baseline:   %.3f" % acc_baseline)
""",
solution="""
w_real, _ = fit(Xtr_s, ytr, max_kroku=20 * len(Xtr_s))

acc_model = float((predict(w_real, Xte_s) == yte).mean())
trida_vetsi = 1.0 if (ytr == 1).sum() >= (ytr == -1).sum() else -1.0
acc_baseline = float((yte == trida_vetsi).mean())

print("perceptron: %.3f" % acc_model)
print("baseline:   %.3f" % acc_baseline)
"""),

md("""
Baseline je tu kolem $0{,}63$, protože jedna třída je početnější. Číslo
u modelu má smysl číst jen jako **rozdíl proti ní**. Kdyby model dosáhl
$0{,}63$, nenaučil se nic, i když to na první pohled vypadá slušně.

> **Otázka k zamyšlení.** Perceptron se zastaví, jen když jsou data lineárně
> separabilní. Co se stane, pokud separabilní nejsou, a proč je proto vhodné
> vracet nejlepší dosud viděné váhy místo posledních?
"""),

md("""
## 8) Bonus: XOR vyřeší dvouvrstvá síť

Že XOR není beznadějný, ukáže hotový model ze `scikit-learn`. Jak takovou síť
naučit, je náplní druhého cvičení.
"""),

code("""
from sklearn.neural_network import MLPClassifier

cil = (y_xor > 0).astype(int)
for skryte in [(2,), (4,)]:
    uspech = []
    for seed in range(4):
        mlp = MLPClassifier(hidden_layer_sizes=skryte, activation="tanh",
                            solver="lbfgs", alpha=1e-6, max_iter=5000,
                            random_state=seed).fit(X[:, 1:], cil)
        uspech.append(bool(np.array_equal(mlp.predict(X[:, 1:]), cil)))
    print("skryta vrstva %-5s -> XOR vyresen pro seedy: %s"
          % (skryte, uspech))
"""),

md("""
Dvě skryté jednotky teoreticky stačí, ale učení do správného řešení netrefí
pokaždé — záleží na náhodné inicializaci. Se čtyřmi jednotkami už vyjde vždy.
Že se optimalizace může zaseknout jinde, než bychom chtěli, je téma čtvrté
přednášky.
"""),

md("""
## Co si z cvičení odnést

- Perceptron opravuje váhy jen podle chybně zařazených vzorů; každá oprava
  otočí normálu hranice.
- Konvergence je zaručena pouze pro lineárně separabilní data. XOR takový není.
- Kódování vstupů mění počet kroků, i když je úloha stejná.
- Přesnost bez baseline není informace.

**Na doma.** Zkuste do funkce `fit` přidat učicí krok $\\eta$ (tedy
$\\bm w \\leftarrow \\bm w \\pm \\eta\\,\\overline{\\bm x}$) a ověřte, že
u perceptronu s nulovou inicializací nemá na výsledné rozhodování vliv.
Proč tomu tak je?
"""),
]

build(__file__, cislo=1, nazev="Perceptron, lineární separabilita a baseline",
      cells=CELLS)
