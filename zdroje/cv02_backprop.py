# -*- coding: utf-8 -*-
"""Cviceni 02 - dopredne site a backpropagation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbbuild import md, code, task, build

CELLS = [

md("""
**Cíle cvičení**

- Napsat dopředný a zpětný průchod sítě jen s `numpy`.
- Zreprodukovat ruční příklad z přednášky (síť 2–2–1) a shodnout se na číslech.
- Ověřit gradient **numerickou derivací** a vědět, jaká tolerance je v pořádku.
- Natrénovat síť na úloze, kterou perceptron nezvládl.
- Vidět, že hloubka bez nelinearity nic nepřidá.

Navazuje na **přednášku 2** a kapitolu 2 skript.
"""),

md("""
## 1) Příprava
"""),

code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
np.set_printoptions(suppress=True, precision=6, linewidth=120)

def sigmoida(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoida_derivace(a):
    \"\"\"Derivace vyjadrena pomoci uz spocitane aktivace: sigma' = a (1 - a).\"\"\"
    return a * (1.0 - a)
"""),

md("""
## 2) Síť 2–2–1 z přednášky

Vezmeme přesně tu síť, která je na slajdech: dva vstupy, dva skryté neurony
a jeden výstup, všude logistická sigmoida, ztráta $E=\\tfrac12(o-t)^2$.

| | hodnota |
|---|---|
| $w_{11}$ (x₁→h₁), $w_{21}$ (x₂→h₁) | $0{,}5$, $-0{,}4$ |
| $w_{12}$ (x₁→h₂), $w_{22}$ (x₂→h₂) | $0{,}3$, $0{,}8$ |
| $b_1$, $b_2$ | $0{,}1$, $-0{,}2$ |
| $v_1$, $v_2$, $b_o$ | $0{,}6$, $-0{,}5$, $0{,}2$ |

Vstup je $\\bm x = (1, 0)$, cíl $t = 1$, učicí krok $\\gamma = 0{,}5$.
"""),

code("""
# radek matice W1 = jeden skryty neuron, sloupce = vstupy
W1 = np.array([[0.5, -0.4],     # h1
               [0.3,  0.8]])    # h2
b1 = np.array([0.1, -0.2])
W2 = np.array([[0.6, -0.5]])    # jediny vystupni neuron
b2 = np.array([0.2])

x = np.array([1.0, 0.0])
t = np.array([1.0])
gamma = 0.5
"""),

md("""
### Úkol 1: dopředný průchod

Doplňte funkci `dopredu`. Vrací slovník s mezivýsledky — ty budeme potřebovat
při zpětném průchodu, takže je nezahazujte.
"""),

task(
stub="""
def dopredu(x, W1, b1, W2, b2):
    \"\"\"Vraci slovnik s net1, h, net_o, o.\"\"\"
    # TODO: net1 = W1 x + b1, h = sigma(net1),
    #       net_o = W2 h + b2,  o = sigma(net_o)
    raise NotImplementedError("doplnte dopredny pruchod")
""",
solution="""
def dopredu(x, W1, b1, W2, b2):
    \"\"\"Vraci slovnik s net1, h, net_o, o.\"\"\"
    net1 = W1 @ x + b1
    h = sigmoida(net1)
    net_o = W2 @ h + b2
    o = sigmoida(net_o)
    return {"net1": net1, "h": h, "net_o": net_o, "o": o}
"""),

code("""
stav = dopredu(x, W1, b1, W2, b2)
E = 0.5 * float(np.sum((stav["o"] - t) ** 2))

print("net1 =", stav["net1"], "  ocekavano [0.6, 0.1]")
print("h    =", stav["h"],    "  ocekavano [0.6457, 0.5250]")
print("net_o=", stav["net_o"], " ocekavano [0.3249]")
print("o    =", stav["o"],    "  ocekavano [0.5805]")
print("E    = %.4f          ocekavano 0.0880" % E)
"""),

md("""
## 3) Zpětný průchod

Pro ztrátu $E=\\tfrac12(o-t)^2$ a sigmoidu na výstupu je

$$\\delta_o = (o - t)\\,o(1-o),$$

a pro skrytou vrstvu

$$\\delta_j = h_j(1-h_j)\\, v_j\\, \\delta_o.$$

Gradienty vah jsou pak vnější součiny $\\delta$ a vstupu dané vrstvy.

### Úkol 2
Doplňte `zpet`. Vrací gradienty ve stejném tvaru, jaký mají samotné parametry.
"""),

task(
stub="""
def zpet(x, t, stav, W2):
    \"\"\"Vraci (dW1, db1, dW2, db2).\"\"\"
    h, o = stav["h"], stav["o"]
    # TODO: spocitejte delta_o, delta_h a z nich gradienty.
    #       Vnejsi soucin: np.outer(delta, vstup)
    raise NotImplementedError("doplnte zpetny pruchod")
""",
solution="""
def zpet(x, t, stav, W2):
    \"\"\"Vraci (dW1, db1, dW2, db2).\"\"\"
    h, o = stav["h"], stav["o"]
    delta_o = (o - t) * sigmoida_derivace(o)
    dW2 = np.outer(delta_o, h)
    db2 = delta_o
    delta_h = (W2.T @ delta_o) * sigmoida_derivace(h)
    dW1 = np.outer(delta_h, x)
    db1 = delta_h
    return dW1, db1, dW2, db2
"""),

code("""
dW1, db1, dW2, db2 = zpet(x, t, stav, W2)

print("delta_o = %.4f   ocekavano -0.1022" % db2[0])
print("delta_h =", db1, "  ocekavano [-0.0140, 0.0127]")
print()
print("dE/dv1 = %.4f  ocekavano -0.0660" % dW2[0, 0])
print("dE/dv2 = %.4f  ocekavano -0.0536" % dW2[0, 1])
"""),

md("""
Krok učení se dělá až po spočítání **všech** zpětně šířených chyb. Kdybychom
váhy opravovali průběžně, další `delta` už by se počítala z jiné sítě a
aktualizace by neodpovídala zápornému gradientu.
"""),

code("""
W1n, b1n = W1 - gamma * dW1, b1 - gamma * db1
W2n, b2n = W2 - gamma * dW2, b2 - gamma * db2

print("vaha        pred      po")
for jm, pred, po in [("v1", W2[0, 0], W2n[0, 0]), ("v2", W2[0, 1], W2n[0, 1]),
                     ("b_o", b2[0], b2n[0]),
                     ("w11", W1[0, 0], W1n[0, 0]), ("w21", W1[0, 1], W1n[0, 1]),
                     ("w12", W1[1, 0], W1n[1, 0]), ("w22", W1[1, 1], W1n[1, 1])]:
    print("%-4s   %8.4f  %8.4f" % (jm, pred, po))

E_nova = 0.5 * float(np.sum((dopredu(x, W1n, b1n, W2n, b2n)["o"] - t) ** 2))
print()
print("E pred krokem: %.4f" % E)
print("E po kroku:    %.4f   (na slajdu 0.0791)" % E_nova)
"""),

md("""
Všimněte si, že $w_{21}$ a $w_{22}$ se nezměnily. Obě vedou od vstupu
$x_2 = 0$, takže jejich gradient je nulový — váha, jejíž vstup je nula, se
z tohoto vzoru nemá jak naučit.
"""),

md("""
## 4) Numerická kontrola gradientu

Analytický gradient porovnáme s centrální diferencí

$$\\frac{\\partial E}{\\partial \\theta_i} \\approx
\\frac{E(\\theta + \\varepsilon e_i) - E(\\theta - \\varepsilon e_i)}{2\\varepsilon}.$$

Porovnává se **relativní** rozdíl, ne absolutní — u velkých gradientů by
absolutní rozdíl vypadal špatně i pro správný kód.

### Úkol 3
Doplňte výpočet numerického gradientu pro matici `W1`.
"""),

task(
stub="""
def ztrata(W1, b1, W2, b2):
    return 0.5 * float(np.sum((dopredu(x, W1, b1, W2, b2)["o"] - t) ** 2))

def numericky_gradient_W1(W1, b1, W2, b2, eps=1e-5):
    g = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            # TODO: vychylte W1[i, j] o +eps a o -eps, spoctete obe ztraty
            #       a z nich centralni diferenci
            raise NotImplementedError("doplnte numericky gradient")
    return g
""",
solution="""
def ztrata(W1, b1, W2, b2):
    return 0.5 * float(np.sum((dopredu(x, W1, b1, W2, b2)["o"] - t) ** 2))

def numericky_gradient_W1(W1, b1, W2, b2, eps=1e-5):
    g = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            plus = W1.copy();  plus[i, j] += eps
            minus = W1.copy(); minus[i, j] -= eps
            g[i, j] = (ztrata(plus, b1, W2, b2)
                       - ztrata(minus, b1, W2, b2)) / (2 * eps)
    return g
"""),

code("""
def relativni_rozdil(a, b):
    meritko = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
    return float(np.max(np.abs(a - b) / meritko))

g_num = numericky_gradient_W1(W1, b1, W2, b2)
print("analyticky:"); print(dW1)
print("numericky:");  print(g_num)
print()
rel = relativni_rozdil(dW1, g_num)
print("maximalni relativni rozdil: %.2e" % rel)
print("verdikt:", "OK" if rel < 1e-7 else "podezrele")
"""),

md("""
Hodnota pod $10^{-7}$ je v pořádku, nad $10^{-4}$ znamená chybu. Test má ale
svoje podmínky, bez nichž neprojde ani bezchybný kód: počítat v `float64`,
volit rozumné $\\varepsilon$ a vypnout náhodné prvky (dropout).

### Úkol 4
Vyzkoušejte, jak relativní rozdíl závisí na $\\varepsilon$. Projděte hodnoty
$10^{-1}$ až $10^{-11}$ a najděte, kde je test nejpřesnější.
"""),

task(
stub="""
# TODO: pro kazde eps spoctete maximalni relativni rozdil proti dW1
#       a vypiste tabulku
raise NotImplementedError("doplnte prubeh presnosti podle eps")
""",
solution="""
print("   eps      rel. rozdil")
for k in range(1, 12):
    eps = 10.0 ** (-k)
    g = numericky_gradient_W1(W1, b1, W2, b2, eps=eps)
    print("  1e-%-4d   %.2e" % (k, relativni_rozdil(dW1, g)))
"""),

md("""
Průběh má tvar písmene U: vlevo dominuje chyba aproximace (diference je příliš
hrubá), vpravo zaokrouhlovací chyba (odečítáme dvě téměř stejná čísla).
Optimum leží kolem $10^{-5}$ až $10^{-6}$.

> **Otázka k zamyšlení.** Proč se u ReLU nesmíme divit, když test selže přesně
> v nule?
"""),

md("""
## 5) Trénink na úloze, kterou perceptron nezvládl

Rozšíříme kód na dávky a naučíme síť dataset **two moons** — dvě propletené
množiny, které přímka neoddělí.
"""),

code("""
from sklearn.datasets import make_moons

Xd, yd = make_moons(n_samples=400, noise=0.20, random_state=0)
yd = yd.astype(float).reshape(-1, 1)

plt.figure(figsize=(4.2, 3.4))
plt.scatter(Xd[:, 0], Xd[:, 1], c=yd.ravel(), cmap="coolwarm", s=14,
            edgecolors="k", linewidths=.3)
plt.title("two moons"); plt.tight_layout(); plt.show()
"""),

code("""
class Sit:
    \"\"\"Dopredna sit 2-h-1, sigmoida, ztrata = prumerna kvadraticka chyba.\"\"\"

    def __init__(self, n_vstup, n_skryte, rng):
        # skalovani podle poctu vstupu (podrobne v sestem cviceni)
        self.W1 = rng.normal(0, np.sqrt(1.0 / n_vstup), (n_skryte, n_vstup))
        self.b1 = np.zeros(n_skryte)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_skryte), (1, n_skryte))
        self.b2 = np.zeros(1)

    def dopredu(self, X):
        self.H = sigmoida(X @ self.W1.T + self.b1)
        self.O = sigmoida(self.H @ self.W2.T + self.b2)
        return self.O

    def krok(self, X, y, eta):
        n = len(X)
        O = self.dopredu(X)
        Do = (O - y) * sigmoida_derivace(O)              # n x 1
        Dh = (Do @ self.W2) * sigmoida_derivace(self.H)  # n x h
        self.W2 -= eta * (Do.T @ self.H) / n
        self.b2 -= eta * Do.mean(0)
        self.W1 -= eta * (Dh.T @ X) / n
        self.b1 -= eta * Dh.mean(0)
        return 0.5 * float(np.mean((O - y) ** 2))
"""),

code("""
sit = Sit(2, 16, np.random.default_rng(1))
historie = [sit.krok(Xd, yd, eta=3.0) for _ in range(6000)]

presnost = float(((sit.dopredu(Xd) > 0.5) == (yd > 0.5)).mean())
print("ztrata na zacatku: %.4f" % historie[0])
print("ztrata na konci:   %.4f" % historie[-1])
print("presnost:          %.3f" % presnost)

plt.figure(figsize=(4.2, 3))
plt.plot(historie); plt.xlabel("krok"); plt.ylabel("ztrata")
plt.title("prubeh uceni"); plt.tight_layout(); plt.show()
"""),

code("""
def nakresli_hranici(model, X, y, titul):
    h = .02
    xx, yy = np.meshgrid(np.arange(X[:, 0].min() - .5, X[:, 0].max() + .5, h),
                         np.arange(X[:, 1].min() - .5, X[:, 1].max() + .5, h))
    Z = model.dopredu(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(4.6, 3.6))
    plt.contourf(xx, yy, Z, levels=[Z.min() - 1e-9, .5, Z.max() + 1e-9],
                 alpha=.3, cmap="coolwarm")
    plt.contour(xx, yy, Z, levels=[.5], colors="k", linewidths=1)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", s=14,
                edgecolors="k", linewidths=.3)
    plt.title(titul); plt.tight_layout(); plt.show()

nakresli_hranici(sit, Xd, yd, "naucena hranice")
"""),

md("""
## 6) Proč je potřeba nelinearita

Složením lineárních vrstev vznikne zase jen lineární zobrazení. Ověříme to
experimentem.

### Úkol 5
Doplňte třídu `SitLinearni`, ve které je aktivace **identita**, a natrénujte ji
na týchž datech. Porovnejte přesnost se sítí se sigmoidou.
"""),

task(
stub="""
class SitLinearni(Sit):
    def dopredu(self, X):
        # TODO: zadna nelinearita
        raise NotImplementedError("doplnte dopredny pruchod bez aktivace")

    def krok(self, X, y, eta):
        # TODO: totez co v Sit.krok, ale derivace identity je 1
        raise NotImplementedError("doplnte krok bez aktivace")

# lin = SitLinearni(2, 16, np.random.default_rng(1))
# for _ in range(6000):
#     lin.krok(Xd, yd, eta=0.05)
""",
solution="""
class SitLinearni(Sit):
    def dopredu(self, X):
        self.H = X @ self.W1.T + self.b1          # identita misto sigmoidy
        self.O = self.H @ self.W2.T + self.b2
        return self.O

    def krok(self, X, y, eta):
        n = len(X)
        O = self.dopredu(X)
        Do = (O - y)                              # derivace identity je 1
        Dh = Do @ self.W2
        self.W2 -= eta * (Do.T @ self.H) / n
        self.b2 -= eta * Do.mean(0)
        self.W1 -= eta * (Dh.T @ X) / n
        self.b1 -= eta * Dh.mean(0)
        return 0.5 * float(np.mean((O - y) ** 2))

lin = SitLinearni(2, 16, np.random.default_rng(1))
for _ in range(6000):
    lin.krok(Xd, yd, eta=0.05)

presnost_lin = float(((lin.dopredu(Xd) > 0.5) == (yd > 0.5)).mean())
print("sit se sigmoidou: %.3f" % presnost)
print("sit s identitou:  %.3f" % presnost_lin)
nakresli_hranici(lin, Xd, yd, "linearni sit: hranice je primka")
"""),

md("""
Lineární síť se dvěma vrstvami dopadne stejně jako jediná lineární vrstva —
šestnáct skrytých jednotek jí nepomůže, protože složením lineárních zobrazení
vznikne opět lineární zobrazení. Hloubka má smysl jedině s nelinearitou.

> **Otázka k zamyšlení.** Věta o univerzální aproximaci říká, že jedna skrytá
> vrstva stačí na libovolnou spojitou funkci na kompaktu. Proč tedy stavíme
> hluboké sítě?
"""),

md("""
## Co si z cvičení odnést

- Zpětný průchod je řetízkové pravidlo aplikované po vrstvách: `delta` se šíří
  zpět a gradienty vah jsou vnější součiny.
- Váha, jejíž vstup je nulový, se z daného vzoru nenaučí nic.
- Numerická kontrola gradientu je levná a odhalí většinu chyb v ručně psané
  vrstvě, má ale svoje podmínky.
- Bez nelinearity je hloubka k ničemu.

**Na doma.** Nahraďte kvadratickou ztrátu binární cross-entropy
$E = -[t\\log o + (1-t)\\log(1-o)]$ a odvoďte, že $\\delta_o$ se zjednoduší na
$o - t$. Ověřte to numerickou kontrolou gradientu — vrátíme se k tomu ve
třetím cvičení.
"""),
]

build(__file__, cislo=2,
      nazev="Dopředné sítě a backpropagation", cells=CELLS)
