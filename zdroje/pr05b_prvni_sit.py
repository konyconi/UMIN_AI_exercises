# -*- coding: utf-8 -*-
"""Prakticka prednaska 05b - prvni sit v PyTorch od dat po test.

Notebook se pousti zive u projektoru behem prednasky (deck
UMIN_DL_05b_prakticka_sit.tex). Cisla v decku pochazeji z behu
prednasky/dl05b_experimenty.py, ktery pouziva tytez funkce a tytez seedy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbbuild import md, code, task, build

CELLS = [

md("""
**Cíle**

- Projít celou cestu od surových dat k vyhodnocenému modelu v PyTorch.
- U každého řádku trénovací smyčky vědět, ze které přednášky pochází.
- Vidět na vlastní oči, co se stane, když se pravidla z přednášek 1–5 poruší.

Navazuje na **přednášky 1–5** (perceptron, backpropagation, ztrátové funkce,
optimalizace, regularizace). Deck k přednášce je *5b. První síť od dat po test*.
"""),

md("""
## 1) Příprava

Všechno běží na CPU. Jedna epocha malé sítě na MNIST trvá pár sekund.
"""),

code("""
import math, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("torch", torch.__version__)
"""),

md("""
## 2) Data

MNIST: 60 000 trénovacích a 10 000 testovacích obrázků číslic 28×28 v odstínech
šedi. Z trénovacích odložíme 10 000 na **validaci** (přednáška 3, dělení dat).
Testovací množinu použijeme jedinkrát, až na úplném konci.
"""),

code("""
def nacti_mnist(slozka="./data"):
    try:
        from torchvision import datasets
        a = datasets.MNIST(slozka, train=True, download=True)
        b = datasets.MNIST(slozka, train=False, download=True)
        return a.data, a.targets, b.data, b.targets
    except ImportError:
        # bez torchvision: soubory ve formatu IDX z data/MNIST/raw
        def idx(nazev):
            with open(os.path.join(slozka, "MNIST", "raw", nazev), "rb") as f:
                d = f.read()
            n = d[3]
            tvar = [int.from_bytes(d[4 + 4*i:8 + 4*i], "big") for i in range(n)]
            return torch.tensor(np.frombuffer(d, np.uint8, offset=4 + 4*n).reshape(tvar))
        return (idx("train-images-idx3-ubyte"), idx("train-labels-idx1-ubyte").long(),
                idx("t10k-images-idx3-ubyte"), idx("t10k-labels-idx1-ubyte").long())

X_all, y_all, X_test_raw, y_test = nacti_mnist()
print(X_all.shape, X_all.dtype, y_all[:10])

fig, axes = plt.subplots(1, 10, figsize=(9, 1.2))
for ax, i in zip(axes, range(10)):
    ax.imshow(X_all[i], cmap="gray_r"); ax.axis("off"); ax.set_title(int(y_all[i]))
plt.show()
"""),

code("""
perm = torch.randperm(60000, generator=torch.Generator().manual_seed(0))
tr, va = perm[:50000], perm[50000:]

Xtr_raw = X_all[tr].reshape(-1, 784).float();  ytr = y_all[tr]
Xva_raw = X_all[va].reshape(-1, 784).float();  yva = y_all[va]
Xte_raw = X_test_raw.reshape(-1, 784).float()

# statistiky POUZE z trenovaci casti (jinak informace z validace prosakuje)
mu = (Xtr_raw / 255).mean().item()
sd = (Xtr_raw / 255).std().item()
print("prumer %.4f, smerodatna odchylka %.4f" % (mu, sd))

def priprav(X, rezim="std"):
    if rezim == "raw":
        return X.clone()            # 0..255
    if rezim == "01":
        return X / 255              # 0..1
    return (X / 255 - mu) / sd      # standardizace

Xtr, Xva, Xte = priprav(Xtr_raw), priprav(Xva_raw), priprav(Xte_raw)
print(Xtr.shape, ytr.shape, "| cetnosti trid:", torch.bincount(ytr).tolist())
"""),

md("""
Obrázek jsme **zploštili** na vektor 784 čísel. Plně propojená síť tím ztrácí
informaci o sousedství pixelů (přednáška 3, *Obraz: proč jej nezploštit*);
vědomě to tu přijímáme a vrátíme se k tomu u konvolučních sítí.

**Baseline.** Než začneme, ať víme, co je „dobrý“ výsledek.
"""),

code("""
nejcastejsi = torch.bincount(ytr).argmax().item()
print("vzdy trida %d: presnost na validaci %.4f"
      % (nejcastejsi, (yva == nejcastejsi).float().mean()))
"""),

md("""
## 3) Model

`nn.Linear(n, m)` je přesně vrstva z přednášky 2: $\\bm o = \\bm W\\bm x + \\bm b$.
Síť poskládáme ze dvou takových vrstev a ReLU mezi nimi.
"""),

code("""
def mlp(sirky=(784, 128, 10), akt=nn.ReLU, softmax_navic=False, dropout=0.0):
    vrstvy = []
    for i in range(len(sirky) - 1):
        vrstvy.append(nn.Linear(sirky[i], sirky[i + 1]))
        if i < len(sirky) - 2 and akt is not None:
            vrstvy.append(akt())
            if dropout > 0:
                vrstvy.append(nn.Dropout(dropout))
    if softmax_navic:
        vrstvy.append(nn.Softmax(dim=1))
    return nn.Sequential(*vrstvy)

torch.manual_seed(0)
model = mlp()
print(model)
for jmeno, p in model.named_parameters():
    print("  %-8s %-12s %6d" % (jmeno, tuple(p.shape), p.numel()))
print("celkem:", sum(p.numel() for p in model.parameters()))
"""),

code("""
xb = Xtr[:128]
h = model[0](xb);  print("po 1. vrstve:", tuple(h.shape))
h = model[1](h);   print("po ReLU:     ", tuple(h.shape), "zapornych:", int((h < 0).sum()))
z = model[2](h);   print("logity:      ", tuple(z.shape))
"""),

md("""
## 4) Ztráta

`nn.CrossEntropyLoss` dostává **logity** a softmax si udělá sama (přednáška 3,
*Proč se softmax a křížová entropie párují*). Cílem je index třídy, ne one-hot.

**Kontrola před trénováním:** u neučené sítě je rozdělení přes 10 tříd zhruba
rovnoměrné, takže ztráta má vyjít okolo $\\ln 10 \\approx 2{,}303$.
"""),

code("""
loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def vyhodnot(model, X, y):
    model.eval()
    out = model(X)
    return loss_fn(out, y).item(), (out.argmax(1) == y).float().mean().item()

print("ztrata pred trenovanim: %.4f   ln 10 = %.4f" % (vyhodnot(model, Xva, yva)[0], math.log(10)))
"""),

md("""
## 5) Gradient: autograd místo ruční backpropagace

Ověříme na síti 2–2–1 ze **druhé přednášky** (snímek *Demonstrace: jeden krok
v síti 2–2–1*). Autograd musí dát tytéž gradienty, jaké jsme tam počítali
ručně.
"""),

code("""
W1 = torch.tensor([[0.5, -0.4], [0.3, 0.8]], dtype=torch.float64, requires_grad=True)
b1 = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
v  = torch.tensor([0.6, -0.5], dtype=torch.float64, requires_grad=True)
bo = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
x, t = torch.tensor([1.0, 0.0], dtype=torch.float64), 1.0

h = torch.sigmoid(W1 @ x + b1)
o = torch.sigmoid(v @ h + bo)
E = 0.5 * (o - t) ** 2
E.backward()                       # zpetny pruchod

print("h =", h.detach().numpy().round(4), " o = %.4f  E = %.4f" % (o.item(), E.item()))
print("dE/dv  =", v.grad.numpy().round(4))
print("dE/dW1 =", W1.grad.numpy().round(4).tolist())
"""),

md("""
**Gradienty se v `.grad` sčítají.** Přesně jako v uzlu výpočetního grafu
(přednáška 2, *Proč se v uzlu sčítá*): každé `backward()` k `.grad` přičte.
Proto se před každým krokem volá `opt.zero_grad()`.
"""),

code("""
w = torch.tensor(3.0, requires_grad=True)
(w ** 2).backward(); print("po 1. backward:", w.grad.item())
(w ** 2).backward(); print("po 2. backward:", w.grad.item())
"""),

md("""
## 6) Trénovací smyčka

Celé učení je tento cyklus. Každý řádek odpovídá něčemu z přednášek:
mini-batch a míchání (4), dopředný průchod (2), ztráta (3), zpětný průchod (2),
krok optimalizátoru (4).
"""),

code("""
def trenuj(model, opt, X=None, y=None, epoch=5, bs=128, seed=0, zero_grad=True,
           Xv=None, yv=None, tisk=True):
    X = Xtr if X is None else X;  y = ytr if y is None else y
    Xv = Xva if Xv is None else Xv;  yv = yva if yv is None else yv
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=bs, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    ztraty = []
    for ep in range(1, epoch + 1):
        model.train()
        for xb, yb in loader:
            if zero_grad:
                opt.zero_grad()            # vynulovat nasbirane gradienty
            loss = loss_fn(model(xb), yb)  # dopredny pruchod + ztrata
            loss.backward()                # zpetny pruchod
            opt.step()                     # krok optimalizatoru
            ztraty.append(loss.item())
        l, a = vyhodnot(model, Xv, yv)
        if tisk:
            print("epocha %d: val ztrata %.4f, val presnost %.4f" % (ep, l, a))
        if not math.isfinite(ztraty[-1]):
            print("ztrata je", ztraty[-1], "-> konec"); break
    return ztraty

torch.manual_seed(0)
model = mlp()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
t0 = time.time()
z_hlavni = trenuj(model, opt)
print("%.1f s" % (time.time() - t0))
"""),

md("""
### Tři úmyslné chyby

Každá odpovídá jednomu pravidlu z přednášek. Než buňku spustíte, tipněte si,
co se stane.
"""),

code("""
# (a) bez nelinearity: dve linearni vrstvy jsou zase jen linearni zobrazeni
torch.manual_seed(0); m = mlp(akt=None)
trenuj(m, torch.optim.SGD(m.parameters(), lr=0.1), tisk=False)
print("bez ReLU:      val presnost %.4f" % vyhodnot(m, Xva, yva)[1])

# (b) softmax navic pred CrossEntropyLoss
torch.manual_seed(0); m = mlp(softmax_navic=True)
trenuj(m, torch.optim.SGD(m.parameters(), lr=0.1), tisk=False)
print("softmax 2x:    val presnost %.4f, ztrata %.4f" % (vyhodnot(m, Xva, yva)[1], vyhodnot(m, Xva, yva)[0]))

# (c) zapomenute zero_grad
torch.manual_seed(0); m = mlp()
z = trenuj(m, torch.optim.SGD(m.parameters(), lr=0.1), epoch=1, zero_grad=False, tisk=False)
print("bez zero_grad: ztrata po 10, 50, 200 krocich:", [round(z[i], 3) for i in (9, 49, 199)])
"""),

md("""
## 7) Optimalizace

Velikost kroku (přednáška 4, *Vliv velikosti kroku*). Stejná síť, stejný seed,
jen jiné `lr`.
"""),

code("""
plt.figure(figsize=(6, 3.5))
for lr in (0.001, 0.01, 0.1, 1.0, 3.0):
    torch.manual_seed(0); m = mlp()
    z = trenuj(m, torch.optim.SGD(m.parameters(), lr=lr), epoch=3, tisk=False)
    hl = np.convolve(z, np.ones(25) / 25, mode="valid")
    plt.plot(hl, label="lr = %g" % lr)
    print("lr = %-6g val presnost %.4f" % (lr, vyhodnot(m, Xva, yva)[1]))
plt.ylim(0, 2.5); plt.xlabel("krok"); plt.ylabel("trenovaci ztrata")
plt.legend(); plt.grid(alpha=.3); plt.show()
"""),

code("""
volby = {
    "SGD lr=0.1":          lambda p: torch.optim.SGD(p, lr=0.1),
    "SGD+mom lr=0.01":     lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "Adam lr=1e-3":        lambda p: torch.optim.Adam(p, lr=1e-3),
    "AdamW lr=1e-3":       lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01),
    "Adam lr=0.1 (!)":     lambda p: torch.optim.Adam(p, lr=0.1),
}
for jmeno, f in volby.items():
    torch.manual_seed(0); m = mlp()
    trenuj(m, f(m.parameters()), epoch=5, tisk=False)
    print("%-18s val presnost %.4f" % (jmeno, vyhodnot(m, Xva, yva)[1]))
"""),

md("""
## 8) Měřítko vstupů a aktivace

Přednáška 3: *Proč na měřítku vstupů záleží* a *Mizející a explodující gradienty*.
"""),

code("""
for rezim in ("raw", "01", "std"):
    torch.manual_seed(0); m = mlp()
    trenuj(m, torch.optim.SGD(m.parameters(), lr=0.1), epoch=3, tisk=False,
           X=priprav(Xtr_raw, rezim), Xv=priprav(Xva_raw, rezim))
    print("vstup %-4s val presnost %.4f" % (rezim, vyhodnot(m, priprav(Xva_raw, rezim), yva)[1]))
"""),

code("""
def he_init(model):
    for l in model:
        if isinstance(l, nn.Linear):
            nn.init.kaiming_normal_(l.weight, nonlinearity="relu")
            nn.init.zeros_(l.bias)

hluboka = (784,) + (64,) * 10 + (10,)
for jmeno, akt, he in (("sigmoida", nn.Sigmoid, False), ("ReLU", nn.ReLU, False),
                       ("ReLU + He init", nn.ReLU, True)):
    torch.manual_seed(0); m = mlp(hluboka, akt=akt)
    if he:
        he_init(m)
    loss_fn(m(Xtr[:512]), ytr[:512]).backward()
    normy = [l.weight.grad.norm().item() for l in m if isinstance(l, nn.Linear)]
    plt.semilogy(range(1, 12), normy, "o-", label=jmeno)
    m.zero_grad()
    trenuj(m, torch.optim.SGD(m.parameters(), lr=0.1), epoch=3, tisk=False)
    print("%-15s val presnost po 3 epochach %.4f" % (jmeno, vyhodnot(m, Xva, yva)[1]))
plt.xlabel("vrstva (1 = u vstupu)"); plt.ylabel("norma gradientu vah")
plt.legend(); plt.grid(alpha=.3); plt.show()
"""),

md("""
Sigmoida má derivaci nejvýš $1/4$, takže gradient k první vrstvě zeslábne
o více než sedm řádů. ReLU tenhle činitel nemá, ale s výchozí inicializací PyTorche se
rozptyl signálu vrstvu po vrstvě stejně zmenšuje. Inicializace, která s ReLU
počítá (He/Kaiming), je téma šesté přednášky.
"""),

md("""
## 9) Diagnostika

**Přeučit jednu dávku** (přednáška 4, *Diagnostika běhu*). Síť se 100 tisíci
parametry si 64 příkladů musí zapamatovat. Když to nedokáže, je chyba v datech,
ztrátě nebo grafu.
"""),

code("""
xb, yb = Xtr[:64], ytr[:64]
for sm in (False, True):
    torch.manual_seed(0); m = mlp(softmax_navic=sm)
    o = torch.optim.Adam(m.parameters(), lr=1e-3)
    z = []
    for k in range(300):
        o.zero_grad(); loss = loss_fn(m(xb), yb); loss.backward(); o.step()
        z.append(loss.item())
    plt.semilogy(z, label="softmax navic" if sm else "spravne")
    print("softmax navic" if sm else "spravne", "-> ztrata po 300 krocich %.4f" % z[-1])
plt.xlabel("krok"); plt.ylabel("ztrata na jedne davce"); plt.legend(); plt.show()
"""),

md("""
**Dlouhý běh** větší sítě. Sledujeme trénovací i validační ztrátu; kde se
rozejdou, začíná přeučení z páté přednášky; co s ním, je v oddílu 10.
"""),

code("""
torch.manual_seed(0)
model = mlp((784, 512, 512, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr),
                                     batch_size=128, shuffle=True,
                                     generator=torch.Generator().manual_seed(0))
hist, nejlepsi = [], (-1, None, 0)
for ep in range(1, 31):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    lt, at = vyhodnot(model, Xtr, ytr)
    lv, av = vyhodnot(model, Xva, yva)
    hist.append((lt, lv))
    if av > nejlepsi[0]:
        nejlepsi = (av, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    print("epocha %2d  train %.4f / %.4f   val %.4f / %.4f" % (ep, lt, at, lv, av))

plt.plot([h[0] for h in hist], label="trenovaci"); plt.plot([h[1] for h in hist], label="validacni")
plt.yscale("log"); plt.xlabel("epocha"); plt.ylabel("ztrata"); plt.legend(); plt.show()
"""),

md("""
## 10) Regularizace

Na plných 50 000 příkladech je mezera malá a rozdíly mezi technikami by
zanikly v šumu. Vezmeme proto jen **prvních 5 000 trénovacích příkladů**;
validace zůstává celá. Každou variantu trénujeme 60 epoch (2 400 kroků)
a sledujeme trénovací i validační přesnost (přednáška 5, *Generalizační
mezera*). Na přednášce běží každá varianta s jedním seedem; tabulka na
slajdech je průměr ze tří.
"""),

code("""
Xm, ym = Xtr[:5000], ytr[:5000]

def posun(xb, gen, max_posun=2):
    # kazdy obrazek davky nahodne posune o -2..2 px v obou osach
    n = xb.shape[0]
    img = xb.view(n, 28, 28)
    d = torch.randint(-max_posun, max_posun + 1, (n, 2), generator=gen)
    out = torch.stack([torch.roll(img[i], (int(d[i, 0]), int(d[i, 1])), (0, 1))
                       for i in range(n)])
    return out.view(n, 784)

def zrcadleni(xb, gen):
    # polovinu obrazku davky zrcadli zleva doprava
    n = xb.shape[0]
    img = xb.view(n, 28, 28).clone()
    k = torch.rand(n, generator=gen) < 0.5
    img[k] = img[k].flip(2)
    return img.view(n, 784)

@torch.no_grad()
def jistota_omylu(model, X, y):
    # prumerna pravdepodobnost, kterou model dal chybne predikovane tride
    model.eval()
    p = torch.softmax(model(X), 1)
    pred = p.argmax(1)
    spatne = pred != y
    return p[spatne, pred[spatne]].mean().item()

def trenuj_reg(X, y, seed=0, epoch=60, dropout=0.0, wd=0.0, ls=0.0, aug=None,
               sirky=(784, 512, 512, 10)):
    torch.manual_seed(seed)
    m = mlp(sirky, dropout=dropout)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    lf = nn.CrossEntropyLoss(label_smoothing=ls)
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y),
                                         batch_size=128, shuffle=True, generator=gen)
    hist, nejlepsi = [], (-1, 0, None)
    for ep in range(1, epoch + 1):
        m.train()
        for xb, yb in loader:
            if aug is not None:
                xb = aug(xb, gen)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
        lt, at = vyhodnot(m, X, y)
        lv, av = vyhodnot(m, Xva, yva)
        hist.append((ep, at, av, lt, lv))
        if av > nejlepsi[0]:
            nejlepsi = (av, ep, {k: v.clone() for k, v in m.state_dict().items()})
    return m, hist, nejlepsi

def vypis(jmeno, m, hist, nejlepsi):
    ep, at, av, lt, lv = hist[-1]
    print("%-12s train %.4f  val %.4f  (nejlepsi %.4f v epose %d)  jistota omylu %.3f"
          % (jmeno, at, av, nejlepsi[0], nejlepsi[1], jistota_omylu(m, Xva, yva)))
"""),

code("""
t0 = time.time()
m, hist, nej = trenuj_reg(Xm, ym)
vypis("bez regul.", m, hist, nej)
print("%.0f s" % (time.time() - t0))

plt.plot([h[1] for h in hist], label="trenovaci"); plt.plot([h[2] for h in hist], label="validacni")
plt.xlabel("epocha"); plt.ylabel("presnost"); plt.legend(); plt.grid(alpha=.3); plt.show()
"""),

code("""
# weight decay (AdamW, oddeleny od gradientu -- prednaska 5)
for wd in (0.01, 0.1, 1.0):
    vypis("wd = %g" % wd, *trenuj_reg(Xm, ym, wd=wd))
"""),

code("""
# dropout za kazdou skrytou ReLU
for p in (0.2, 0.5):
    m, hist, nej = trenuj_reg(Xm, ym, dropout=p)
    vypis("dropout %.1f" % p, m, hist, nej)

# zapomenute model.eval(): dropout zustane zapnuty i pri vyhodnoceni
m.train()
with torch.no_grad():
    for _ in range(2):
        print("bez eval():", (m(Xva).argmax(1) == yva).float().mean().item())
print("s eval():  ", vyhodnot(m, Xva, yva)[1])
"""),

code("""
# augmentace: posun ano, zrcadleni ne
vypis("posun", *trenuj_reg(Xm, ym, aug=posun))
vypis("zrcadleni", *trenuj_reg(Xm, ym, aug=zrcadleni))

# label smoothing: jine cile, ne jina data
vypis("ls 0.1", *trenuj_reg(Xm, ym, ls=0.1))
vypis("posun + ls", *trenuj_reg(Xm, ym, aug=posun, ls=0.1))
"""),

md("""
## 11) Závěrečný model a test

Kombinaci, která na validaci vyšla nejlépe (posun a label smoothing; na
slajdech je srovnání všech variant se třemi seedy), natrénujeme na **všech**
50 000 příkladech. Váhy vezmeme z epochy s nejvyšší validační
přesností (early stopping s uloženým checkpointem) a teprve pak, jednou,
změříme test.
"""),

code("""
t0 = time.time()
model, hist, nejlepsi = trenuj_reg(Xtr, ytr, epoch=30, aug=posun, ls=0.1)
for ep, at, av, lt, lv in hist:
    print("epocha %2d  train %.4f  val %.4f" % (ep, at, av))
print("%.0f s" % (time.time() - t0))
"""),

code("""
model.load_state_dict(nejlepsi[2])
print("vahy z epochy", nejlepsi[1], "| validace %.4f" % nejlepsi[0])
lte, ate = vyhodnot(model, Xte, y_test)
print("TEST: ztrata %.4f, presnost %.4f" % (lte, ate))

with torch.no_grad():
    pred = model(Xte).argmax(1)
C = torch.zeros(10, 10, dtype=torch.long)
for a, b in zip(y_test.tolist(), pred.tolist()):
    C[a, b] += 1
print(C)
"""),

md("""
## 12) Na doma

Čtyři krátké úlohy. Každá ověřuje jednu věc z přednášek tím, že ji napíšete sami.
"""),

md("""
**Úkol 1 — vlastní SGD s momentem.** Napište krok optimalizátoru ručně
(přednáška 4, *Setrvačnost*) a ověřte, že po 100 krocích dostanete tytéž váhy
jako `torch.optim.SGD(..., momentum=0.9)`. Pozor: PyTorch používá tvar
$\\bm v \\leftarrow \\alpha\\bm v + \\bm g$, $\\bm w \\leftarrow \\bm w - \\eta\\bm v$,
který se od tvaru ze slajdu liší jen přeškálováním $\\bm v$.
"""),

task(stub="""
def muj_krok(parametry, rychlosti, lr=0.01, alfa=0.9):
    # TODO: pro kazdy parametr p a jeho rychlost v proved krok momenta
    #       (uvnitr torch.no_grad(), gradient je v p.grad)
    raise NotImplementedError

torch.manual_seed(0); m1 = mlp()
torch.manual_seed(0); m2 = mlp()
opt2 = torch.optim.SGD(m2.parameters(), lr=0.01, momentum=0.9)
rychlosti = [torch.zeros_like(p) for p in m1.parameters()]
for k in range(100):
    xb, yb = Xtr[128*k:128*(k+1)], ytr[128*k:128*(k+1)]
    for m in (m1, m2):
        m.zero_grad(); loss_fn(m(xb), yb).backward()
    muj_krok(list(m1.parameters()), rychlosti)
    opt2.step()
print("max. rozdil vah:", max((a - b).abs().max().item()
                              for a, b in zip(m1.parameters(), m2.parameters())))
""", solution="""
def muj_krok(parametry, rychlosti, lr=0.01, alfa=0.9):
    with torch.no_grad():
        for p, v in zip(parametry, rychlosti):
            v.mul_(alfa).add_(p.grad)
            p.sub_(lr * v)

torch.manual_seed(0); m1 = mlp()
torch.manual_seed(0); m2 = mlp()
opt2 = torch.optim.SGD(m2.parameters(), lr=0.01, momentum=0.9)
rychlosti = [torch.zeros_like(p) for p in m1.parameters()]
for k in range(100):
    xb, yb = Xtr[128*k:128*(k+1)], ytr[128*k:128*(k+1)]
    for m in (m1, m2):
        m.zero_grad(); loss_fn(m(xb), yb).backward()
    muj_krok(list(m1.parameters()), rychlosti)
    opt2.step()
print("max. rozdil vah:", max((a - b).abs().max().item()
                              for a, b in zip(m1.parameters(), m2.parameters())))
"""),

md("""
**Úkol 2 — hranice stability pro Adam.** Na přednášce jsme hledali, od jakého
`lr` SGD diverguje. Udělejte totéž pro Adam: projděte
`lr` $\\in \\{10^{-4}, 3\\cdot10^{-4}, 10^{-3}, 3\\cdot10^{-3}, 10^{-2}, 3\\cdot10^{-2}, 10^{-1}\\}$
(dvě epochy) a vypište validační přesnost. *K zamyšlení:* proč je rozumný
rozsah pro Adam o dva řády níž než pro SGD?
"""),

task(stub="""
for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    # TODO: natrenujte mlp() s Adamem a timto lr (2 epochy, seed 0)
    #       a vypiste validacni presnost
    raise NotImplementedError
""", solution="""
for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    torch.manual_seed(0); m = mlp()
    trenuj(m, torch.optim.Adam(m.parameters(), lr=lr), epoch=2, tisk=False)
    print("Adam lr = %-7g val presnost %.4f" % (lr, vyhodnot(m, Xva, yva)[1]))
"""),

md("""
**Úkol 3 — nejčastější záměny.** Z matice záměn `C` vypište pět dvojic
(skutečná třída, predikce) s nejvyšším počtem chyb a pro každou ukažte jeden
obrázek. *K zamyšlení:* jsou to chyby, které by udělal i člověk?
"""),

task(stub="""
# TODO: vynulujte diagonalu kopie C, najdete 5 nejvetsich prvku
#       a ke kazde dvojici (a, b) zobrazte jeden testovaci obrazek s y=a, pred=b
raise NotImplementedError
""", solution="""
D = C.clone(); D.fill_diagonal_(0)
hodnoty, poradi = D.flatten().sort(descending=True)
fig, axes = plt.subplots(1, 5, figsize=(8, 1.9))
for ax, h, k in zip(axes, hodnoty[:5], poradi[:5]):
    a, b = divmod(k.item(), 10)
    i = ((y_test == a) & (pred == b)).nonzero()[0].item()
    ax.imshow(X_test_raw[i], cmap="gray_r"); ax.axis("off")
    ax.set_title("%d -> %d (%dx)" % (a, b, h))
    print("skutecna %d, predikce %d: %d chyb" % (a, b, h))
plt.show()
"""),

md("""
**Úkol 4 — weight decay bez biasů.** Pátá přednáška (*Další varianty a co se
nepenalizuje*) říká, že biasy se z weight decay vyjímají. `AdamW` ale ve
výchozím nastavení zmenšuje všechny parametry. Rozdělte parametry do dvou
skupin (`param_groups`: váhy s `weight_decay=0.1`, biasy s `0`) a porovnejte
s variantou, kde decay dostanou všechny. *K zamyšlení:* proč je rozdíl
u této sítě malý?
"""),

task(stub="""
def adamw_bez_biasu(model, wd):
    # TODO: vratte torch.optim.AdamW se dvema skupinami parametru:
    #       vahy (p.dim() > 1) s weight_decay=wd, biasy s weight_decay=0
    raise NotImplementedError

for jmeno, f in (("vsechny", lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)),
                 ("bez biasu", lambda m: adamw_bez_biasu(m, 0.1))):
    torch.manual_seed(0); m = mlp((784, 512, 512, 10))
    trenuj(m, f(m), X=Xm, y=ym, epoch=20, tisk=False)
    print("%-10s val presnost %.4f" % (jmeno, vyhodnot(m, Xva, yva)[1]))
""", solution="""
def adamw_bez_biasu(model, wd):
    vahy = [p for p in model.parameters() if p.dim() > 1]
    biasy = [p for p in model.parameters() if p.dim() <= 1]
    return torch.optim.AdamW([{"params": vahy, "weight_decay": wd},
                              {"params": biasy, "weight_decay": 0.0}], lr=1e-3)

for jmeno, f in (("vsechny", lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)),
                 ("bez biasu", lambda m: adamw_bez_biasu(m, 0.1))):
    torch.manual_seed(0); m = mlp((784, 512, 512, 10))
    trenuj(m, f(m), X=Xm, y=ym, epoch=20, tisk=False)
    print("%-10s val presnost %.4f" % (jmeno, vyhodnot(m, Xva, yva)[1]))
"""),

md("""
## Co si odnést

- Trénovací smyčka má pět řádků a každý odpovídá jedné přednášce.
- Před trénováním: baseline, ztráta $\\approx \\ln K$, přeučení jedné dávky.
- `CrossEntropyLoss` chce logity; `zero_grad` není zdvořilost, ale nutnost.
- Učicí krok se ladí jako první a pro každý optimalizátor zvlášť.
- Regularizace se porovnává na validaci, při stejném rozpočtu a s několika seedy.
- `model.eval()` není formalita: s dropoutem mění výsledek.
- Validace slouží k rozhodování, test k jedinému závěrečnému měření.
"""),
]

build(__file__, cislo='05b', nazev="První síť od dat po test", cells=CELLS,
      predpona='pr', druh='Praktická přednáška')
