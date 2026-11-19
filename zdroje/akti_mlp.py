# -*- coding: utf-8 -*-
"""AKTI -- MLP na MNIST v PyTorch (90 min, pokrocilejsi publikum).

Prepracovana verze 2024/AKTI.ipynb. Generuje akti/AKTI_MLP.ipynb
(jedina verze, bez reseni -- experimenty se ridi zmenou parametru).

    python zdroje/akti_mlp.py
"""
import io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbbuild import md, code, _cell, _notebook, REPO

CELLS = [

md(r"""
# AKTI — vícevrstvý perceptron na ručně psaných číslicích

Postavíme a natrénujeme malou neuronovou síť (MLP), která rozpozná číslice
0–9 z datasetu MNIST s přesností kolem 97,5 %. Pak se podíváme, **co se síť
naučila**, **kde chybuje** a **co se stane, když něco změníme**.

**Předpoklady:** perceptron, vícevrstvý perceptron a hrubá představa
o backpropagation. PyTorch znát nemusíte; ukážeme si, jak se známé pojmy
zapisují v něm.
"""),

# ---------------------------------------------------------------- 1
md(r"""
## 1) Knihovny a zařízení

| import | k čemu |
|---|---|
| `torch` | tenzory (vícerozměrná pole) a automatický výpočet gradientů |
| `torch.nn` | stavební bloky sítí: vrstvy, aktivační a ztrátové funkce |
| `torch.utils.data` | `DataLoader`: dělení dat na dávky (minibatche) a jejich míchání |
| `matplotlib` | grafy a obrázky |

Pevné **semínko** (`manual_seed`) zajistí, že při opakovaném spuštění vyjdou
stejné počáteční váhy, a tedy i stejná čísla.
"""),

code(r"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| počítáme na:", device)
"""),

# ---------------------------------------------------------------- 2
md(r"""
## 2) Data

MNIST obsahuje 60 000 trénovacích a 10 000 testovacích obrázků 28×28 pixelů
v odstínech šedi; hodnota pixelu je celé číslo 0–255.

Načteme celý dataset rovnou do paměti jako tenzory (je malý, 50 MB). Je to
rychlejší než převádět každý obrázek zvlášť při každém průchodu.
"""),

code(r"""
def nacti_mnist(slozka="./data"):
    try:
        from torchvision import datasets
        a = datasets.MNIST(slozka, train=True, download=True)
        b = datasets.MNIST(slozka, train=False, download=True)
        return a.data, a.targets, b.data, b.targets
    except ImportError:
        # bez torchvision: soubory IDX z data/MNIST/raw
        def idx(nazev):
            with open(os.path.join(slozka, "MNIST", "raw", nazev), "rb") as f:
                d = f.read()
            n = d[3]
            tvar = [int.from_bytes(d[4 + 4*i:8 + 4*i], "big") for i in range(n)]
            return torch.tensor(np.frombuffer(d, np.uint8, offset=4 + 4*n).reshape(tvar))
        return (idx("train-images-idx3-ubyte"), idx("train-labels-idx1-ubyte").long(),
                idx("t10k-images-idx3-ubyte"), idx("t10k-labels-idx1-ubyte").long())

X_all, y_all, X_test_raw, y_test = nacti_mnist()
print("trénovací:", tuple(X_all.shape), X_all.dtype,
      "| hodnoty", X_all.min().item(), "až", X_all.max().item())
print("testovací:", tuple(X_test_raw.shape))
"""),

code(r"""
fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))
for ax, i in zip(axes.flat, range(20)):
    ax.imshow(X_all[i], cmap="gray_r")
    ax.set_title(int(y_all[i]), fontsize=10)
    ax.axis("off")
plt.show()

print("počty obrázků jednotlivých číslic:", torch.bincount(y_all).tolist())
"""),

md(r"""
### Trénovací, validační a testovací část

Z trénovacích obrázků odložíme 5 000 na **validaci**. Na ní budeme během
trénování sledovat, jak si síť vede na datech, která nevidí. **Testovací**
množinu použijeme jen jednou, úplně na konci. Kdybychom podle ní ladili
nastavení, odhad přesnosti by vyšel příznivěji, než jaký ve skutečnosti je.

### Normalizace

Pixely převedeme na škálu 0–1 a pak je **standardizujeme**: odečteme průměr
a vydělíme směrodatnou odchylkou. Vstupy pak mají průměr 0 a rozptyl 1, což
optimalizaci výrazně usnadní. Průměr a odchylku počítáme **jen z trénovací
části**.
"""),

code(r"""
perm = torch.randperm(60000, generator=torch.Generator().manual_seed(0))
idx_tr, idx_va = perm[:55000], perm[55000:]

mu = (X_all[idx_tr].float() / 255).mean().item()
sd = (X_all[idx_tr].float() / 255).std().item()
print(f"průměr {mu:.4f}, směrodatná odchylka {sd:.4f}")

def priprav(X):
    # (N, 28, 28) uint8  ->  (N, 1, 28, 28) float, standardizovane
    return ((X.float() / 255 - mu) / sd).unsqueeze(1)

Xtr, ytr = priprav(X_all[idx_tr]), y_all[idx_tr]
Xva, yva = priprav(X_all[idx_va]), y_all[idx_va]
Xte, yte = priprav(X_test_raw), y_test
print("trénovací", tuple(Xtr.shape), "| validační", tuple(Xva.shape),
      "| testovací", tuple(Xte.shape))
"""),

md(r"""
Rozměr `(N, 1, 28, 28)` znamená *počet obrázků × kanály × výška × šířka*.
Kanál je jeden, protože obrázky jsou šedotónové (barevné by měly tři).

### Dávky

`DataLoader` rozdělí data na **dávky** (*minibatch*) po 64 obrázcích a na začátku
každé epochy je zamíchá. Síť po každé dávce jednou upraví váhy, za jednu epochu
tedy zhruba 860krát.
"""),

code(r"""
BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)

xb, yb = next(iter(train_loader))
print("jedna dávka:", tuple(xb.shape), tuple(yb.shape), "| počet dávek:", len(train_loader))
"""),

# ---------------------------------------------------------------- 3
md(r"""
## 3) Model

Síť je obyčejný vícevrstvý perceptron, jen zapsaný v PyTorch. Každý model
v PyTorch je třída odvozená od `nn.Module`. V **konstruktoru** `__init__` se
vytvoří vrstvy (a tím jejich váhy), v metodě **`forward`** se popíše, jak jimi
tečou data.

Naše síť:

```
obrázek 28×28 ─► zploštit na 784 ─► Linear 784→128 ─► ReLU ─► Linear 128→128 ─► ReLU ─► Linear 128→10 ─► 10 skóre
```

- `nn.Flatten()` udělá z obrázku vektor 784 čísel. Síť tím **přijde
  o informaci, které pixely spolu sousedí**; k tomu se vrátíme v experimentu 7c.
- `nn.Linear(n, m)` je vrstva $m$ neuronů, z nichž každý je propojený se všemi
  $n$ vstupy: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$, kde $W$ má rozměr
  $m\times n$. Řádek $W$ jsou váhy jednoho neuronu, tedy jeden perceptron
  bez prahové funkce.
- Jako aktivační funkci použijeme místo sigmoidy **ReLU**, $\max(0, x)$. Počítá
  se rychleji a její derivace je pro kladné vstupy rovna jedné, takže se gradient
  při zpětném šíření tolik nezmenšuje. Bez nelinearity by se tři lineární vrstvy
  složily v jedinou lineární funkci (experiment 7a).
- Výstupem je 10 čísel, **logitů**, jedno skóre pro každou číslici. Na
  pravděpodobnosti je převede až softmax (níže), který je schovaný ve ztrátové
  funkci.

Parametry `hidden`, `aktivace` a `dropout` využijeme v experimentech.
"""),

code(r"""
class SimpleMLP(nn.Module):
    def __init__(self, hidden=128, aktivace=nn.ReLU, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_1 = nn.Linear(28 * 28, hidden)
        self.layer_2 = nn.Linear(hidden, hidden)
        self.output_layer = nn.Linear(hidden, 10)
        self.act = aktivace()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)                          # (B, 1, 28, 28) -> (B, 784)
        x = self.drop(self.act(self.layer_1(x)))     # (B, 128)
        x = self.drop(self.act(self.layer_2(x)))     # (B, 128)
        return self.output_layer(x)                  # (B, 10) logity

torch.manual_seed(0)
model = SimpleMLP().to(device)
print(model)
"""),

md(r"""
**Předpověď.** Kolik má tato síť parametrů (vah a biasů)?

<details><summary>Odpověď</summary>

$784\cdot128+128 = 100\,480$, $128\cdot128+128 = 16\,512$, $128\cdot10+10 = 1\,290$,
celkem $118\,282$. Přes 85 % parametrů je v první vrstvě: každý ze 128 neuronů
má vlastní váhu pro každý pixel.
</details>
"""),

code(r"""
for jmeno, p in model.named_parameters():
    print(f"  {jmeno:22s} {str(tuple(p.shape)):12s} {p.numel():7d}")
print("celkem parametrů:", sum(p.numel() for p in model.parameters()))
"""),

md(r"""
### Softmax a křížová entropie

U jednoho výstupního neuronu převádí skóre na pravděpodobnost sigmoida. Pro
deset tříd se používá její zobecnění, **softmax**:

$$p_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}.$$

Exponenciála udělá ze všech skóre kladná čísla a dělení součtem zajistí, že
dají dohromady jedničku. Největší skóre dostane největší pravděpodobnost.

Ztrátovou funkcí není střední kvadratická chyba, ale **křížová entropie**: pro
správnou třídu $c$ je ztráta $-\ln p_c$. Je nulová, když si je síť správnou
odpovědí jistá, a roste nade všechny meze, když správné třídě dává
pravděpodobnost blízkou nule. `nn.CrossEntropyLoss` dostane přímo logity
a softmax si spočítá sama.
"""),

code(r"""
z = torch.tensor([[2.0, 1.0, 0.1]])          # logity pro 3 třídy
p = z.softmax(dim=1)
print("pravděpodobnosti:", p.numpy().round(3), f"| součet {p.sum().item():.3f}")
for c in range(3):
    print(f"správná třída {c}: -ln p = {-torch.log(p[0, c]).item():.3f}",
          f"| CrossEntropyLoss = {nn.CrossEntropyLoss()(z, torch.tensor([c])).item():.3f}")
"""),

md(r"""
### Co umí síť, která se ještě nic nenaučila

Napíšeme si funkci, která spočítá průměrnou ztrátu a přesnost na celé množině.
`model.eval()` přepne síť do vyhodnocovacího režimu (vypne dropout) a
`torch.no_grad()` vypne sledování gradientů, které při vyhodnocení nepotřebujeme.

**Předpověď.** Jakou přesnost a jakou ztrátu bude mít síť s náhodnými vahami?

<details><summary>Odpověď</summary>

Přesnost kolem 10 %, tedy náhodné hádání mezi deseti třídami. Ztráta kolem
$\ln 10 \approx 2{,}30$: s malými náhodnými vahami jsou logity skoro stejné,
softmax dá každé třídě asi $1/10$ a $-\ln(1/10) = \ln 10$. Ztráta výrazně
nad 2,3 na začátku učení by znamenala, že je něco špatně (například špatně
škálovaný vstup).
</details>
"""),

code(r"""
criterion = nn.CrossEntropyLoss()

def vyhodnot(model, X, y, davka=2000):
    model.eval()
    soucet_ztrat, spravne = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), davka):
            xb, yb = X[i:i + davka].to(device), y[i:i + davka].to(device)
            logity = model(xb)
            soucet_ztrat += criterion(logity, yb).item() * len(yb)
            spravne += (logity.argmax(dim=1) == yb).sum().item()
    return soucet_ztrat / len(X), spravne / len(X)

ztrata, presnost = vyhodnot(model, Xva, yva)
print(f"nenaučená síť: ztráta {ztrata:.3f} (ln 10 = {np.log(10):.3f}), přesnost {presnost:.1%}")
"""),

# ---------------------------------------------------------------- 4
md(r"""
## 4) Trénování

### Backpropagation za nás udělá PyTorch

Při dopředném průchodu si PyTorch zaznamenává, jaké operace se s tenzory
prováděly. Zavolání `loss.backward()` pak projde tento záznam pozpátku
a spočítá derivaci ztráty podle **každého** parametru:
přesně backpropagation, jen ji nemusíme psát ručně. Výsledek uloží do
atributu `.grad`, který má stejný tvar jako parametr sám.

Ukážeme si jeden krok gradientního sestupu úplně ručně, na jedné dávce.
"""),

code(r"""
torch.manual_seed(0)
model = SimpleMLP().to(device)
xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)

W1 = model.layer_1.weight
print("před backward: grad =", W1.grad)

loss = criterion(model(xb), yb)
loss.backward()
print("po backward:   tvar vah", tuple(W1.shape), "| tvar gradientu", tuple(W1.grad.shape))
print(f"ztráta na dávce před krokem: {loss.item():.4f}")

lr = 0.1
with torch.no_grad():                      # úprava vah se nemá zaznamenávat
    for p in model.parameters():
        p -= lr * p.grad                   # w <- w - lr * dL/dw
        p.grad = None                      # gradienty pro další krok vynulovat

print(f"ztráta na téže dávce po kroku: {criterion(model(xb), yb).item():.4f}")
"""),

md(r"""
Ztráta na dávce klesla. Celé učení je jen tento krok opakovaný tisíckrát na
různých dávkách.

### Trénovací smyčka

V praxi úpravu vah neprovádíme ručně, ale přenecháme ji **optimalizátoru**.
Jeden krok učení má vždy stejných pět řádků:

1. `optimizer.zero_grad()`: vynulovat gradienty z minulého kroku (PyTorch je jinak sčítá),
2. `logity = model(xb)`: dopředný průchod,
3. `loss = criterion(logity, yb)`: ztráta na dávce,
4. `loss.backward()`: zpětné šíření; ke každé váze se spočítá derivace ztráty,
5. `optimizer.step()`: posun vah proti gradientu.

Ruční krok z předchozí buňky odpovídá `torch.optim.SGD(model.parameters(), lr=0.1)`.
My použijeme **Adam**, vylepšený gradientní sestup. Pamatuje si průměr
posledních gradientů, takže se nenechá rozhodit šumem jedné dávky, a pro
každou váhu si sám přizpůsobuje velikost kroku. Rychlost učení `lr = 0.001`
je pro Adam obvyklá výchozí hodnota.

Po každé epoše spočítáme ztrátu a přesnost na trénovací i validační části,
abychom mohli nakreslit **křivky učení**.
"""),

code(r"""
EPOCHS = 5
LEARNING_RATE = 1e-3

torch.manual_seed(0)
model = SimpleMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
historie = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(EPOCHS):
    start = time.time()
    model.train()                                   # zapne dropout (pokud je)
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()                       # 1
        logity = model(xb)                          # 2
        loss = criterion(logity, yb)                # 3
        loss.backward()                             # 4
        optimizer.step()                            # 5

    tl, ta = vyhodnot(model, Xtr, ytr)
    vl, va = vyhodnot(model, Xva, yva)
    for k, v in zip(historie, (tl, vl, ta, va)):
        historie[k].append(v)
    print(f"epocha {epoch + 1}/{EPOCHS}: ztráta train {tl:.4f}, val {vl:.4f} | "
          f"přesnost train {ta:.2%}, val {va:.2%} | {time.time() - start:.1f} s")
"""),

code(r"""
def kresli(historie, popisky=None):
    if isinstance(historie, dict):
        historie, popisky = [historie], [""]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    for h, pop, barva in zip(historie, popisky, plt.cm.tab10.colors):
        ep = range(1, len(h["train_loss"]) + 1)
        a1.plot(ep, h["train_loss"], "--", color=barva, label=f"{pop} train".strip())
        a1.plot(ep, h["val_loss"], "-", color=barva, label=f"{pop} val".strip())
        a2.plot(ep, h["train_acc"], "--", color=barva)
        a2.plot(ep, h["val_acc"], "-", color=barva)
    a1.set(xlabel="epocha", ylabel="ztráta"); a1.legend(fontsize=8)
    a2.set(xlabel="epocha", ylabel="přesnost")
    plt.tight_layout(); plt.show()

kresli(historie)
"""),

md(r"""
Čárkovaně je trénovací, plnou čarou validační část.

**K diskusi.** Trénovací ztráta klesá s každou epochou, validační se po pár
epochách zastaví. Rozevírající se nůžky mezi nimi jsou první známka
**přeučení**: síť se začíná učit zvláštnosti trénovacích obrázků, které se
na nové obrázky nepřenášejí. Kolik epoch byste podle grafu zvolili?
"""),

# ---------------------------------------------------------------- 5
md(r"""
## 5) Vyhodnocení na testovacích datech

Teprve teď, s hotovým modelem, se jednou podíváme na testovací množinu.
"""),

code(r"""
tl, ta = vyhodnot(model, Xte, yte)
print(f"TEST: ztráta {tl:.4f}, přesnost {ta:.2%}  ({round((1 - ta) * len(yte))} chyb z {len(yte)})")

model.eval()
with torch.no_grad():
    logity = model(Xte.to(device)).cpu()
pravd = logity.softmax(dim=1)
pred = pravd.argmax(dim=1)
"""),

md(r"""
### Matice záměn

Řádek je skutečná číslice, sloupec predikovaná. Na diagonále jsou správné
odpovědi; zajímavé jsou čísla mimo ni. Které dvojice si síť plete nejčastěji
a dávají ty záměny smysl?
"""),

code(r"""
M = torch.zeros(10, 10, dtype=torch.long)
M.index_put_((yte, pred), torch.ones_like(pred), accumulate=True)

mimo = M.clone().fill_diagonal_(0)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.imshow(mimo, cmap="Reds")
for i in range(10):
    for j in range(10):
        if M[i, j] > 0:
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                    color="gray" if i == j else
                          ("white" if mimo[i, j] > 0.6 * mimo.max() else "black"))
ax.set(xticks=range(10), yticks=range(10), xlabel="predikce", ylabel="skutečnost",
       title="matice záměn (barva = jen chyby)")
plt.show()

nej = mimo.flatten().topk(5)
for pocet, k in zip(nej.values, nej.indices):
    print(f"skutečná {k.item() // 10} -> predikce {k.item() % 10}: {pocet.item()}x")
"""),

md(r"""
### Nejsebevědomější chyby

Mezi špatně klasifikovanými obrázky vybereme ty, u kterých si síť byla
nejjistější. Často jsou to obrázky, se kterými by váhal i člověk, nebo
obrázky s chybným štítkem.
"""),

code(r"""
spatne = (pred != yte).nonzero().flatten()
jistota = pravd[spatne, pred[spatne]]
poradi = spatne[jistota.argsort(descending=True)][:12]

fig, axes = plt.subplots(2, 6, figsize=(10, 4))
for ax, i in zip(axes.flat, poradi):
    ax.imshow(X_test_raw[i], cmap="gray_r")
    ax.set_title(f"síť: {pred[i].item()} ({pravd[i, pred[i]]:.0%})\nštítek: {yte[i].item()}",
                 fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()
"""),

md(r"""
Vysoká jistota u chybné odpovědi je obecná vlastnost: softmax z neuronové sítě
není spolehlivý odhad pravděpodobnosti, že má síť pravdu. Sítě bývají
**přehnaně sebevědomé**.
"""),

# ---------------------------------------------------------------- 6
md(r"""
## 6) Co se naučila první vrstva

Každý ze 128 neuronů první vrstvy má 784 vah, jednu na pixel. Můžeme je tedy
vykreslit jako obrázek 28×28. Červená místa neuron *zvyšují*, když je tam
inkoust, modrá ho *snižují*. Neuron tak reaguje na určitý vzor tahů.
"""),

code(r"""
W = model.layer_1.weight.detach().cpu().reshape(-1, 28, 28)
fig, axes = plt.subplots(3, 8, figsize=(10, 4))
for ax, w in zip(axes.flat, W):
    m = w.abs().max()
    ax.imshow(w, cmap="RdBu_r", vmin=-m, vmax=m)
    ax.axis("off")
plt.suptitle("váhy prvních 24 neuronů první vrstvy"); plt.show()
"""),

md(r"""
Vzory jsou rozmazané a těžko čitelné; jednotlivé neurony nejsou detektory
celých číslic. Rozhodnutí vzniká až kombinací mnoha neuronů v dalších vrstvách.
Okraje jsou bledé: v pixelech, kde nikdy není inkoust, je vstup pořád stejný
a váhy se tam skoro nehnuly.

Několik neuronů vypadá jako čistý šum. To jsou **mrtvé neurony**: ReLU na nich
vrací nulu pro všechny trénovací obrázky, takže přes ně neteče žádný gradient
a jejich váhy zůstaly takové, jaké byly při inicializaci.
"""),

code(r"""
with torch.no_grad():
    aktivace_1 = model.act(model.layer_1(model.flatten(Xtr.to(device))))
podil_aktivnich = (aktivace_1 > 0).float().mean(dim=0).cpu()
mrtve = (podil_aktivnich == 0).nonzero().flatten().tolist()
print(f"mrtvých neuronů v první vrstvě: {len(mrtve)} ze {len(podil_aktivnich)}:", mrtve)
"""),

# ---------------------------------------------------------------- 7
md(r"""
## 7) Experimenty

Trénovací smyčku zabalíme do funkce, abychom mohli rychle porovnávat různá
nastavení. Uvnitř je přesně to, co jsme psali výše.
"""),

code(r"""
def experiment(model, epochs=3, lr=1e-3, X=None, y=None, Xv=None, yv=None,
               weight_decay=0.0, batch_size=64, nazev=""):
    X = Xtr if X is None else X;   y = ytr if y is None else y
    Xv = Xva if Xv is None else Xv; yv = yva if yv is None else yv
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    h = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        tl, ta = vyhodnot(model, X, y)
        vl, va = vyhodnot(model, Xv, yv)
        for k, v in zip(h, (tl, vl, ta, va)):
            h[k].append(v)
    print(f"{nazev:26s} přesnost train {h['train_acc'][-1]:.2%}, val {h['val_acc'][-1]:.2%}"
          f" | ztráta val {h['val_loss'][-1]:.3f}")
    return h
"""),

md(r"""
### 7a) Je nelinearita opravdu potřeba?

Nahradíme ReLU identitou (`nn.Identity`), takže síť bude složením tří
lineárních vrstev.

**Předpověď.** Bude síť bez nelinearity výrazně horší? O kolik?

<details><summary>Odpověď</summary>

Složení lineárních funkcí je lineární funkce, takže síť bez nelinearity je jen
jinak zapsaný lineární klasifikátor (multinomická logistická regrese). Na MNIST
to přitom vůbec není špatné, kolem 91 %. Rozdíl proti MLP je tedy „jen“ několik
procentních bodů, jenže to znamená **víc než třikrát tolik chyb**.
</details>
"""),

code(r"""
torch.manual_seed(0); h_relu = experiment(SimpleMLP(aktivace=nn.ReLU), nazev="ReLU")
torch.manual_seed(0); h_lin = experiment(SimpleMLP(aktivace=nn.Identity), nazev="bez nelinearity")
kresli([h_relu, h_lin], ["ReLU", "lineární"])
"""),

md(r"""
### 7b) Rychlost učení

**Předpověď.** Co se stane s rychlostí učení 0,1 (stokrát větší) a 0,00001
(stokrát menší)?

<details><summary>Odpověď</summary>

Příliš velký krok přeskakuje minimum: ztráta skáče nebo se zasekne vysoko
a přesnost zůstane mizerná. Příliš malý krok vede správným směrem, ale tak
pomalu, že za tři epochy síť nedojde daleko. Dobré hodnoty leží v poměrně
širokém pásmu kolem 0,001.
</details>
"""),

code(r"""
hs = []
for lr in (1e-1, 1e-3, 1e-5):
    torch.manual_seed(0)
    hs.append(experiment(SimpleMLP(), lr=lr, nazev=f"lr = {lr:g}"))
kresli(hs, ["lr 0.1", "lr 0.001", "lr 0.00001"])
"""),

md(r"""
### 7c) Zamíchané pixely

Vezmeme jednu pevnou náhodnou permutaci 784 pozic a přeházíme podle ní pixely
ve **všech** obrázcích stejně. Pro člověka se z obrázků stane šum.
"""),

code(r"""
g = torch.Generator().manual_seed(1)
pix = torch.randperm(784, generator=g)
zamichej = lambda X: X.flatten(1)[:, pix].reshape(X.shape)
Xtr_p, Xva_p = zamichej(Xtr), zamichej(Xva)

fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for k in range(8):
    axes[0, k].imshow(Xtr[k, 0], cmap="gray_r"); axes[0, k].set_title(int(ytr[k]))
    axes[1, k].imshow(Xtr_p[k, 0], cmap="gray_r")
for ax in axes.flat:
    ax.axis("off")
plt.show()
"""),

md(r"""
**Předpověď.** Jakou přesnost bude mít MLP natrénovaná na zamíchaných obrázcích?

<details><summary>Odpověď</summary>

Prakticky stejnou jako na původních. MLP nemá žádnou představu o tom, které
pixely spolu sousedí: první vrstva propojí každý pixel s každým neuronem
a permutace jen přeuspořádá sloupce matice $W$. Síť se naučí totéž, jen
s přeházenými vahami.

To je zároveň slabina: sousedství pixelů je pro rozpoznávání obrázků velmi
užitečná informace a MLP ji zahazuje. Existují architektury, které ji
využívají, **konvoluční sítě**. Ty by na zamíchaných obrázcích naopak výrazně
ztratily.
</details>
"""),

code(r"""
torch.manual_seed(0)
h_perm = experiment(SimpleMLP(), X=Xtr_p, Xv=Xva_p, nazev="zamíchané pixely")
kresli([h_relu, h_perm], ["původní", "zamíchané"])
"""),

md(r"""
### 7d) Málo dat: přeučení a regularizace

Necháme síti jen **1 000** trénovacích obrázků a trénujeme 40 epoch. Pak
totéž s **dropoutem** (při trénování se náhodně vypíná polovina neuronů)
a **weight decay** (penalizace velkých vah). Nakonec pro srovnání **5 000**
obrázků bez regularizace.

**Předpověď.** Jak budou vypadat křivky pro 1 000 obrázků bez regularizace?
Co z toho regularizace spraví?

<details><summary>Odpověď</summary>

Trénovací přesnost rychle dosáhne 100 % a trénovací ztráta jde k nule: síť si
tisíc obrázků zapamatuje. Validační přesnost se zastaví kolem 89 % a validační
ztráta po několika epochách začne **růst**, přestože přesnost stojí. Roste
jen sebevědomí, se kterým síť odpovídá špatně.

Regularizace tady validační přesnost skoro nezmění; sníží validační ztrátu,
tedy přehnanou jistotu. Informaci, která v tisíci obrázcích není, nevyrobí.
Pětkrát víc dat pomůže víc než jakákoli regularizace.
</details>
"""),

code(r"""
X1k, y1k = Xtr[:1000], ytr[:1000]
torch.manual_seed(0)
h_bez = experiment(SimpleMLP(), epochs=40, X=X1k, y=y1k, nazev="1000 obrázků")
torch.manual_seed(0)
h_reg = experiment(SimpleMLP(dropout=0.5), epochs=40, X=X1k, y=y1k, weight_decay=1e-3,
                   nazev="1000 + dropout + wd")
torch.manual_seed(0)
h_5k = experiment(SimpleMLP(), epochs=40, X=Xtr[:5000], y=ytr[:5000], nazev="5000 obrázků")
kresli([h_bez, h_reg, h_5k], ["1000", "1000 + regularizace", "5000"])
"""),

md(r"""
### Další nápady na vlastní zkoušení

- Změňte šířku skryté vrstvy (`SimpleMLP(hidden=16)`, `hidden=512`). Jak se mění
  počet parametrů a přesnost?
- Zkuste jiný optimalizátor: `torch.optim.SGD(model.parameters(), lr=0.1)`.
  (Ve funkci `experiment` stačí přepsat jeden řádek.)
- Zkuste trénovat bez normalizace, s pixely 0–255:
  `experiment(SimpleMLP(), X=(Xtr * sd + mu) * 255, Xv=(Xva * sd + mu) * 255)`.
- Kolik trénovacích obrázků je potřeba na 95 % validační přesnosti?

## Shrnutí

- Trénovací krok je vždy stejný: `zero_grad`, dopředný průchod, ztráta,
  `backward`, `step`.
- Před učením se vyplatí znát výchozí bod: náhodná síť má přesnost 10 %
  a ztrátu $\ln 10$.
- Validační množina slouží k rozhodování během vývoje, testovací jen
  k jedinému závěrečnému měření.
- Nelinearita mezi vrstvami je to, co odlišuje síť od lineárního modelu.
- MLP nevyužívá sousedství pixelů. Sítě, které ho využívají (konvoluční),
  jsou dalším krokem.
"""),

]


def main():
    out = [_cell(c.kind, c.src, i) for i, c in enumerate(CELLS)]
    path = os.path.join(REPO, 'akti', 'AKTI_MLP.ipynb')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(_notebook(out), f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(path, 'bunek:', len(out))


if __name__ == '__main__':
    main()
