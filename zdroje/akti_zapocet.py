# -*- coding: utf-8 -*-
"""AKTI -- sablona zapoctoveho ukolu (experimentalni studie na Fashion-MNIST).

Navazuje na akti/AKTI_MLP.ipynb. Generuje akti/AKTI_zapocet_sablona.ipynb.

    python zdroje/akti_zapocet.py
"""
import io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbbuild import md, code, _cell, _notebook, REPO

COLAB = ('https://colab.research.google.com/github/konyconi/UMIN_AI_exercises/'
         'blob/main/akti/AKTI_zapocet_sablona.ipynb')

CELLS = [

md(r"""
[![Otevřít v Colabu](https://colab.research.google.com/assets/colab-badge.svg)](%s)

# AKTI — zápočtový úkol: malá experimentální studie na Fashion-MNIST

Úkol navazuje na cvičení *Vícevrstvý perceptron na ručně psaných číslicích*
(`AKTI_MLP.ipynb`). Místo číslic budete klasifikovat obrázky oblečení
z datasetu **Fashion-MNIST**: stejný formát (28×28 pixelů, 10 tříd), ale
úloha je těžší, takže rozdíly mezi nastaveními jsou lépe vidět.

Cílem není vyladit co nejlepší síť, ale **poctivě odpovědět na jednu otázku
experimentem**: hypotéza předem, několik opakování, správné použití
validačních a testovacích dat.

## Co odevzdat

Odkaz na **váš** notebook v Colabu (*Soubor → Uložit kopii na Disk*, pak
*Sdílet → kdokoli s odkazem může zobrazit*).

## Podmínky zápočtu

Zápočet je za splnění **všech** bodů:

| | bod | co musí být v notebooku |
|---|---|---|
| ☐ | 1. Výchozí model | MLP ze cvičení natrénovaný na Fashion-MNIST, validační přesnost |
| ☐ | 2. Hypotéza | otázka a odhad výsledku s odůvodněním, napsané **před** experimentem |
| ☐ | 3. Experiment | ≥ 3 nastavení × 3 seedy, tabulka průměr ± sm. odchylka, graf; test jen na konci |
| ☐ | 4. Rozbor chyb | dvě nejčastěji zaměňované třídy, ukázky obrázků, vysvětlení |
| ☐ | 5. Závěr | 5–10 vět, které odkazují na hypotézu a na vaše čísla |
| ☐ | technika | *Runtime → Run all* proběhne bez chyby do 10 minut; texty jsou v markdownových buňkách |

Nevadí, když se hypotéza nepotvrdí. Vadí, když závěr tvrdí něco, co z vašich
čísel neplyne.

Buňky označené **TODO** doplňujete vy, ostatní jsou připravené a nemusíte je
měnit. Kód i text můžete libovolně přidávat.
""" % COLAB),

md(r"""
**Jméno:** *doplňte*
"""),

# ---------------------------------------------------------------- priprava
md(r"""
## 0) Příprava: knihovny, data, pomocné funkce

Nic tu nemusíte měnit, jen spusťte. Funkce jsou stejné jako ve cvičení,
navíc je tu `MLP` s volitelným počtem a šířkou skrytých vrstev a funkce
`opakuj`, která spustí totéž nastavení s několika seedy.
"""),

code(r"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| počítáme na:", device)
"""),

code(r"""
TRIDY = ["tričko", "kalhoty", "svetr", "šaty", "kabát",
         "sandál", "košile", "tenisky", "kabelka", "kotníková bota"]

def nacti_fashion(slozka="./data"):
    try:
        from torchvision import datasets
        a = datasets.FashionMNIST(slozka, train=True, download=True)
        b = datasets.FashionMNIST(slozka, train=False, download=True)
        return a.data, a.targets, b.data, b.targets
    except ImportError:
        # bez torchvision: soubory IDX z data/FashionMNIST/raw
        def idx(nazev):
            with open(os.path.join(slozka, "FashionMNIST", "raw", nazev), "rb") as f:
                d = f.read()
            n = d[3]
            tvar = [int.from_bytes(d[4 + 4*i:8 + 4*i], "big") for i in range(n)]
            return torch.tensor(np.frombuffer(d, np.uint8, offset=4 + 4*n).reshape(tvar))
        return (idx("train-images-idx3-ubyte"), idx("train-labels-idx1-ubyte").long(),
                idx("t10k-images-idx3-ubyte"), idx("t10k-labels-idx1-ubyte").long())

X_all, y_all, X_test_raw, y_test = nacti_fashion()

# 55 000 trenovacich, 5 000 validacnich; rozdeleni je pevne (seed 0)
perm = torch.randperm(60000, generator=torch.Generator().manual_seed(0))
idx_tr, idx_va = perm[:55000], perm[55000:]
mu = (X_all[idx_tr].float() / 255).mean().item()
sd = (X_all[idx_tr].float() / 255).std().item()

def priprav(X):
    return ((X.float() / 255 - mu) / sd).unsqueeze(1)

Xtr, ytr = priprav(X_all[idx_tr]), y_all[idx_tr]
Xva, yva = priprav(X_all[idx_va]), y_all[idx_va]
Xte, yte = priprav(X_test_raw), y_test
X_va_raw = X_all[idx_va]
print("trénovací", tuple(Xtr.shape), "| validační", tuple(Xva.shape),
      "| testovací", tuple(Xte.shape))

fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for ax, i in zip(axes.flat, range(20)):
    ax.imshow(X_all[i], cmap="gray_r")
    ax.set_title(TRIDY[y_all[i]], fontsize=8)
    ax.axis("off")
plt.show()
"""),

code(r"""
class MLP(nn.Module):
    # skryte=(128, 128) je sit ze cviceni; (256,) jedna sirsi vrstva, (64, 64, 64) tri uzsi
    def __init__(self, skryte=(128, 128), aktivace=nn.ReLU, dropout=0.0):
        super().__init__()
        vrstvy, vstup = [nn.Flatten()], 28 * 28
        for h in skryte:
            vrstvy += [nn.Linear(vstup, h), aktivace(), nn.Dropout(dropout)]
            vstup = h
        vrstvy.append(nn.Linear(vstup, 10))
        self.sit = nn.Sequential(*vrstvy)

    def forward(self, x):
        return self.sit(x)

def pocet_parametru(model):
    return sum(p.numel() for p in model.parameters())

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

def experiment(model, epochs=5, lr=1e-3, X=None, y=None, weight_decay=0.0,
               batch_size=64, nazev="", tisk=True):
    # trenuje model; po kazde epose meri ztratu a presnost na train a VALIDACI (ne na testu)
    X = Xtr if X is None else X
    y = ytr if y is None else y
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
        vl, va = vyhodnot(model, Xva, yva)
        for k, v in zip(h, (tl, vl, ta, va)):
            h[k].append(v)
    h["model"] = model
    if tisk:
        print(f"{nazev:28s} train {h['train_acc'][-1]:.2%}, val {h['val_acc'][-1]:.2%}"
              f" | ztráta val {h['val_loss'][-1]:.3f}")
    return h

def opakuj(vytvor_model, seedy=(0, 1, 2), nazev="", **kwargs):
    # stejne nastaveni s nekolika seedy; vytvor_model je funkce bez argumentu, ktera vrati novy model
    behy = []
    for s in seedy:
        torch.manual_seed(s)
        behy.append(experiment(vytvor_model(), nazev=f"{nazev} (seed {s})", **kwargs))
    return behy

def tabulka(vysledky):
    # vysledky: {nazev nastaveni: seznam behu z opakuj}
    print(f"{'nastavení':28s} {'val přesnost':>18s} {'val ztráta':>16s}")
    for nazev, behy in vysledky.items():
        acc = np.array([b["val_acc"][-1] for b in behy])
        los = np.array([b["val_loss"][-1] for b in behy])
        print(f"{nazev:28s} {acc.mean():10.2%} ± {acc.std():.2%}"
              f" {los.mean():9.3f} ± {los.std():.3f}")

def kresli(vysledky, klic="val_acc"):
    # prumer pres seedy (cara) a rozsah min-max (stin); klic: val_acc, val_loss, train_acc, train_loss
    fig, ax = plt.subplots(figsize=(7, 4))
    for (nazev, behy), barva in zip(vysledky.items(), plt.cm.tab10.colors):
        k = np.array([b[klic] for b in behy])
        ep = np.arange(1, k.shape[1] + 1)
        ax.plot(ep, k.mean(0), color=barva, label=nazev)
        ax.fill_between(ep, k.min(0), k.max(0), color=barva, alpha=0.2)
    ax.set(xlabel="epocha", ylabel=klic, xticks=ep); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

def matice_zamen(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 2000].to(device)).argmax(1).cpu()
                          for i in range(0, len(X), 2000)])
    M = torch.zeros(10, 10, dtype=torch.long)
    M.index_put_((y, pred), torch.ones_like(pred), accumulate=True)
    mimo = M.clone().fill_diagonal_(0)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(mimo, cmap="Reds")
    for i in range(10):
        for j in range(10):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                        color="gray" if i == j else
                              ("white" if mimo[i, j] > 0.6 * mimo.max() else "black"))
    ax.set_xticks(range(10), TRIDY, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(10), TRIDY, fontsize=8)
    ax.set(xlabel="predikce", ylabel="skutečnost", title="matice záměn (barva = jen chyby)")
    plt.tight_layout(); plt.show()
    nej = mimo.flatten().topk(5)
    for pocet, k in zip(nej.values, nej.indices):
        print(f"{TRIDY[k.item() // 10]:>15s} -> {TRIDY[k.item() % 10]:15s} {pocet.item()}x")
    return pred

def ukaz_chyby(pred, y, X_raw, skutecna, predikce, kolik=8):
    # obrazky tridy `skutecna`, ktere sit oznacila jako `predikce` (cisla trid 0-9)
    i = ((y == skutecna) & (pred == predikce)).nonzero().flatten()[:kolik]
    if len(i) == 0:
        print("žádné takové chyby"); return
    fig, axes = plt.subplots(1, len(i), figsize=(1.5 * len(i), 1.9), squeeze=False)
    for ax, k in zip(axes[0], i):
        ax.imshow(X_raw[k], cmap="gray_r"); ax.axis("off")
    fig.suptitle(f"skutečnost: {TRIDY[skutecna]}, síť: {TRIDY[predikce]}", fontsize=10)
    plt.tight_layout(); plt.show()

print("síť ze cvičení má", pocet_parametru(MLP()), "parametrů")
"""),

# ---------------------------------------------------------------- 1
md(r"""
## 1) Výchozí model

Natrénujte síť ze cvičení (`MLP()`, tedy dvě skryté vrstvy po 128 neuronech)
na 5 epoch a vypište validační přesnost. Pro Fashion-MNIST čekejte zhruba
87–89 %.
"""),

code(r"""
# TODO: vychozi model. Staci jeden beh, napr.:
# torch.manual_seed(0)
# vychozi = experiment(MLP(), epochs=5, nazev="výchozí MLP")
"""),

md(r"""
**TODO:** Jednou větou: jaká je validační přesnost a jak se liší od přesnosti
na trénovacích datech?
"""),

# ---------------------------------------------------------------- 2
md(r"""
## 2) Hypotéza

Vyberte **jednu** otázku. Do buňky pod ní napište, co čekáte a proč (2–3 věty),
**dřív, než cokoli spustíte**.

1. **Šířka, nebo hloubka?** Je lepší jedna široká skrytá vrstva, nebo několik
   užších se zhruba stejným počtem parametrů? (Počet ověříte funkcí
   `pocet_parametru`, například `MLP(skryte=(256,))` proti `MLP(skryte=(160, 160, 160))`.)
   Rozdíly tu bývají malé; i „rozdíl je menší než rozptyl mezi seedy“ je
   platný výsledek, pokud ho doložíte.
2. **Velikost dávky.** Jak záleží na `batch_size` (např. 16 / 64 / 512) při
   stejném počtu epoch? Měřte i čas trénování (`time.time()` před a po);
   malé dávky jsou výrazně pomalejší, zapněte si GPU.
3. **Kolik dat stačí?** Od kolika trénovacích obrázků přesnost na validaci
   přestane znatelně růst? (Podmnožinu předáte jako `X=Xtr[:n], y=ytr[:n]`.)
4. **Regularizace při malých datech.** Pomůže dropout nebo `weight_decay`, když
   má síť jen 1 000 trénovacích obrázků?
5. **Vlastní otázka** po domluvě s vyučujícím.

Měňte jen to, na co se ptáte; ostatní nechte stejné.
"""),

md(r"""
**TODO — hypotéza.**

*Otázka:*

*Co čekám a proč:*
"""),

# ---------------------------------------------------------------- 3
md(r"""
## 3) Experiment

Každé nastavení spusťte se **3 seedy** pomocí `opakuj` a výsledky uložte do
slovníku `vysledky`. Rozptyl mezi seedy ukazuje, jak velký rozdíl je ještě
náhoda: dvě nastavení, jejichž rozsahy se překrývají, nejsou prokazatelně
různá.

`opakuj` bere **funkci**, která vytvoří nový model (každý seed musí začít
od nových vah), proto `lambda: MLP(...)`.

Hlídejte čas: 3 nastavení × 3 seedy × 5 epoch trvají na CPU v Colabu
několik minut. Na GPU (*Runtime → Change runtime type*) je to rychlejší.
"""),

code(r"""
vysledky = {}

# TODO: nastaveni podle vasi otazky. Priklad tvaru (jina otazka nez vase):
# vysledky["lr 1e-3"] = opakuj(lambda: MLP(), epochs=5, lr=1e-3, nazev="lr 1e-3")
# vysledky["lr 1e-2"] = opakuj(lambda: MLP(), epochs=5, lr=1e-2, nazev="lr 1e-2")
"""),

code(r"""
if vysledky:
    tabulka(vysledky)
    kresli(vysledky, "val_acc")
    kresli(vysledky, "val_loss")
"""),

md(r"""
### Závěrečné měření na testu

Až po vyhodnocení na validaci vyberte nejlepší nastavení a změřte **jednou**
přesnost na testu, pro něj a pro výchozí model. Podle testu už nic neměňte.

Natrénované modely jsou uložené v bězích: `vysledky["nazev"][0]["model"]` je
model se seedem 0.
"""),

code(r"""
# TODO: test jen pro vybrane nastaveni a vychozi model, napr.:
# for nazev in ["lr 1e-3", "lr 1e-2"]:
#     ztrata, presnost = vyhodnot(vysledky[nazev][0]["model"], Xte, yte)
#     print(f"{nazev:28s} test {presnost:.2%}")
"""),

md(r"""
**TODO:** Krátký komentář k tabulce a grafům: které nastavení vyšlo nejlépe,
jsou rozdíly větší než rozptyl mezi seedy? Liší se test od validace?
"""),

# ---------------------------------------------------------------- 4
md(r"""
## 4) Rozbor chyb

Vezměte nejlepší model a zobrazte matici záměn na **validačních** datech.
Pro dvě nejčastěji zaměňované dvojice ukažte chybně klasifikované obrázky.
Čísla tříd: 0 tričko, 1 kalhoty, 2 svetr, 3 šaty, 4 kabát, 5 sandál,
6 košile, 7 tenisky, 8 kabelka, 9 kotníková bota.
"""),

code(r"""
# TODO: napr.
# nejlepsi = vysledky["lr 1e-3"][0]["model"]
# pred = matice_zamen(nejlepsi, Xva, yva)
# ukaz_chyby(pred, yva, X_va_raw, skutecna=6, predikce=0)
"""),

md(r"""
**TODO:** Proč si síť plete právě tyto třídy? Pomohlo by něco z toho, co
víte o MLP (například že nevyužívá sousedství pixelů)? 3–5 vět.
"""),

# ---------------------------------------------------------------- 5
md(r"""
## 5) Závěr

**TODO:** 5–10 vět. Potvrdila se hypotéza? Opřete se o konkrétní čísla
z tabulky. Co byste zkusili dál?
"""),

md(r"""
---

### Před odevzdáním

1. *Runtime → Restart and run all.* Notebook musí proběhnout bez chyby
   a do 10 minut.
2. Projděte tabulku podmínek nahoře a zkontrolujte všechny body.
3. Pošlete odkaz s oprávněním ke čtení.
"""),

]


def main():
    out = [_cell(c.kind, c.src, i) for i, c in enumerate(CELLS)]
    path = os.path.join(REPO, 'akti', 'AKTI_zapocet_sablona.ipynb')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(_notebook(out), f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(path, 'bunek:', len(out))


if __name__ == '__main__':
    main()
