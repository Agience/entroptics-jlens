# The identity component of a Jacobian lens inflates the noise floor computed from it

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`

*Pre-print. September 2026.* [doi:10.5281/zenodo.22293929](https://doi.org/10.5281/zenodo.22293929)

---

## Abstract

A transformer's residual stream **adds** each layer's output to what came before, so the
corpus-averaged Jacobian $J_l=\mathbb{E}[\partial h_{\text{final}}/\partial h_l]$ contains a copy
of the identity by construction, and its share grows with depth as less transformation remains
above the layer. At the deepest fitted layer of the six lenses read here from models above
1B parameters it is **66% to 88% of the matrix**.

That matters for any spectral read whose threshold is derived from the frame it is judging. We read
resolved rank with Entroptics [Sessford 2026]: $K=\#\{\sigma_k>\text{floor}\}$ against a derived
finite-size Marchenko–Pastur / Tracy–Widom edge. Its variance estimate is a de-biased **median row
energy**, and an added $\alpha I$ contributes $\alpha^2$ to *every* row — the one contribution a
median over rows cannot reject. **Measured, the identity raises the floor computed from it by more than 2× on eight of the ten
models, and by up to 4.8×**, and the modes in between fall under the line.

The separation is exact and closed form. $\alpha=\operatorname{tr}(J)/d$ is the orthogonal
projection of $J$ onto $\operatorname{span}(I)$ under the Frobenius inner product, and the
**residual transport** is $M=J-\alpha I$: a trace and a subtraction, $O(d^2)$, with no threshold in
it.

Reading $M$ in place of $J$ changes the count by **2× to 21×** across all eleven lens files read
here — ten of the twenty-two on the Neuronpedia mirror — in 9 of 10 models, five families, widths
512 to 4096. **Most of that range is
the estimator's response to an added identity**, which two structure-free surrogates measure: one
holding the spectrum and $\alpha$, one holding the spectrum, $\alpha$ **and every row norm** — the
floor's own variance input, preserved to $2\times10^{-16}$. **The excess attributable to the
transport lies between $1.15\times$ and $1.87\times$**, the bracket's width being how far a
surrogate is allowed to move the statistic the floor is made of (§1.2). The correlation with
identity share behaves the same way on the surrogates ($\rho=+0.745$ real, $+0.758$ and $+0.657$
surrogate), so it too is a property of the floor.

**The directions below the raw read's floor are reproducible.** Decoded through the model's own
readout, the modes between $K(J)=25$ and $K(M)=183$ at Qwen3.5-4B layer 30 name the same vocabulary
across the catalogue's two independent fits at Jaccard $0.390$, against $0.038$ to $0.054$ for
random directions decoded the same way — roughly 7 to 10 times chance. But decoded-token agreement decays as a smooth power law in mode
index ($r^2=0.988$) with only a $+7.3\%$ excess at the floor and no knee there, so this establishes
that reproducible structure extends well past $K(J)$ — not that it stops at $K(M)$ (§1.7).

**Scope.** Below relative depth 0.6 the two reads agree. The effect lives where the identity share
passes about 0.4, which is the deep layers. Every count here is an `mp` read on a corpus-averaged
transport, an object §2.3 argues has no bulk for a derived edge to locate; the threshold-free reads
of §4 are the right instrument there, and under them the same decomposition moves the answer the
other way ($\mathrm{PR}(J)=1328$ against $\mathrm{PR}(M)=124$). Three criteria give three
reproducible dimensions for the same matrix — 183 by `mp`, $\ge 400$ by principal angles between
fits, 1383 above the measured estimator noise — and §2.3 reconciles them. The reads here are geometric: §8's matched control crosses an *unresolved* direction above a
resolved one, placing the behavioural question outside this work.

**The lens files are a correct measurement and nothing here asks for a change to them**: the
identity is genuinely part of $\mathbb{E}[\partial h_{\text{final}}/\partial h_l]$, and any
faithful Jacobian of a residual stream contains it. What §1 is about is what a threshold that
derives its own scale does when handed that matrix, which is a property of the threshold and a
step in the pipeline of whoever applies one. Nothing here inspects another author's analysis code.
Subtracting $\alpha I$ is the obvious step once the identity is noticed. The contribution is the
measurement: the size of the
identity at depth, what it does to a derived floor, how much of that survives a structure-free
control, and how far the reproducible structure below the raw floor extends.

§1 is the measurement and the controls that bound it. §2 sets out what Entroptics supplies and the
property of its floor the result turns on. §3 and §4 are the reads that survive on a
corpus-averaged transport and §5 the floor for one that carries noise, §6 what stands outside
J-space on real streams, §7 the reach of the linearisation, §8 a crossing between two models and
its controls, §9 validation against planted truth, §9a what the reads cover, §10 the
instrument's scope, §11 the work this builds on, and §12 the script and stored run behind
every figure.

---

## 1. The claim

Eleven lens files, ten of the twenty-two on the Neuronpedia mirror, at each model's deepest
fitted layer, `mp` null at $\alpha_{\text{FA}}=0.05$:

| model | $d$ | identity share | $K(J)$ | $K(J-\alpha I)$ | understated by |
|---|---|---|---|---|---|
| qwen3-1.7b | 2048 | 0.798 | 3 | 64 | **21.3×** |
| qwen3-4b | 2560 | 0.876 | 7 | 149 | **21.3×** |
| gemma-3-4b | 2560 | 0.703 | 16 | 165 | 10.3× |
| qwen3.5-4b | 2560 | 0.790 | 25 | 183 | 7.3× |
| gpt2 | 768 | 0.422 | 6 | 39 | 6.5× |
| qwen3.5-0.8b | 1024 | 0.778 | 12 | 68 | 5.7× |
| gemma-3-1b | 1152 | 0.660 | 17 | 64 | 3.8× |
| llama3.1-8b | 4096 | 0.720 | 74 | 269 | 3.6× |
| pythia-70m | 512 | 0.458 | 2 | 4 | 2.0× |
| gemma-3-270m | 640 | 0.107 | 39 | 42 | 1.1× |

An eleventh lens file is a second fit of qwen3.5-4b over 417 sequences rather than 1000; it gives
$7.28\times$ against the other's $7.32\times$, so the gain replicates across fits as well as
across models.

### 1.1 Why it holds

**Definition 1.1.** $\alpha=\operatorname{tr}(J)/d=\langle J,I\rangle/\langle I,I\rangle$; the
**residual transport** is $M=J-\alpha I$.

On Qwen3.5-4B ($n{=}1000$, $d{=}2560$, 31 layers):

| layer | $\alpha$ | off-diag rms | $\alpha/\text{rms}$ | $\lVert\alpha I\rVert_F^2/\lVert J\rVert_F^2$ | median $\sigma(J)$ |
|---|---|---|---|---|---|
| 0 | 0.181 | 0.038 | 4.8 | 0.008 | 0.097 |
| 12 | 0.280 | 0.019 | 14.6 | 0.070 | 0.131 |
| 24 | 1.037 | 0.017 | 60.3 | 0.583 | 1.096 |
| 30 | 0.994 | 0.010 | 98.9 | 0.790 | 1.052 |

At layer 30 the transport is 79% identity by energy and **its median singular value is within 6%
of $\alpha$** (1.052 against 0.994). The flat block a spectral floor reads as "the bulk" is the
identity itself.

Entroptics' `mp` provider estimates its per-cell variance from the whole matrix, signal included.
So the identity does two things at once: it flattens the spectrum, and it raises the floor that is
computed from that spectrum. The modes between the old floor and the new one disappear.

The division of labour is this.
`decompose` is not an Entroptics read — it is $\operatorname{tr}(J)/d$ and a subtraction, carrying
the *domain* fact that a residual stream adds. Entroptics supplies the *measurement*: `noise_floor`
under `null_providers.mp` is the only thing here that turns a matrix into a number, and every
figure in the table above is one of its counts. The claim is a precondition for that read, supplied
by the domain it is pointed at.

**Consequence 1.1a.** The apparent monotone decline of resolved rank with depth reverses under the
decomposition. It was the identity growing.

### 1.2 How much of this is the transport, and how much is the estimator

Spearman between the identity share and the gain is $+0.745$ over the ten models and $+0.773$ over
the eleven lens files — the eleventh is a second fit of one model, so the ten-model value is the
one to quote. A bootstrap over models puts the 95% interval at $[+0.18, +0.99]$, and width is a
partial confound at $\rho(d, \text{share})=+0.669$.

That correlation measures the floor's response to an added identity. The `mp` floor rises with any
added identity, whatever the matrix underneath, so separating the transport's contribution needs a
surrogate: a matrix with the same spectrum and the same $\alpha$ and no model in it, read exactly
as the transport is. The `mp` floor's only free input is a de-biased **median row energy**, so what
a surrogate does to that statistic decides what the comparison measures, and two surrogates bracket
it, over five draws each, median reported:

| model | share | gain | spectrum surrogate | excess | rotation surrogate | excess |
|---|---|---|---|---|---|---|
| gemma-3-270m | 0.107 | 1.08× | 1.03× | 1.04× | 1.08× | 1.00× |
| gpt2 | 0.422 | 6.50× | 2.00× | 3.25× | 5.57× | 1.17× |
| pythia-70m | 0.458 | 2.00× | 2.33× | 0.86× | 2.00× | 1.00× |
| gemma-3-1b | 0.660 | 3.76× | 2.69× | 1.40× | 4.00× | 0.94× |
| gemma-3-4b | 0.703 | 10.31× | 4.31× | 2.39× | 9.17× | 1.12× |
| llama3.1-8b | 0.720 | 3.64× | 1.97× | 1.85× | 2.10× | 1.73× |
| qwen3.5-0.8b | 0.778 | 5.67× | 3.00× | 1.89× | 4.00× | 1.42× |
| qwen3.5-4b | 0.790 | 7.32× | 4.06× | 1.80× | 4.82× | 1.52× |
| qwen3-1.7b | 0.798 | 21.33× | 3.67× | 5.82× | 21.33× | 1.00× |
| qwen3-4b | 0.876 | 21.29× | 7.88× | 2.70× | 16.56× | 1.29× |
| **median** | | | | **1.87×** | | **1.15×** |

**The two surrogates differ in one invariant, and that invariant is the floor's own input.**

  * **spectrum** — $(U s)V^{\mathsf T}$ with Haar $U$ and $V$. Holds the singular values and the
    Frobenius norm; spreads energy evenly across rows, so the median row energy moves and the floor
    moves with it. On gpt2 it raises the variance estimate $7.15\times$, against the $\pm10\%$ that
    §1.4 shows moves $K$ from 222 to 150.
  * **rotation** — $MQ$ with Haar orthogonal $Q$. Holds the singular values, $\alpha$, **and every
    row norm** — verified to $2\times10^{-16}$, so the floor's variance estimate is identical by
    construction — while destroying the alignment between the transport's input and output
    directions. It keeps the left singular vectors and with them the massive-activation row
    structure of §3.3, so it is the conservative end of the bracket.

**The excess attributable to the transport lies between $1.15\times$ and $1.87\times$**, and the
width of that bracket is how much a surrogate is allowed to move the statistic the floor is made
of. The rotation surrogate is the tighter measurement in a second sense: its excesses span
$0.94\times$ to $1.73\times$ where the spectrum surrogate's span $0.86\times$ to $5.82\times$. **Five of
the ten sit at or below $1.00\times$ under it**, so at the conservative end of the bracket half the
catalogue shows no excess over a matrix with no model in it. The rotation holds every row norm and
so the floor on the $M$ side exactly; on the raw side $\text{floor}(MQ+\alpha I)$ is within 0.1% of
$\text{floor}(J)$ on gemma-3-1b and 1.9% and 3.8% low on gpt2 and Qwen3.5-0.8B, which moves the
latter's excess from $1.42\times$ to $1.25\times$.

Two rows carry most of the spread and both are informative. gpt2 has the most localised rows in the
catalogue — a median row energy $0.079$ of the mean — and shows the largest gap between the two
surrogates ($3.25\times$ against $1.17\times$). pythia-70m has the flattest, at $0.950$, and is the
one model where the spectrum surrogate *lowers* the variance estimate; its two surrogates agree at
$0.86\times$ and $1.00\times$.

The correlation with identity share behaves the same way under both surrogates as it does on the
real transports: $+0.745$ real, $+0.758$ spectrum, $+0.657$ rotation. The association between share
and gain is a property of the floor, and the excess is what remains once it is held fixed.

The surrogate is drawn once per model per seed and the counts are stable where they are large: the
spectrum surrogate's gain is identical across all five seeds on three of the ten. pythia-70m sets
the limit on
precision, reading $K=3\to7$ or $2\to7$ depending on the draw. **A ratio needs a denominator large
enough to carry it**, which is the second reason the bottom of the headline range is quoted with
the surrogate beside it.

### 1.3 What the planted-rank check establishes

The objection is immediate: something was removed and the number grew, so on what grounds is the
larger number nearer the truth?

Rank 20 planted in $d=256$ at the fixture's seed ($100\times\text{rank}$), with $\alpha I$ added
for growing $\alpha$:

| | $\alpha{=}0$ | $0.5$ | $1.5$ | $3.0$ | $6.0$ |
|---|---|---|---|---|---|
| identity share | 0.000 | 0.032 | 0.230 | 0.543 | 0.826 |
| $K(J)$ | 20 | 20 | 19 | 16 | **6** |
| $K(J-\alpha I)$ | **20** | **20** | **20** | **20** | **20** |

**The bottom row is an algebraic identity.** With the planted matrix fixed across the sweep,
$A=B+\alpha I$ and $\alpha_{\text{est}}=\operatorname{tr}(B)/d+\alpha$, so the residual
$M=B-(\operatorname{tr}(B)/d)I$ is independent of $\alpha$ exactly. Measured across the sweep,
$\max|M(\alpha)-M(0)|$ runs $6\times10^{-17}$ to $5\times10^{-16}$: the five entries are one number
computed five times. Any read of $M$ whatsoever would produce that row, including a wrong one.

The top row is a measurement, and it is the one that carries the section: **a spectral read of $J$
loses a known rank as the identity grows**, 20 down to 6 at a share of 0.826. Over planted ranks
5–40 against five identity strengths, 20 cases, the read on $M$ recovers the planted rank exactly
in 15 of 20 (mean absolute error 2.50) against the read on $J$ in 11 of 20 (5.70), and every one of
the raw read's misses is an understatement that grows with the identity share.

Two limits on how far this transfers. The fixture is a spiked model with i.i.d. Gaussian noise and
planted values a factor of three above the Marchenko–Pastur edge — the ensemble `mp` was derived
for, which §3 argues a real transport is not. And on a *geometrically decaying* planted spectrum,
which is the shape real transports have, the read on $M$ is invariant in $\alpha$ and still well
below the planted rank. Invariance is necessary and the recovery figures supply the rest, over the
regime the fixture covers: **in a spiked-Gaussian ensemble, a read of $J$ loses planted rank as the
identity grows and a read of $M$ holds it.** Whether 183 is the right count for a real transport at depth is
a separate question, taken up in §1.7 and §2.3.

### 1.4 What the count is sensitive to

$K$ is counted against a floor at a rate the reader chooses, so the effect could be an artefact of
that choice. Swept across 5.7 orders of magnitude, every lens at its deepest fitted layer
at $\alpha_{\text{FA}}=0.05$:

| $\alpha_{\text{FA}}$ | lens files over 1.5× | range |
|---|---|---|
| 0.5 | 10/11 | 1.1× – 21.7× |
| 0.05 | 10/11 | 1.1× – 21.3× |
| 0.005 | 10/11 | 1.1× – 21.1× |
| 0.0005 | 10/11 | 1.1× – 21.0× |
| $10^{-6}$ | 10/11 | 1.1× – 20.9× |

Ten of eleven files — nine of ten models — at every setting. **This is a weaker check than it
looks**, and the reason is the geometry of the edge. Tracy–Widom
fluctuations are $O(d^{-2/3})$ relative to the Johnstone centring, so at $d=2560$ the whole sweep
moves the edge by about $1.3\%$. The sweep demonstrates that the one parameter a reader can set is
the one that cannot matter.

The quantity the count *is* sensitive to is the variance estimate $\hat\sigma$, which
`null_providers.mp` takes as a de-biased median row energy. A $\pm10\%$ change in $\hat\sigma$ moves
$K$ at Qwen3.5-4B layer 30 from 222 to 150, against the $\pm5$ the false-alarm sweep produces. The
identity acts on exactly that quantity: it adds $\alpha^2$ to every row's energy, which is the one
contribution a median over rows cannot reject. That is the mechanism of §1.1, stated as the
sensitivity it is.

Note the two halves separately. The *decomposition* takes no threshold — $\alpha=\operatorname{tr}(J)/d$
is closed form. Only the *counting* involves a false-alarm rate.

### 1.5 Where it applies

The table above is each model's deepest layer. Across **every layer of the five lenses narrow
enough to sweep whole** — gemma-3-1b, gemma-3-270m, gpt2, pythia-70m and qwen3.5-0.8b, 81 layers —
by relative depth, at $\alpha_{\text{FA}}=0.05$:

| relative depth | median identity | median gain | layers over 1.5× |
|---|---|---|---|
| 0.0–0.2 | 0.040 | 0.9× | 0 / 17 |
| 0.2–0.4 | 0.017 | 1.0× | 0 / 15 |
| 0.4–0.6 | 0.080 | 1.0× | 0 / 16 |
| 0.6–0.8 | 0.281 | 1.2× | 5 / 15 |
| **0.8–1.0** | **0.474** | **3.6×** | **14 / 18** |

The five wider lenses are read at their deepest layer only (§1), because a per-layer sweep of a
4096-wide transport costs a pair of full SVDs per layer. The profile is therefore over the narrow
half of the catalogue, and the models with the largest identity shares are not in it.

**Below relative depth 0.6 the two reads agree**, and the median gain sits at or below $1.0\times$
in all three of those bins. Above 0.8 the gain is large in 14 of 18 layers. The cut is the identity
share rather than the depth index: over these 81 layers every layer whose gain exceeds $1.5\times$
has a share between $0.348$ and $0.778$, and every layer where it does not between $0.003$ and
$0.433$. The two populations meet over $0.348$ to $0.433$ — four layers above the line and one below —
and the deepest layers of the wider models extend the upper range to $0.876$.

At low identity the effect reverses — $K$ falls slightly (203 → 178 at Qwen layer 12, identity
share 0.070). The mechanism is the estimator rather than lost signal: `mp` takes a **median** row
energy while $\alpha$ is the **mean** diagonal entry, and at shallow layers those disagree — Qwen
layer 0 has mean 0.181 against median 0.051, so subtracting $\alpha$ from a typical row increases
its energy and lifts the floor there (3.39712 to 3.39960, $K$ 94 to 93). At layer 12 the floor
falls slightly and the residual spectrum falls faster, which is the same arithmetic reaching the
count by the other term. The effect changes sign, so it is not a factor that can be applied
after the fact.
### 1.6 The scope of the claim

The published lens files carry the identity at the shares tabulated, and reading $M=J-\alpha I$ in
place of $J$ resolves more directions by the factors shown. The measurement is of those files. Any
analysis that already removes the identity is outside it, and nothing here inspects another
author's code.

Subtracting $\alpha I$ is the obvious step once the identity is noticed, and
$\operatorname{tr}(J)/d$ is the first thing anyone would write down. The contribution is the
measurement: that the identity is 66–88% of the matrix at depth, that it is worth a factor of 2 to
21, that the size tracks the identity share, that the read on $M$ is invariant on planted truth,
that the effect starts near relative depth 0.6, and that the directions it recovers reproduce
across independent fits (§1.7).

### 1.7 What the recovered directions contain

A count says how many directions there are, not what they are. Both questions have an answer here,
because a transport's left singular vectors live in the final residual basis — the space the
unembedding reads — so each one can be pushed through the model's own readout and printed as the
tokens it moves.

**They name things.** gpt2, deepest fitted layer, where $J$ resolves 6 directions and $M$ resolves
39. Taking eight of the 33 recovered modes at even spacing, so the selection is fixed before the
content is seen, and printing the full top-8 of each rather than the legible part
unpruned:

| mode | strongest tokens, unpruned |
|---|---|
| 6 | `senal`, ` pione`, and six undecodable byte fragments |
| 10 | `` ' ``, `]`, ` ...`, ` "`, ` [`, `)`, ` `, `"` |
| 15 | `.''.`, ` Malk`, *(one undecodable fragment)*, `''.`, `.」`, ` lineback`, ` Tuls`, ` challeng` |
| 19 | `theless`, ` arrang`, `basketball`, ` glim`, ` sugg`, ` Nuggets`, ` Fei`, ` misunder` |
| 24 | ` USDA`, ` Veter`, ` NB`, ` SNAP`, ` UD`, ` FALSE`, ` IB`, ` BAR` |
| 28 | *(one undecodable byte fragment)*, ` Adin`, ` pestic`, ` nutrit`, ` ingested`, ` Nieto`, ` pathogens`, ` Yosemite` |
| 33 | `Rub`, ` Huma`, ` Herrera`, `Afee`, `Tea`, ` Cuban`, ` Tanaka`, ` Iranian` |
| 38 | `SPONSORED`, `pmwiki`, `mble`, `)].`, `arine`, `nered`, ` neglig`, ` ►` |

Federal-agency abbreviations at 24, food contamination at 28, surnames and nationalities at 33,
punctuation against truncated word-stems at 10. Four of the eight read that way and four do not.
A singular direction of a corpus-averaged transport is not a monosemantic feature and nothing here
treats it as one. These rows are illustration; the measurement follows.

**Reproducibility.** The measurement is carried by a test the selection cannot bias. The catalogue publishes two fits of Qwen3.5-4B, over 1000 and 417 sequences — independent
estimates of the same transport, where a direction that is structure has to appear in both.
Decoding each fit's modes and taking the union of the 20 strongest tokens at each end gives a token
set per block, compared by Jaccard. Blocks rather than individual modes: singular vectors with
close singular values rotate freely within their subspace between two fits, so a mode-by-mode
comparison measures that rotation.

Seven consecutive blocks of 158 modes, layer 30, where $K(J)=25$ and $K(M)=183$
at $\alpha_{\text{FA}}=0.05$:

| block | 25–183 | 183–341 | 341–499 | 499–657 | 657–815 | 815–973 | 973–1131 |
|---|---|---|---|---|---|---|---|
| Jaccard | **0.390** | 0.234 | 0.178 | 0.149 | 0.132 | 0.123 | 0.118 |

**This is a single power law in mode index, and the recovered block sits on it.** Fitting
$J = c\cdot\text{mid}^{\,k}$ to the six blocks *beyond* the floor gives $k=-0.504$ at $r^2=0.988$,
and extrapolating back to the recovered block predicts $0.363$ against an observed $0.390$ — an
excess of $+7.3\%$. Comparing the recovered block against the next one gives $0.390/0.234=1.67$,
but any adjacent pair gives something similar, because the curve is smooth. **The two-block
comparison measures where the blocks sit, not whether the floor marks anything.**

The chance level also has to account for the readout. Decoding any direction through $W_U$ lands on
a privileged slice of the vocabulary, so two *unrelated* directions overlap far more than two
uniform token draws:

| null | Jaccard |
|---|---|
| uniform draw from the vocabulary | 0.008 |
| 100 random directions, decoded through the same readout, size-matched | **0.038** |
| 158 random directions, decoded | 0.054 |

Against the decode-matched null the recovered block stands at roughly $7$–$10\times$ chance. The
uniform draw would put it at $49\times$.

**What this section establishes.** Reproducible token-level structure extends well below the floor
a read of $J$ imposes: the recovered block reproduces at $0.390$ where a uniform null gives $0.008$
and a decode-matched null gives $0.038$. **The extent of that structure is bounded below by 183 and
not bounded above by anything measured here.** The curve has no knee there, and §3.2's principal-angle criterion has the two fits agreeing at
$0.999$ out to $k=400$, the largest $k$ tested. A read of $J$ is therefore a substantial
underestimate of the reproducible dimension of a deep transport and a read of $M$ a closer one,
with the upper end open.

Two premises this rests on. The two fits are treated as independent; §10 notes that the smaller may
be a subsample of the larger, and if it is, this agreement is inflated rather than conservative.
And no interval is quoted on any Jaccard here — one pair of fits gives one number per block.

**The concrete benefit.** At Qwen3.5-4B layer 30 a spectral read of $J$ reports 25 directions and a
read of $M$ reports 183, and the token evidence says the reproducible structure extends at least
that far. Whatever is built on the resolved set of a deep transport — a workspace estimate, a
capacity claim, a rank-truncated intervention basis — is built on 25 directions or on 183 depending
on one trace and one subtraction.

---

## 2. What Entroptics supplies

This work is a domain wrapper over Entroptics [Sessford 2026], installed from PyPI as
`entroptics>=0.2.1` (Apache-2.0). The division is worth being exact about, because the result in §1
is a precondition *for* an Entroptics read rather than a result that merely happens to use one.

### 2.1 What the library provides

| what | used for |
|---|---|
| `projection.noise_floor` with `null_providers.mp` | **the resolved rank.** $K=\#\{\sigma_k>\text{floor}\}$ against the derived finite-size Marchenko–Pastur / Tracy–Widom edge at a stated false-alarm rate. Every $K$ in this paper is one of its counts. |
| `projection.mode_significance` | per-mode Tracy–Widom deviates and $p$-values, which let a count be re-taken at another rate without recomputing a spectrum |
| `null_providers.floor_from_null_sampler` | the contract by which a caller supplies its own null mechanism and the library keeps the quantile — how the magnitude-preserving surrogates of §5 become providers |
| `null_providers.robust` | a Tukey upper fence on the spectrum, documented as a heuristic for heavy-tailed spectra and not a calibrated null. It carries no false-alarm rate: sweeping $\alpha_{\text{FA}}$ from $0.5$ to $10^{-6}$ leaves its floor at $2.40679$ to five decimals while `mp`'s moves. Reported alongside `mp` where the two disagree, and never as a second estimate of the same quantity. |
| `Screen`, `Projection`, `coupling` | the cross-frame reads of §8: two systems placed on one basis, with a permutation null |
| `entropy.MAD_SCALE` and the whitening | the per-channel scale estimate the floor is built on |

The library supplies the measurement. The caller supplies the knowledge that a residual stream adds — that is domain, it belongs in the caller, and it is what §1 is about.

### 2.2 The property that makes the result possible

Entroptics' floor is *derived* rather than chosen: it estimates the per-cell variance of the frame
it is handed and places the edge a noise matrix of that variance would produce. That is what makes
$K$ mean anything. A count against a threshold somebody picked would carry no false-alarm rate, and
the sweep in §1.4 would test nothing.

It is also exactly why the identity has to come off first. The floor is computed **from the matrix
it is judging**: hand it $J$ and the identity raises the variance estimate, hand it $M=J-\alpha I$
and it does not. A fixed threshold would be immune to this and would also carry no calibration.
**The sensitivity that makes the read calibrated is the sensitivity the architecture acts on**, so
supplying a frame whose guaranteed components have been removed is a precondition the caller owes
the library.

### 2.3 What bounds the measurement

Two properties of the floor set the limits of what §1 can say, and both are why this paper reports
$K$ beside a threshold-free effective rank rather than alone.

- **`mp` is a lower bound.** It estimates variance from the whole matrix, signal included, so a
  signal-dense matrix lifts its own floor. On planted ranks at $d=256$ with every mode
  supra-threshold by the BBP criterion, the count saturates well below the planted rank (§1.3). This is why the read on $M$
  in §1.3 saturates near 30 at planted rank 40: that residual is the floor's, it is present with
  or without an identity, and it stays *flat* across $\alpha$ where a contamination would grow.
- **A derived edge presumes a bulk whose edge it locates.** A corpus-averaged transport has had
  its noise averaged away and has no bulk (§3), so the threshold-free reads are the right
  instrument there and the floor belongs on single samples.

**Three criteria, three numbers, and which to use.** At Qwen3.5-4B layer 30 this paper reports
three reproducible dimensions for the same matrix $M$, and they do not agree:

| criterion | dimension | what it answers |
|---|---|---|
| `mp` floor at $\alpha_{\text{FA}}=0.05$ (§1) | **183** | how many modes clear a derived Gaussian edge |
| principal angles between two fits, $\ge0.999$ (§3.2) | **$\ge 400$** | how many directions two independent fits agree on |
| above the measured estimator noise (§3.1) | **1383** | how many exceed the noise the fits themselves show (996 if the smaller fit is a subsample of the larger, §10) |

They are not competing estimates of one quantity. `mp` asks whether a mode could have come from a
noise matrix of the frame's own variance — a question whose premise §3 undercuts on this object,
since the corpus average has removed the noise the edge is meant to locate. The other two ask what
survives re-estimation, which is the question a reader building on a lens actually has. **The
threshold-free reads and the two-fit criteria are the ones to build on; $K$ is reported because the
identity's effect on it is the subject of §1, and the two-fit criteria are what a dimension should
be read from.**
The spectrum at that layer is a smooth gapless decay — 39 singular values lie in $[0.85, 0.95]$
alone, and the floor lands at 0.895 — so $K$ is a smooth function of where the bar falls
(271 at $0.8\times$ the floor, 183 at $1.0\times$, 111 at $1.25\times$) rather than a detection.

### 2.4 What the wrapper adds

One line of arithmetic and the knowledge of when to apply it: $\alpha=\operatorname{tr}(J)/d$, the
transports read one at a time at float64 from a float16 checkpoint (`io`), the identity share as an
$O(d^2)$ screen so a caller pays for a spectrum only where it changes the answer, and the
measurement that says it is worth 2× to 21× at depth and nothing below relative depth 0.6.

---

## 3. The corpus average removes the estimator noise

$J_l$ is an expectation over a corpus. Averaging suppresses the incoherent part, so the object handed to a detection threshold may have had its noise removed before the threshold ever sees it. It has.

### 3.1 The estimator noise, measured

Two fits of Qwen3.5-4B are published: $n_A=1000$ and $n_B=417$. For independent means, $\mathrm{sd}(A-B)/\mathrm{sd}(A)=\sqrt{1+n_A/n_B}=1.843$, so their difference gives the estimator noise with no fitting and no distributional assumption.

| layer | $\lVert A-B\rVert/\lVert A\rVert$ on $J$ | $\operatorname{corr}(A,B)$ on $J$ | noise floor on $M$ | $K(M)$ vs that floor |
|---|---|---|---|---|
| 0 | 0.416 | 0.920 | 13.24 | 11 |
| 12 | 0.146 | 0.990 | 1.542 | 218 |
| 24 | 0.042 | 0.9991 | 0.415 | 1474 |
| 30 | 0.017 | **0.9999** | 0.144 | 1383 |

By layer 30 two independently fitted transports agree to four decimal places, and roughly 1400 of 2560 directions stand above the estimator noise.

### 3.2 Reproducible dimension, by principal angles

A scalar floor summarises that with one number; principal angles give it per dimension. Following the stability construction of [Scanu et al. 2026, eq. 6] — principal angles between dominant eigenspaces under a controlled perturbation, with *fit* in place of noise level — the mean cosine of principal angles between the two fits' top-$k$ right subspaces of $M$:

| layer | $k{=}1$ | $k{=}10$ | $k{=}50$ | $k{=}100$ | $k{=}200$ | $k{=}400$ |
|---|---|---|---|---|---|---|
| 0 | 0.879 | 0.863 | 0.937 | 0.927 | 0.911 | 0.859 |
| 6 | 0.970 | 0.979 | 0.984 | 0.986 | 0.978 | 0.970 |
| 12 | 0.981 | 0.974 | 0.981 | 0.985 | 0.986 | 0.991 |
| 24 | 0.997 | 0.996 | 0.992 | 0.994 | 0.993 | 0.995 |
| 30 | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | **0.999** |

**Consequence 3.2a.** There is no $k$ within the tested range at which the two fits stop agreeing. The reproducible dimension is $\ge 400$ at every layer from 6 upward. Layer 0 is the exception, degrading to $0.86$–$0.94$, consistent with its $\lVert A-B\rVert/\lVert A\rVert=0.42$.

### 3.3 What a permutation control shows

A distribution-free control locates the behaviour. Permuting $J$'s entries destroys all rank structure while preserving the entry distribution exactly, so a Tracy–Widom floor calibrated for that ensemble would resolve nothing. It resolves this:

| gpt2 layer | excess kurtosis | $K$ real | $K$ shuffled | $K$ Gaussian |
|---|---|---|---|---|
| 0 | 81 | 71 | 8 | 0 |
| 5 | 709 | 41 | 18 | 0 |
| 10 | $7.4\times10^{4}$ | 6 | **43** | 0 |

At layer 10 the real transport resolves *fewer* modes than its own entry-shuffled version. The floor is calibrated for a Gaussian ensemble and a real transport is not one: the largest entry is $455\times$ the rms.

The tails are a handful of coordinates. gpt2 layers 5 and 10 both attain their maximum at $(447,138)$; Qwen layers 0 and 12 both at the **diagonal** entry $(510,510)$, layer 30 at $(795,795)$. The top 1% of rows carry 49% of the energy at gpt2 layer 10. These are the residual stream's massive-activation dimensions, inherited by the transport.

So an entry shuffle answers a different question: it spreads those coordinates uniformly and gives the control a bulk the transport itself does not have. §3.1 gives the reason a threshold has little to do here at all.

**The inversion is mostly the identity.** Heavy tails are part of it, and the identity is the larger part. A permutation moves the diagonal off the diagonal, so it also destroys the identity component §1.1 establishes is there — on the control side only, while the real side keeps it holding its own floor up. Reading both sides on $J-\alpha I$ separates the two explanations:

| gpt2 layer | $K(J)$ | $K(\text{shuf } J)$ | $K(J-\alpha I)$ | $K(\text{shuf } J-\alpha I)$ |
|---|---|---|---|---|
| 7 | 24 | 25 | 48 | 13 |
| 8 | 12 | 28 | 53 | 15 |
| 9 | 6 | 34 | 46 | 21 |
| 10 | 6 | 43 | 39 | 44–51 |

The inversion appears at **four** consecutive layers on $J$, not one, and after the decomposition it is gone at 7–9 and emphatic in the expected direction — layer 9 moves from 6-against-34 to 46-against-21, with the shuffle stable at 20–24 over 8 seeds.

Layer 10 survives, and there the original explanation is closer: $\mathrm{PR}(J-\alpha I)=1.5$, so the transport is nearly rank-one and a single dominant direction concentrates energy exactly as an identity does. Both are one mechanism — **the `mp` floor estimates its per-cell variance from the matrix it is judging, so anything that concentrates energy raises the bar that same matrix must then clear.** A control that removes structure can therefore read higher than the real thing, and that is the floor moving rather than evidence about the transport. The rule that follows is that a null must preserve what the architecture guarantees, and an entry shuffle does not preserve a residual stream's identity.

---

## 4. Threshold-free reads of a corpus-averaged transport

Participation ratio $\mathrm{PR}=\left(\sum_k s_k^2\right)^2/\sum_k s_k^4$ and Shannon effective rank $H_2=2^{\,\mathrm{H}(p)}$ with $p_k=s_k^2/\sum s^2$ require no null and no threshold. On Qwen3.5-4B, all 31 layers:

| | L0 | L6 | L12 | L18 | L24 | L26 | L28 | L30 |
|---|---|---|---|---|---|---|---|---|
| $\mathrm{PR}(J)$ | 12.4 | 69.3 | 88.7 | 240.3 | 987.0 | 1115.2 | 1249.1 | 1328.5 |
| $\mathrm{PR}(M)$ | 12.3 | 67.8 | 80.3 | 163.2 | 326.1 | **345.0** | 262.6 | 124.4 |
| $H_2(M)$ | 47.3 | 156.8 | 267.2 | 529.5 | 843.3 | 836.7 | 692.0 | 382.3 |

$\mathrm{PR}(J)$ climbs monotonically — that is the identity growing, and it shows no band. $\mathrm{PR}(M)$ traces an arc peaking at **layer 26 of 31** and collapsing at both ends; $H_2(M)$ peaks at layer 25. Scaled to a 100-layer model, layer 26/31 is $\approx$ layer 84. The band reported by [Anthropic 2026] spans layers 38–92, which is over half that network, so a single rescaled point landing inside it is a consistency check rather than a corroboration.

**The identity removal is what reveals the band.**

### 4.1 The peak's position across the catalogue

Run across the published catalogue, plotted against relative depth so models of different depth compare:

| model | $d$ | layers | peak | rel. depth | peak $\mathrm{PR}(M)$ | final id. energy |
|---|---|---|---|---|---|---|
| gemma-3-270m | 640 | 17 | 0 | **0.00** | 75.0 | 0.107 |
| pythia-70m-deduped | 512 | 5 | 0 | **0.00** | 8.9 | 0.458 |
| gpt2-small | 768 | 11 | 2 | 0.20 | 63.9 | 0.422 |
| gemma-3-1b | 1152 | 25 | 13 | 0.54 | 30.4 | 0.660 |
| qwen3-1.7b | 2048 | 27 | 19 | 0.73 | 15.4 | 0.798 |
| qwen3-4b | 2560 | 35 | 25 | 0.74 | **24.6** | 0.876 |
| llama3.1-8b | 4096 | 31 | 23 | 0.77 | 352.7 | 0.720 |
| gemma-3-4b | 2560 | 33 | 26 | 0.81 | 29.7 | 0.703 |
| qwen3.5-0.8b | 1024 | 23 | 18 | 0.82 | 140.2 | 0.778 |
| qwen3.5-4b | 2560 | 31 | 26 | 0.87 | **345.0** | 0.790 |

Spearman rank correlations, mid-ranks for ties, over the eleven lens files the script reads: peak
relative depth against $d_{\text{model}}$ $\rho=+0.729$, against layer count $\rho=+0.664$; peak
$\mathrm{PR}(M)$ against $d_{\text{model}}$ $\rho=+0.447$, against layer count $\rho=+0.183$. Four
of these files share $d_{\text{model}}=2560$, two of them fits of one model, so the sample is
pseudo-replicated and the ties matter: ranking without averaging them reports $+0.545$ where the
third of these is $+0.447$. Two models peak at layer 0, which is the boundary, so peak relative
depth is censored from below and its association with width is inflated by that.

**The peak's position tracks scale; its height tracks generation and family.** Two facts make that concrete:

- The gemma-3 family does not grow: $75.0\to30.4\to29.7$ from 270m to 1b to 4b, while its peak marches $0.00\to0.54\to0.81$. The 1b/4b difference is 2.3%, inside the fit noise established below.
- **The peak's height separates the two Qwen generations more than the two sizes within each.** One observation per cell, and the generations differ in training data, tokenizer and architecture together, so this orders the four checkpoints without identifying which factor does it:

  | generation | model | $d$ | peak $\mathrm{PR}(M)$ | peak rel. depth |
  |---|---|---|---|---|
  | Qwen3 | qwen3-1.7b | 2048 | 15.4 | 0.73 |
  | Qwen3 | qwen3-4b | 2560 | 24.6 | 0.74 |
  | Qwen3.5 | qwen3.5-0.8b | 1024 | 140.2 | 0.82 |
  | Qwen3.5 | qwen3.5-4b | 2560 | 345.0 | 0.87 |

  The generations differ by $12.1\times$ in mean peak effective rank, and qwen3.5-0.8b at $d=1024$ exceeds qwen3-4b at $d=2560$ by $5.7\times$ -- higher rank at 2.5× less width. Peak depth is consistent within generation (0.73/0.74 against 0.82/0.87) and shifts between them. Qwen3-4B is also the most identity-dominated transport measured, at final identity energy 0.876.

**An error bar, for free.** The catalogue publishes two fits of Qwen3.5-4B. At the same model and the same layer they give $\mathrm{PR}(M)=345.0$ ($n=1000$) and $329.4$ ($n=417$) — a 4.5% spread from the fit alone. Differences below roughly 5% are not measurements.

**Consequence 4.1a.** An interior peak in $\mathrm{PR}(M)$ is present in **8 of the 10 models**, including gpt2 at 124M parameters, where it sits at relative depth 0.20. The two without one are pythia-70m and gemma-3-270m, and both take their maximum at layer 0 — a boundary, which a profile cannot show an interior peak at, and pythia-70m has five fitted layers in total. **What tracks scale is the peak's position rather than its presence**: relative depth against $d_{\text{model}}$ gives $\rho=+0.729$. Its height separates families and generations more strongly than sizes. Width, depth, family and generation covary across these ten models and no single driver is identified here. Peaked depth profiles of this kind are reported for much smaller models by [Ansuini et al. 2019] and [Valeriani et al. 2023] (§11).

---

## 5. Choosing a floor for a transport that does carry noise

For an object that does carry noise, the floor must be distribution-free and its null must preserve what is not in question. Three surrogates, grouped by what each holds fixed:

- **`sign_flip`** — $\lvert J\rvert$ entrywise, exactly. Every row energy, every column energy, the Frobenius norm and the whole magnitude distribution survive; only sign coherence dies. The tightest entropy-matched null available for a dense operator.
- **`within_row_shuffle`** / **`within_col_shuffle`** — each line's multiset, hence that line's energy. Cross-line alignment dies.

The floor is the $(1-\alpha)$ quantile of the top singular value over draws, through the library's own sampled-null contract [Sessford 2026, `null_providers.floor_from_null_sampler`]: the caller owns the null mechanism, the library owns the quantile.

**A floor has two properties.** A floor must be checked on draws it was *not* fitted to; scoring against its own quantile sample returns $\alpha$ by construction and tests nothing. On held-out draws:

- **valid** — exceedance $\le\alpha$. The one-sided promise a detection threshold must keep. Exceeding the nominal rate invalidates every count taken against the floor.
- **calibrated** — exceedance $\approx\alpha$. Strictly stronger, and often false at a few hundred draws.

Measured on a planted rank-6 transport at 200 draws: exceedance $0.005$ against a nominal $0.05$, and all 6 planted modes still resolved. The empirical $(1-\alpha)$ quantile of a right-skewed null is a high-variance estimator that lands high more often than low, so the floor comes out conservative by an order of magnitude. That costs sensitivity and never validity, which is why the two properties are kept apart.

**Where the surrogates apply.** Preserving the magnitude profile *exactly* means a massive-activation cell appears in every draw, and one such cell carries a large top singular value by itself — so the floor climbs above the transport's real structure. On the clean planted rank-6 above every surrogate is right. Setting **one** cell of that same matrix to 400 against an rms of 1.12:

| surrogate | $K$ clean | $K$ with one outlier cell |
|---|---|---|
| `sign_flip` | 6 | 1 |
| `within_col_shuffle` | 6 | 1 |
| `within_row_shuffle` | 6 | 0 |
| `mp` (analytic) | 6 | 7 |

A real transport is the second column, and removing the identity does not help: on gpt2's $J-\alpha I$ the largest cell runs from $70\times$ the rms at layer 0 to $599\times$ at layer 10, at $(447,138)$ from layer 5 down. `sign_flip` reads $K = 2, 1, 1, 1$ across layers 0/5/9/10 where `mp` reads 67, 51, 46, 39.

The two nulls move in opposite directions on a real transport: `mp` counts high because it estimates its variance from the matrix it is judging and a heavy-tailed spectrum sets that estimate low, while the sampled surrogates count low because one preserved cell sets their floor high. A winsorised surrogate would close the gap and would be fitting, so none is proposed. §4's threshold-free reads remain the recommendation for a corpus-averaged transport, and this section's floors apply where energy is not dominated by a handful of cells.

`robust`, the library's other closed-form provider, is documented there as "a Tukey upper fence … not a calibrated null" and is not a candidate for this role. It carries no false-alarm rate — sweeping $\alpha$ from 0.5 to $10^{-6}$ leaves its floor at 2.40679 to five decimals while `mp`'s moves — and on gpt2 it reads *fewer* modes than `mp` from layer 2 up, the gap widening with excess kurtosis.

---

## 6. What stands outside J-space

### 6.1 gemma-3-1b, where the transport overlaps the stream's resolved set

gemma-3-1b is the first of three models measured where the complement falls substantially below the stream's full resolved set (sec 6.2 has the other two). In its early layers the transport's directions lie *inside* that set:

| layer | K(stream) | K(complement) | removed | carried directions | outside% | J-space energy |
|---|---|---|---|---|---|---|
| 0 | 26.1 | 16.2 | **9.9 $\pm$ 5.7** | **12.0** | 62.2% | 0.04% |
| 1 | 25.1 | 18.6 | 6.5 $\pm$ 1.9 | 9.0 | 74.1% | 0.04% |
| 3 | 22.4 | 19.0 | 3.4 $\pm$ 1.1 | 5.0 | 84.9% | 0.01% |
| 9 | 22.1 | 21.2 | 0.9 $\pm$ 0.8 | 16.8 | 96.0% | 0.02% |

8 wikitext-2 `test` sequences; the spread on *removed* is the standard deviation across them.


At layer 0 the complement removes $9.9\pm5.7$ modes against 12.0 directions carried: the count
removed is of the order of the transport's carried rank, agreeing to within a standard error.

**The agreement is at the precision of eight prompts.** The per-prompt removals are
$11, 10, 16, 11, 8, 5, 18, 0$ -- a standard deviation of 5.7 over a range from none to eighteen --
while the carried count is $12$ on every prompt of the first sample and $11$ on one prompt of the
second. The carried count reads the transported frame $HJ^{\mathsf T}$, so it is stream-dependent
too, and far steadier than a difference of two mode counts. Two disjoint samples of eight sequences,
the first eight of the split and the next eight, give **9.9 and 9.8** against carried ranks of 12.0
and 11.9: they agree with each other, and both sit below the carried rank. The relationship is an
order-of-magnitude one and the sample supports it at that resolution.

The size of the effect is what the section rests on. The ordering at layer 0 runs three ways: gemma removes
of order ten, Qwen3.5-0.8B removes $3.38$ (per-prompt 3, 3, 3, 3, 4, 3, 5, 3 — a standard deviation
of 0.7, tighter than gemma's), and gpt2, pythia-70m, Qwen3-1.7B and Qwen3.5-4B remove essentially
nothing despite carrying 5 to 21 directions. The overlap decays with depth and is gone by layer 9, where gemma
rejoins the pattern.

**A note on precision.** Every number here is a mean over prompts, and the spread behind them is
uneven. `outside%` carries a per-prompt standard deviation of 2.6-4.0 points on the three models
that sit near 100%, so eight prompts state it comfortably. gemma-3-1b is the noisiest model
measured -- its `K(stream)` has a standard deviation of 7.45 about a mean of 26.1 -- and *removed*
is a difference of two such counts, so it inherits more relative noise than either. Across all six
models and every quantity reported, that difference is the only one whose spread rivals its value.

**Consequence 6.1a.** This is the converse of sec 6.5 and completes it. There, a transport captured 12-42% of the stream's *energy* and none of its resolved structure. Here, gemma's transport captures 0.04% of the energy and 38% of the resolved structure. Energy share and structural overlap come apart in both directions, which is what a variance share cannot settle on its own.

### 6.2 Replication across six models

Same protocol on pythia-70m-deduped (GPTNeoX, d=512, 5 fitted layers) and Qwen3.5-0.8B (d=1024, 23 fitted layers). Qwen3.5-0.8B carries both the interior $\mathrm{PR}(M)$ peak of sec 4.1, at relative depth 0.82, and an interior etendue peak, so it tests the read where workspace-like structure exists rather than only where it does not.

| | gpt2 (d=768) | pythia-70m (d=512) | Qwen3.5-0.8B (d=1024) |
|---|---|---|---|
| interior etendue band | no | no | **yes** |
| K(stream) | 17.8-22.1 | 14.8-16.8 | 12.2-22.8 |
| outside% | 97.7-100.0% | 98.4-101.6% | **85.2-141.8%** |
| J-space energy share | 2.8-9.2% | 0.5-6.4% | 9.5-19.9% |

All three rows at 8 wikitext-2 `test` sequences of 128 tokens, transport truncated at the
participation ratio of $J-\alpha I$, with the truncation and its pseudo-inverse taken from one SVD
as a matched pair (sec 6.5). Qwen3.5-0.8B's energy share reproduces to the digit under the matched
pair; gpt2's and pythia's move, because their participation-ratio truncations are small enough to
fall in the regime where a general pseudo-inverse fails.

The complement result holds on gpt2 and pythia-70m, whose complements remove a signed mean of $+0.09$ and $0.00$ resolved modes per layer. On Qwen3.5-0.8B it removes $+0.74$, and the ratio spans 85.2% to 141.8%. The six-model extension below sets that beside the other transports.

Resolved dimension grows with width, sublinearly:

| model | $d$ | mean K(stream) | K/$d$ |
|---|---|---|---|
| pythia-70m | 512 | 15.7 | 0.031 |
| gpt2 | 768 | 19.3 | 0.025 |
| Qwen3.5-0.8B | 1024 | 18.7 | 0.018 |
| Qwen3-1.7B | 2048 | 26.8 | 0.013 |

The rank correlation with width is $+1.000$ over these four points and the log-log slope is $+0.351$ -- sublinear, nearer $\sqrt{d}$ than flat. A fourfold range of width gives a 1.7-fold change in resolved dimension. The 128-position ordered axis bounds the count at 128 and none of these approaches it, so the growth is a property of the stream rather than of the window -- but it is growth, not invariance.

**The band and the carrying capacity are different curves.** On Qwen3.5-0.8B the transport's own effective rank rises to its peak at layer 18, while the number of directions it actually resolves on a stream peaks early -- 21.8 at layer 2 -- and falls monotonically to 13.0 by layer 22. The etendue match traces its own arc, peaking at 0.545 around layer 6 on the four-sequence draw. Three quantities that might have been expected to move together do not, and with four prompts this is an observation rather than a result.

**Extended to six models.** Qwen3.5-4B (d=2560, 31 fitted layers) was read with its forward passes on a GPU and its spectral reads locally, the two halves verified identical on a smaller model first:

| model | $d$ | prompts | outside% | J-space energy share | interior etendue peak |
|---|---|---|---|---|---|
| pythia-70m | 512 | 8 | 98.4-101.6% | 0.5-6.4% | no |
| gpt2 | 768 | 8 | 97.7-100.0% | 2.8-9.2% | no |
| Qwen3.5-0.8B | 1024 | 8 | **85.2-141.8%** | 9.5-19.9% | yes |
| gemma-3-1b | 1152 | 8 | 62.2-101.7% | 0.0-0.2% | yes |
| Qwen3-1.7B | 2048 | 8 | 99.1-100.9% | 0.0-1.1% | yes |
| **Qwen3.5-4B** | **2560** | **6** | **78.8-162.1%** | **0.7-14.6%** | **yes** |

Every row is computed with the truncation and its pseudo-inverse taken from one SVD (sec 6.5), at
the participation-ratio rank, on wikitext-2 `test` sequences of 128 tokens. Qwen3.5-4B carries 6 prompts where the rest carry 8.

**Three of the six transports reach into the stream's resolved set, and the energy share does not
say which.** Per layer, $\Delta K = K(\text{stream}) - K(\text{complement})$ is positive where the
transport's removal costs the stream resolved modes and negative where projecting it out unmasks
others — sec 6.5 notes that this is not a partition. Both the signed mean and the mean magnitude
are given, because they answer different questions:

| model | J-space share | mean $\Delta K$ | mean $\lvert\Delta K\rvert$ | layers $+/-$ | outside% |
|---|---|---|---|---|---|
| pythia-70m | 6.4% | 0.00 | 0.15 | 2 / 2 | 98.4-101.6% |
| Qwen3-1.7B | 1.1% | +0.02 | 0.08 | 10 / 5 | 99.1-100.9% |
| gpt2 | 9.2% | +0.09 | 0.09 | 4 / 0 | 97.7-100.0% |
| Qwen3.5-0.8B | 19.9% | +0.74 | 2.42 | 15 / 8 | 85.2-141.8% |
| gemma-3-1b | **0.2%** | **+1.75** | 1.84 | 17 / 5 | 62.2-101.7% |
| Qwen3.5-4B | 14.6% | **-0.57** | 3.88 | 16 / 15 | 78.8-162.1% |

**gemma-3-1b carries the lowest energy share in the table and the largest signed removal**, while
gpt2 at forty-six times its share removes 0.09. That crossing is the result, and it holds under either
statistic: Consequence 6.1a measured across six models rather than argued from two. A rank
correlation between share and removal is not reported, because it runs $+0.543$ against the
magnitude and $-0.257$ against the signed mean and so carries nothing.

The two Qwen3.5 models behave the same way as each other and differently from gemma: they remove
through the first half of the network and unmask through the second, flipping sign at layer 15 and
layer 18. On Qwen3.5-0.8B that is the identity-jump layer sec 6.3 singles out. Two draws of 8 and
12 prompts give 85.2-141.8% and 85.3-146.5%.

The complement result holds on three of the six models — pythia-70m, gpt2 and Qwen3-1.7B — where
the transport's removal leaves the stream's resolved set intact to within a tenth of a mode on
the signed mean. On the other three it does not: gemma-3-1b throughout its early layers, and the
two Qwen3.5 models in the first half of the network. **What separates the two groups is not the
energy share**, which spans 1.1% to 9.2% among the three where the result holds and 0.2% to 19.9%
among the three where it does not, nor the presence of the interior band of sec 4.1, which five of
the six carry. Six models is too few to identify what does.

In Qwen3.5-4B the etendue peak (layer 26) falls exactly on the peak of the transport's own effective rank (layer 26). That happens in one of the four banded models -- Qwen3.5-0.8B is 18 against 6, Qwen3-1.7B 19 against 4, gemma-3-1b 13 against 8 -- so the alignment is coincidental.

**Insensitive to the layer pairing.** For pythia the alignment statistics cannot separate offset 0 from offset +1 -- top-1 agreement is 0.176 against 0.168, because the model is too weak for confident-position agreement to discriminate (cosine again prefers the degenerate +2, which top-1 rejects at 0.00). It does not matter: run at offset 0 the same reads give outside% of 98.0-102.2% against 97.9-102.2% at offset +1. On Qwen3.5-0.8B the statistics do separate them, and they corroborate the documented pairing (top-1 0.81 at +1 against 0.76 and 0.42).

### 6.3 An etendue band on real streams

On Qwen3.5-0.8B the etendue match -- the ratio of the two sides' phase space, and so the share of the stream's phase space the transport can carry -- traces a band across depth:

| layers | 0-3 | 4-14 | 15 | 16-22 |
|---|---|---|---|---|
| etendue match | 0.17-0.33 | **0.36-0.52** | 0.24 | 0.25 down to 0.08 |

It peaks at **0.524 at layer 6** on the twelve-sequence draw, holds a plateau through layer 14, then falls **0.393 to 0.236 in a single step at layer 15** -- the largest single-layer drop in the profile by a factor of two over the next -- and declines to 0.080 at the last layer. A middle band with collapse at both ends, on real streams, from an invariant with no threshold in it.

**Reproducible across prompt samples.** Two independent draws (4 and 12 wikitext sequences) give the same etendue profile: $r=0.9993$ on the etendue match with a mean absolute difference of 0.0097, and $r=0.9935$ on the carried-direction count. What those two draws establish is the etendue profile; the complement on this model is in sec 6.2, where two draws of 8 and 12 prompts give 85.2-141.8% and 85.3-146.5%.

The layer-15 step is visible in the transport alone: identity energy jumps $+0.152$ there against a next-largest single-layer change of $+0.075$ in that model. These are not independent witnesses -- more identity leaves less non-identity structure to carry, so the etendue drop is downstream of the identity jump. What the two together establish is that the transition is sharp and localized rather than gradual.

**It is a property of this model.** Across ten models the largest single-layer identity jump sits at median relative depth 0.68, which looks like a shared transition until the ratio of the largest jump to the second largest is checked: 1.06-1.40 for nine of the ten, where identity energy grows smoothly and the argmax marks nothing. Only Qwen3.5-0.8B (2.02) has a distinctive one. The clustering is an artefact of taking an argmax over a smooth curve.

**The etendue band tracks the transport's effective rank.** Run across all layers of five models in three families:

| model | family | interior band in $\mathrm{PR}(M)$ | interior etendue peak |
|---|---|---|---|
| pythia-70m | GPTNeoX | no | **no** -- strictly monotone, peak at layer 0 |
| gpt2 | GPT2 | yes -- peak at relative depth 0.20 | **no** -- strictly monotone, peak at layer 0 |
| qwen3-1.7b | Qwen | yes | yes -- peak at relative depth 0.15 |
| Qwen3.5-0.8B | Qwen | yes | yes -- peak at relative depth 0.27 |
| gemma-3-1b | Gemma | yes | yes -- peak at relative depth 0.33 |

pythia-70m, whose transport shows no interior peak, has a strictly monotone etendue profile, and the three models with an interior peak in $\mathrm{PR}(M)$ also show an interior etendue peak. gpt2 has an interior peak in $\mathrm{PR}(M)$ at relative depth 0.20 and a monotone etendue profile, so the association holds on four of the five and gpt2 is the exception to it. Both Qwen models are banded and both non-banded models are non-Qwen, so family and bandedness were perfectly confounded until gemma-3-1b -- banded and not Qwen -- separated them. The association is with the transport's rank, across three families.

The peak's *height* is another matter: qwen3-1.7b reaches only 0.065 against Qwen3.5-0.8B's 0.524, a ratio of nearly eight, and its peak/first ratio of 18.1 comes off a base of 0.0036. A large ratio on a negligible base is not a large band.

**Three bands, three positions.** The transport's own effective rank peaks at layer 18 (sec 4.1), the etendue match at layer 6, and the count of directions actually carried declines monotonically from layer 2. Three depth profiles of the same transport, three different shapes. Whatever the interior band in $\mathrm{PR}(M)$ is, it is not carrying capacity on streams.

### 6.4 The share of stream energy inside J-space

`certify`'s residual is a relative Frobenius norm, so $1-\text{residual}^2$ is the share of stream
energy lying inside J-space -- the quantity [Anthropic 2026] reports as 6-10% of activation
variance. The identity holds while the round trip contracts, which requires the truncation and its
pseudo-inverse to be built from one SVD as a matched pair (sec 6.5).

Measured over every fitted layer, 8 wikitext-2 `test` sequences of 128 tokens:

| model | rank | rank/$d$ | median share | per-layer range |
|---|---|---|---|---|
| gpt2 | 5 | 0.7% | 4.8% | 1.1-5.4% |
| gpt2 | 20 | 2.6% | 5.3% | 2.2-5.8% |
| gpt2 | 100 | 13.0% | **6.6%** | 3.4-13.5% |
| pythia-70m | 50 | 9.8% | **6.7%** | 2.5-14.9% |
| pythia-70m | 100 | 19.5% | 10.6% | 5.2-21.8% |
| pythia-70m | 200 | 39.1% | 20.3% | 11.8-41.5% |

**At a transport rank near 10-13% of $d_{\text{model}}$ the two models agree on a median share of
6.6% and 6.7%**, both inside the reported band, from different architectures at different widths.
The share rises monotonically with rank on both, so the rank at which the band is reproduced is a
property of the truncation rather than a coincidence of either model.

The per-layer spread is wider than the median suggests -- 3.4-13.5% on gpt2 -- and the extremes sit
at the ends of the network, where the transport is closest to the identity (sec 1.1) and where its
resolved rank is smallest (sec 6.5). A median over layers is the statistic that compares with a
single reported figure; the range is what a per-layer read gives.

The 6.6% and 6.7% medians sit inside the 6-10% band [Anthropic 2026] report, from two
architectures at different widths.

### 6.5 The transport captures energy

The two quantities come apart, and this is the sharpest form of the result. At pythia rank 200
J-space holds **12-42% of stream energy** -- and the complement still keeps **99-102%** of the
stream's resolved modes. Raising the rank raises the captured energy monotonically and leaves the
captured *structure* at essentially zero.

The mechanism is the transported frame's own dimensionality: it resolves 1-8 directions at
pythia and 6-21 at gpt2, and those counts barely move with rank. The transport's energy is
concentrated in a few dominant directions -- the massive-activation coordinates the same
transports carry as their heaviest entries (sec 3.3) -- while the stream's resolved-mode structure
lies elsewhere.

**Consequence 6.5a.** A variance share cannot settle the workspace question, and this is why. Six
to ten percent of the variance is compatible with capturing none of the stream's resolved
structure, because energy and resolved dimension are different measurements of a frame. The
complement is not small under either.

**Read the counts.** Several entries above exceed 100%: projecting directions out
of a frame can unmask others, so this is not a partition and the ratio is not bounded by one.

At full rank the certificate degenerates -- residual of order round-off, which the screen's
per-channel whitening lifts into spurious modes -- so certify requires a genuine truncation
while the complement read does not.

**And it degenerates at the other end, for an arithmetic reason.** The round trip is
$H(J_K^+J_K)^{\mathsf T}$, so its residual is bounded by 1 whenever $J_K^+J_K$ is the orthogonal
projector onto $J_K$'s row space. Forming $J_K^+$ with a general pseudo-inverse routine breaks
that. The reconstruction $(U_k\Sigma_k)V_k^{\mathsf T}$ carries float-noise singular values past
rank $k$; a cutoff taken relative to the largest singular value leaves them in on a spectrum
spanning decades, and inverting them amplifies. Measured on gpt2 at the rank the participation
ratio selects:

| layer | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| rank | 23 | 17 | 12 | 7 | 3 | 2 |
| top singular value of $J_K^+J_K$ | 1.00000 | 1.08224 | 1.18513 | 1.00780 | **1.31455** | 1.21965 |

An orthogonal projector has top singular value exactly 1, so from layer 6 down the round trip
amplifies, and at layer 9 the certificate returned a residual of 1.038 -- an energy "share" of
$-7.8\%$. Building $J_K^+ = V_k\Sigma_k^{-1}U_k^{\mathsf T}$ from the same factors that build the
truncation restores the projector exactly at every layer, from one SVD rather than two. The
published figures above are unaffected: pythia at rank 100 and 200 returns a top singular value of
$1.00000$ at every layer.

### 6.6 The etendue ceiling

`Screen.transfer` settles a crossing by comparing the two sides' etendue, which bounds how many directions the receiving side's phase space can carry -- a ceiling, independent of how many are observed occupied. On gpt2:

| layer | modes_to | etendue match | tau |
|---|---|---|---|
| 0 | 19.0 | 0.170 | 0.170 |
| 5 | 7.3 | 0.005 | 0.005 |
| 10 | 2.0 | 0.001 | 0.001 |

The match is the ratio of the two etendues, and it collapses across depth: the transport's phase space is 17% of the stream's at layer 0 and 0.1% by layer 10. `tau` -- the fraction of the sender's energy that fits the receiver's etendue -- follows it exactly.

**Consequence 6.6a.** The transport is etendue-limited, and the limit is not a property of what the stream happens to contain. `modes_to` reproduces the transported frame's resolved direction count (19.0, 7.3, 2.0 against 19.0, 7.3, 2.0), so the low dimensionality of sec 6.5 is not an accident of these prompts: it is the phase space the conversion has. A transport whose etendue admits two directions cannot carry twenty, whatever is presented to it.

`Screen.realise` separates the conversion's own loss from that phase-space bound; it is reported by `exp4` under `--realise`.

---

## 7. How far the linearisation carries

Scoring a transport against the model's own final residual has two requirements, and a reading that skips either one measures something else.

**The prediction is affine.** A Jacobian lens is a first-order expansion, $h_{\mathrm{final}} \approx h_{\mathrm{final}}(h^0) + J(h_l - h^0)$, so it carries an offset. Scoring the bare product omits it and a fitted scalar gain absorbs it: measured, the best gain runs 1.9 to 4.9 and is never near 1, which is the signature of a missing intercept rather than a property of the lens.

**The final state is normalised; the transport's output is raw.** `hidden_states[-1]` arrives with the model's final norm already applied -- in this checkpoint its mean per-token norm is 156.1 against 52.1 for its neighbour, a 3$\times$ discontinuity that is exactly the RMSNorm -- while $J$ maps into the un-normalised stream. The clean resolution is to normalise both sides rather than to invert one: the model's logits are $\mathrm{head}(\mathrm{rmsnorm}(x, w))$ and a lens's are $\mathrm{head}(\mathrm{rmsnorm}(Jh, w))$, so applying the readout's own norm to the transport compares exactly the two vectors that get unembedded. Centred over tokens, because both frames are dominated by a shared component they agree on whatever the transport does.

Qwen3.5-4B, six held-out wikitext-2 `test` sequences, reported as $\cos^2$ -- the explained variance of the token-varying part:

| layer | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| $\cos^2$ | 0.006 | 0.019 | 0.047 | 0.069 | 0.111 | 0.253 | 0.394 | 0.644 | 0.852 |

Monotone, and low for most of the network: mean $\cos^2$ over the first quarter of depth is **0.020**, and the curve crosses half the variance only at relative depth **0.90**. **A shared mean-Jacobian linearisation explains under 5% of the cross-token variance in the final residual at and below layer 8.** At the depths a lens is most wanted for -- early and middle, where the logit lens is known to fail [Anthropic 2026] -- the linear transport reproduces almost none of the token-to-token variation.

**And the shape replicates at a second width.** A depth profile from one network describes that network. Qwen3.5-0.8B is the other model small enough to run a float32 forward pass here ($d=1024$ against 2560, 23 layers against 31), measured identically -- same metric, same prompt count and length, same `test` split:

| | Qwen3.5-0.8B | Qwen3.5-4B |
|---|---|---|
| mean $\cos^2$, first quarter of depth | 0.043 | 0.020 |
| crosses half the variance at | relative depth **0.86** | relative depth **0.90** |
| $\cos^2$ at the last layer | 0.857 | 0.852 |

The crossing depth and the endpoint agree closely across a 2.5$\times$ difference in width, and both models sit far below 5% through the first quarter of depth. The shallower model runs somewhat higher early, which is what a shorter path to the output should give. The profile is a property of the method rather than of one network.

**The floor holds across every scoring; the ceiling moves with it.** Three scorings were tried -- the uncentred affine $R^2$ against the post-norm frame, the raw product against the recovered pre-norm direction, and the readout-space form above. Through layer 16 they agree to within 0.003. At layer 30 they read 0.788, 0.691 and 0.852: a spread of 0.16 driven entirely by scoring choice. The low reach at early and middle depth survives every metric by two orders of magnitude, and is quoted as a number; the saturation point moves with the choice, and is quoted as a range.

Two things fix its scope. It measures the *shared* Jacobian as one linear map across tokens, while each $J$ averages a per-token local linearisation; the two differ by exactly the variation measured here. And a low $\cos^2$ establishes one thing: the transport reproduces little of the final state's token-varying part.

The same measurement orders the catalogue's two Qwen3.5-4B fits, and orders them against expectation: the $n=417$ fit is the better linearisation on **26 of 31 layers**; the five it loses are layers 6 to 9 and layer 24. More fit data produced a worse linearisation.

Three alternative explanations were checked and excluded. Storage is identical: both files are 0.41 GB at the same shape and dtype, differing only in the recorded `n_prompts`, which leaves quantisation out of it. The evaluation sequences are drawn from wikitext-2-raw-v1 **test**, while a fit is conventionally taken on train -- which matters, because a prompt inside both fit sets carries $1/417$ of the smaller mean against $1/1000$ of the larger, and would hand the smaller fit a 2.4$\times$ advantage on it by construction. And the ordering holds under every scoring and every target tried.

What remains is scope. The two fits are the only pair the catalogue publishes, so the finding is a property of these two artefacts measured on held-out wikitext.

## 8. A crossing between models that share nothing

Every read to this point scores a crossing: coverage, `certify`, `transfer` and `realise` each
return a number saying how well two sides agree. A screen is not for scoring. It is a shared basis
two systems both convert onto, so a structure resolved on one side can be **rendered out on the
other** -- `entry` in, `inverse` out -- and what returns is a vector in the receiver's own basis
carrying a structure the sender found.

**The two models share nothing that could carry it otherwise.** gpt2 is 768 wide and pythia-70m-deduped
is 512, so a vector from one cannot be placed in the other at all: the direct-injection control
that would confound this is not merely weaker, it does not exist. Their tokenizers were built
separately, so their vocabularies are different objects with different ids for the same text. What
they share is **strings** -- of 50,257 and 50,277 tokens, **36,938 spell the same text in both**.
That intersection, computed from the text alone, is a surface both models convert onto, and it is
the only route between them. Constructing it is the ontology match: no paired training, no learned
mapping, no crosswalk.

    resolve   the leading direction of M = J - alpha*I at gpt2 layer 6
    cross     enter it on the shared string surface through gpt2's own readout
    render    leave through pythia's inverse: a vector in pythia's 512-wide residual basis
    act       add it to pythia's stream at block 2 and run pythia forward

Scored as **selectivity** -- the correlation between the receiver's induced logit change and the
profile the sender named, both centred over the shared tokens, so a perturbation that merely
inflates logits scores zero. 8 prompts of 64 tokens, 512 positions, bootstrapped:

| arm | selectivity | 95% CI |
|---|---|---|
| crossed | $+0.147$ | $[+0.133, +0.161]$ |
| random | $-0.108$ | $[-0.117, -0.098]$ |
| shuffled | $-0.189$ | $[-0.198, -0.181]$ |
| **unresolved** | **$+0.313$** | $[+0.309, +0.318]$ |

The first three arms say the crossing transmits something: against shuffled the gap is $+0.336$,
CI $[+0.327, +0.345]$; against random, $+0.254$, CI $[+0.247, +0.261]$.

**The fourth arm says it is not the resolved structure.** `shuffled` permutes the profile before
rendering and `random` renders nothing, so neither completes the round trip — both can only score
at or below zero whatever the crossing carries, and neither separates "this direction is structure"
from "a pseudo-inverse returns what was put into it". The arm that does is a direction the lens did
**not** resolve: a random unit vector in gpt2's residual basis, pushed through the same $J$ and the
same readout, crossed identically, and scored against its own profile. It arrives at $+0.313$,
against the leading resolved direction's $+0.147$ — $\text{crossed}-\text{unresolved}=-0.167$, CI
$[-0.180, -0.153]$.

**So the shared string surface transports an arbitrary direction, and transports it better than the
one the lens resolved.** What §8 establishes is that the surface is invertible between two models
that share no width, no tokenizer and no training: a profile entered on one side arrives on the
other, and the receiver moves toward it. Its scope is the surface: the matched control places an
unresolved direction above a resolved one, so the result is a property of the round trip. The $5\times5$ specificity matrix below shows that *which* profile was sent is
recoverable, which is a property of the round trip rather than of the transport, and the same
control applies to it.

| sent | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | **+0.135** | -0.211 | +0.168 | -0.212 | -0.142 |
| **1** | -0.174 | **+0.191** | -0.156 | +0.055 | +0.020 |
| **2** | +0.029 | -0.131 | **+0.138** | -0.210 | -0.171 |
| **3** | -0.034 | +0.020 | +0.031 | **+0.413** | -0.343 |
| **4** | -0.198 | +0.172 | -0.172 | -0.246 | **+0.269** |

Every diagonal entry is positive and the diagonal beats every off-diagonal in its row on 4 of 5
rows; mean diagonal $+0.229$ against mean best off-diagonal $+0.091$. The exception is row 0, where
concept 2 scores $+0.168$ against the diagonal's $+0.135$ — the two leading components of one
transport, the pair most likely to overlap in what they name.

**What this is and is not.** It is a demonstration that two models sharing only a writing system can
be placed on one surface and a profile carried across it, entered through one model's readout and
rendered through the other's inverse, with the receiver's response measured against controls. It is
**not** evidence that the transport's resolved directions carry transferable content: an unresolved
direction crosses better. It is one sender, one receiver, one layer pair, at a single injection
strength, at relative depth 0.60, at the edge of the band where §1.5 puts the two reads in agreement and
where the gain is $1.55\times$; the effect is a
logit-space correlation rather than a change in what the receiver writes; and the shared surface was
handed over by a common writing system, which two arbitrary ontologies would not be. Both published
control arms sit significantly *below* zero rather than at it, which is a property of centring the
response over the kept tokens and is not explained here.

## 9. Validation against planted truth

The instrument recovers what is planted and refuses chance.

**Resolved rank.** 12 modes planted at $d=256$ in bands $[60,120]$, $[40,60]$, $[34,40]$ — all above the BBP threshold $2\sqrt d=32$ [Baik–Ben Arous–Péché 2005] — are recovered exactly, 12 of 12, in every band. Planted at $[8,20]$, below threshold, fewer are returned: a spike buried in the bulk leaves the null standing.

**Identity burial.** A rank-8 planted transport under a flat block $40I$ resolves fewer than 8 modes; removing $\alpha I$ recovers exactly 8. This is §1.3 in miniature.

**The complement.** With the stream carrying structure inside the transport's row space, the complement resolves 8 modes; planting 6 further directions orthogonal to that row space moves it to 14 — exactly $+6$.

**Heavy tails.** An i.i.d. matrix with excess kurtosis $>50$ and no rank structure breaches the Gaussian edge, while a Gaussian matrix of the same shape stays inside it. sec 3.3's failure is reproduced deliberately.

**Calibration.** Both closed-form providers read $K=0$ on i.i.d. Gaussian matrices across six seeds; a scaled rotation has every $p$-value exactly $1.0$ and $K=0$ at any $\alpha$ up to $1-10^{-5}$. The resolved count is scale-invariant.

**Ranking resolution.** Recovery says the read finds what is there; a resolution says how small a difference it can see. A known-good transport degraded by a measured amount supplies that. Two degradations probe opposite failure modes -- $J + sG$ with $G$ isotropic, which adds directions carrying nothing, and truncation to the best rank-$k$ approximation, which removes directions that carry -- and the ground truth is the resulting gap in the readout-space metric of §7, centred cosine to the pre-norm final residual direction, measured per layer rather than assumed. Coverage is monotone in that gap and orders every layer correctly at a gap of **0.07%** under lost rank ($p=0.03$) and **1.04%** under added noise ($p=0.008$), decaying to chance below. Scored against the uncentred error on the post-norm frame instead, the noise figure comes out three times smaller, which is why §7 comes first. Two cheaper reads of the same transported frame stop short of it. Participation ratio reaches 0.88 ($p=0.07$) at a 7.5% gap, its best. The norm ratio scores $0.00$ on every noise row but the weakest and $1.00$ on every rank row, so it reports which degradation was applied. Coverage points the same way for both modes, which is the argument for it over the cheap reads.

The calibration covers those two modes. A third one lies outside it, transports differing by *how much data their fit saw*: on the catalogue's two Qwen3.5-4B fits, separated by a median ${\sim}0.08\%$ of transport error, coverage orders by fit sample size rather than transport accuracy, consistently and with the sign inverted. Why estimation error should invert the read is open.

The suite that pins these identities and the reads they support runs to 695 tests, none of which needs a model or a corpus.

---

## 9a. What the reads cover

Every benchmark here is **self-consistency**: the instrument checked against its own nulls and its own constructions. Planted rank recovers exactly above the BBP threshold; coverage of a signal's own top-$k$ subspace returns $k/k_s$; a matched-spectrum surrogate and a random rank-$K$ map both land at chance where the real transport separates from them; and the ranking resolution is calibrated against a degradation of measured size. Those checks are necessary and they are the instrument agreeing with itself.

**One read intervenes, and it does not reach the directions §1 is about.** §8 carries a profile into
a model and acts with it, with the receiver's response measured against controls rather than
described. Its matched control settles what it can support: a direction the lens did *not* resolve
crosses at $+0.313$ against the leading resolved direction's $+0.147$, so the section demonstrates
that the shared string surface is invertible, and its scope is the round trip rather than the
resolved structure.
**The recovered directions are characterised geometrically, and their behavioural role is open.**
Every read here is descriptive geometry.

**External validation waits on an intervention.** The obvious candidate is the ordering [Anthropic 2026] establish between the Jacobian lens and the logit lens, obtained by other people through other means. Their claim is that the Jacobian lens surfaces intermediate content *absent* from the model's output, while every read available in this work scores agreement *with* something present -- which is the property the same paper criticises the tuned lens for. Scored that way the logit lens leads at every layer, which is what such a metric must produce, and it settles nothing about either lens. The original evidence is causal: swap a vector, watch the output change. §8 now does swap a vector, but between two models rather than within one, so the ordering between two lenses on a single model stays out of reach.

What this is: a measurement instrument for characterising linear readouts, with its assumptions stated, the range each read applies over measured, and its resolution calibrated.

## 10. Limitations

Ten models are profiled in sec 4.1, none above 8B, and the scale trend is an observation over ten points in which width, depth, family and generation all covary. It identifies no driver. The catalogue holds lenses up to Llama-3.3-70B, and the subject of [Anthropic 2026] is a ~100-layer model, larger than anything read here.

Section 6 rests on six models at four to twelve prompts of 128 tokens each. The complement result is stable on three of them across transport ranks from 5 to full and across the layer pairing. The three exceptions are gemma-3-1b and the two Qwen3.5 models, and the energy share does not separate them from the rest (sec 6.2). The sample of *prompts* is small, and every number is a mean over it: on Qwen3.5-0.8B two draws of 8 and 12 prompts agree at the bottom of the range (85.2% and 85.3%) and differ at the top (141.8% and 146.5%). The etendue band of sec 6.3 is measured on five models and three show it, reproducing across prompt samples to $r=0.9993$ on the one sampled twice. Three showing interior structure and two not is five observations, not a pattern.

$\mathrm{PR}$ and $H_2$ are energy-weighted effective ranks. They carry no false-alarm level and therefore no decision, which is what makes them appropriate for a noise-free object and what stops them answering "how many directions are real" -- sec 3.2's principal angles answer that instead, and only out to the $k$ actually tested ($k=400$ of $d=2560$). The agreement does not degrade anywhere inside that range, so the reproducible dimension is bounded below by 400 and not bounded above by anything measured here.

**The estimator-noise measurement assumes the two published fits are independent, and the error
runs the wrong way if they are not.** §3.1 divides $\lVert A-B\rVert$ by
$\sqrt{1+n_A/n_B}=1.843$, which is correct for independent means. If $B$'s corpus sits inside
$A$'s, then $A-B=(n_C/n_A)(C-B)$ with $n_C=583$, and the correct divisor is $1.182$. Dividing by
$1.843$ then understates $\mathrm{sd}(A)$ by 36%, which makes "there is very little estimator
noise" **easier** to conclude, not harder: the assumption is anti-conservative, and §3.1's floor at
layer 30 would be $0.224$ rather than $0.144$, putting 996 directions above it rather than 1383.
The same direction applies to §3.2's principal angles and §1.7's Jaccard, both of which measure
agreement between the two fits and are inflated by any shared corpus. The catalogue does not say
whether $B\subset A$, and one line from the lens authors would settle it.

The sharp identity transition at Qwen3.5-0.8B layer 15 is real for that model and is **not** general. Across ten models the largest single-layer identity jump sits at median relative depth 0.68, but for nine of them the ratio of the largest jump to the second largest is 1.06-1.40: identity energy grows smoothly and the argmax marks nothing. The apparent clustering is an artefact of taking an argmax over a smooth curve.

**Precision, rather than cost, sets the ceiling on model size.** Published lenses run to 27B and 70B, and the subject of [Anthropic 2026] is a ~100-layer model. Every read here is spectral, and the checkpoints ship in bfloat16, which carries about three decimal digits -- not enough for a de-biased variance, a Tracy-Widom deviate or a pseudo-inverse. So the forward pass has to run in float32, and a 27B model at float32 is ~108 GB of weights, past any single accelerator available here. Running it in bfloat16 instead would produce streams at bfloat16 precision, which is the one thing these reads cannot use. The ceiling for reads needing a forward pass is 4B, and 8B for reads taken from a lens file alone; going higher is a genuine methodological problem rather than a matter of buying more compute.

Only §8 intervenes; every other read is descriptive geometry, so whether the directions outside J-space matter behaviourally is a causal question, and the causal evidence in [Anthropic 2026] is of a kind this work does not supply.

The identification of the noise floor with the workspace's access threshold -- the all-or-none *ignition* of [Anthropic 2026], and the mobility edge separating bulk from spikes in [Sessford 2026, sec 9] -- remains untested. Section 4 makes it harder to test than it looked: on the transport there is no edge to cross, so it has to be tested on streams, where the floor is legitimate.

---

## 11. Related work

**The premise is not new.** That a residual network's layer-to-output map is identity-plus-correction,
and that the identity dominates, is established. [Veit et al. 2016] read residual networks as
ensembles of shallow paths; [Jastrzębski et al. 2018] characterise residual blocks as iterative
refinement of a carried representation; [Elhage et al. 2021] set out the residual stream as an
additive communication channel and separate the direct path explicitly. What §1 measures is not the
existence of the identity but its size in *fitted mean-Jacobian lens files* and its effect on one
class of estimator.

**A different lens family starts from the identity, for a different reason.** [Belrose et al. 2023]
train an affine translator per layer by gradient descent to match the model's output distribution,
initialised at the identity so that optimisation begins from the logit lens
[nostalgebraist 2020] and learns a correction to it. That is a fitted predictor, and the
initialisation is an optimisation choice.

The object here is not that. A Jacobian lens is a **measured** quantity,
$\mathbb{E}[\partial h_{\text{final}}/\partial h_l]$, with nothing trained, and the identity in it
is a property of the map rather than a starting point for a fit. The two lines of work therefore
meet only in the observation that a residual stream's layer-to-output map is identity-plus-
correction, which [Elhage et al. 2021] and [Veit et al. 2016] state directly. What §1 measures —
the size of that component in published mean-Jacobian files, and what it does to a threshold that
derives its own scale from the frame it is handed — applies to neither a learned translator nor a
fixed threshold, and no learned lens is affected by it.

**The corpus-averaged Jacobian is the object attribution patching uses.** [Nanda 2023],
[Syed et al. 2023] and [Kramár et al. 2024] use a first-order expansion of the same map and
characterise where it fails; §7's finding that a shared mean-Jacobian linearisation explains under
5% of cross-token variance below layer 8 is a measurement in that literature's terms and should be
read against it.

**Effective rank and residual-stream geometry.** The Shannon effective rank of §4 is [Roy &
Vetterli 2007]. [Dong et al. 2021] and [Noci et al. 2022] analyse rank collapse and signal
propagation in transformers, which is directly why the skip path dominates a depth-wise spectrum.
[Ansuini et al. 2019] and [Valeriani et al. 2023] report peaked intrinsic-dimension profiles across
depth — the nearest existing result to §4.1's interior band, and found in models far smaller than
0.8B, which is why Consequence 4.1a states the association with scale as one of the peak's position rather than its presence.

**Heavy-tailed spectra and the validity of a Gaussian edge.** [Martin & Mahoney 2021] and
[Thamm et al. 2022] establish that trained transformer matrices have heavy-tailed spectra poorly
described by a Marchenko–Pastur bulk. §3.3 and §5 rediscover this for transports and should be read
as a confirmation rather than a new observation. The massive-activation coordinates §3.3 relies on
are characterised by [Sun et al. 2024], [Xiao et al. 2023] and [Dettmers et al. 2022].

**Reading singular directions as features has a known failure mode.** [Bolukbasi et al. 2021] show
that directions selected by a criterion and then interpreted by inspection produce convincing
stories that do not hold up; [Elhage et al. 2022] give the superposition account of why a single
direction is rarely one feature. §1.7's token tables are illustration for exactly this reason, and
the load there is carried by the two-fit reproduction rather than by legibility.

**The crossing in §8 belongs to a cross-model alignment literature.** Aligning two models through a
shared surface and rendering between them is the subject of [Bansal et al. 2021] on stitching,
[Moschella et al. 2023] on relative representations, and [Huh et al. 2024] on representational
convergence. The injection half is activation steering: [Turner et al. 2023], [Zou et al. 2023],
[Rimsky et al. 2024] and [Li et al. 2023] set the standard for controls and for reporting
behavioural rather than logit-space outcomes — a standard §8 does not meet, as its own matched
control shows.

**Method borrowed from outside the domain.** [Scanu et al. 2026] assess robustness of noise-aware
quantum neural networks through Jacobian geometry. The domain does not transfer, but three things
do: their descriptor suite — effective rank, participation ratio, anisotropy, principal angles — is
the set this work arrived at independently; their entropy-matched noise calibration is the
principle behind the surrogates of §5; and their stability construction, principal-angle variation
between dominant eigenspaces under controlled perturbation, is what §3.2 applies to independent
fits. Their own conclusion is that no universal geometric robustness signature exists across noise
channels, at $R^2\approx0.3$–$0.5$.

## 12. Reproducing the measurements

Every figure in this paper is produced by a script in `research/experiments/` and, where a run is
expensive, stored under `results/`. The lens files themselves come from the Neuronpedia mirror and
are fetched with `entroptics-jlens fetch <model>`.

| section | script | stored run |
|---|---|---|
| §1 the claim table | `exp51_the_claim.py` | `results/the_claim.json` |
| §1.2 structure-free surrogates | `exp55_structure_free_control.py` | `results/structure_free_control.json` |
| §1.3 planted rank | `tests/test_claim_ground_truth.py` | — (fixture, seed $100\times\text{rank}$) |
| §1.4 false-alarm sweep, §1.5 depth profile | `exp54_sweep_and_depth.py` | `results/sweep_and_depth.json` |
| §1.7 what the modes name | `exp52_what_the_hidden_modes_name.py` | — |
| §1.7 two-fit reproduction, blocks, nulls | `exp53_hidden_modes_reproduce.py` | `results/hidden_modes_reproduce.json` |
| §3.1 estimator noise, §3.2 principal angles | `exp3_two_fit_noise.py` | `results/twofit.json` |
| §3.3 shuffle controls | `exp1_transport_spectrum.py` | `results/exp1.json` |
| §4, §4.1 depth profiles | `exp2_depth_profile.py` | `results/depth.json` |
| §6 complement and etendue, all six models | `exp4_stream_complement.py` | `results/complement_*.json` |
| §7 linearisation reach | `exp18_prenorm_target.py` | `results/prenorm_target.json` |
| §8 cross-model crossing | `exp26_cross_tokenizer.py` | `results/cross_tokenizer_control.json` |
| §9 ranking resolution | `exp19_recalibrate.py` | `results/recalibrate.json` |

`research/experiments/README.md` indexes every script with a one-line description. The test suite
pins the identities these reads rest on and needs no model or corpus.

---

## References

**[Anthropic 2026]** *Verbalizable Representations Form a Global Workspace in Language Models.* Transformer Circuits Thread, 2026. https://transformer-circuits.pub/2026/workspace/index.html. Code: https://github.com/anthropics/jacobian-lens (Apache 2.0). Lenses: https://huggingface.co/neuronpedia/jacobian-lens.

**[Sessford 2026]** Sessford, I. J. *Entroptics: Reading any 2-D signal as a finite optical aperture at its own entropy-matched resolution.* Pre-print, July 2026. https://doi.org/10.5281/zenodo.21273400. Software: https://pypi.org/project/entroptics/ (`pip install entroptics`; this work uses 0.2.1). Source: https://github.com/Agience/entroptics (Apache-2.0).

**[Ansuini et al. 2019]** Ansuini, A., Laio, A., Macke, J. H., Zoccolan, D. *Intrinsic dimension of data representations in deep neural networks.* NeurIPS 2019.

**[Baik–Ben Arous–Péché 2005]** Baik, J., Ben Arous, G., Péché, S. *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices.* Annals of Probability 33(5), 1643–1697.

**[Bansal et al. 2021]** Bansal, Y., Nakkiran, P., Barak, B. *Revisiting Model Stitching to Compare Neural Representations.* NeurIPS 2021.

**[Belrose et al. 2023]** Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., Steinhardt, J. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.

**[Bolukbasi et al. 2021]** Bolukbasi, T., Pearce, A., Yuan, A., Coenen, A., Reif, E., Viégas, F., Wattenberg, M. *An Interpretability Illusion for BERT.* arXiv:2104.07143.

**[Dettmers et al. 2022]** Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022.

**[Dong et al. 2021]** Dong, Y., Cordonnier, J.-B., Loukas, A. *Attention is not all you need: pure attention loses rank doubly exponentially with depth.* ICML 2021.

**[Elhage et al. 2021]** Elhage, N., Nanda, N., Olsson, C., et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.

**[Elhage et al. 2022]** Elhage, N., Hume, T., Olsson, C., et al. *Toy Models of Superposition.* Transformer Circuits Thread, 2022.

**[Huh et al. 2024]** Huh, M., Cheung, B., Wang, T., Isola, P. *The Platonic Representation Hypothesis.* ICML 2024.

**[Jastrzębski et al. 2018]** Jastrzębski, S., Arpit, D., Ballas, N., Verma, V., Che, T., Bengio, Y. *Residual Connections Encourage Iterative Inference.* ICLR 2018.

**[Johnstone 2001]** Johnstone, I. M. *On the distribution of the largest eigenvalue in principal components analysis.* Annals of Statistics 29(2), 295–327.

**[Kramár et al. 2024]** Kramár, J., Lieberum, T., Shah, R., Nanda, N. *AtP\*: An efficient and scalable method for localizing LLM behaviour to components.* arXiv:2403.00745.

**[Li et al. 2023]** Li, K., Patel, O., Viégas, F., Pfister, H., Wattenberg, M. *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS 2023.

**[Martin & Mahoney 2021]** Martin, C. H., Mahoney, M. W. *Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory.* JMLR 22(165).

**[Moschella et al. 2023]** Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., Rodolà, E. *Relative representations enable zero-shot latent space communication.* ICLR 2023.

**[Nanda 2023]** Nanda, N. *Attribution Patching: Activation Patching At Industrial Scale.* 2023.

**[Noci et al. 2022]** Noci, L., Anagnostidis, S., Biggio, L., Orvieto, A., Singh, S. P., Lucchi, A. *Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse.* NeurIPS 2022.

**[nostalgebraist 2020]** nostalgebraist. *Interpreting GPT: the logit lens.* LessWrong, 2020.

**[Rimsky et al. 2024]** Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. M. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024.

**[Roy & Vetterli 2007]** Roy, O., Vetterli, M. *The effective rank: a measure of effective dimensionality.* EUSIPCO 2007.

**[Scanu et al. 2026]** Scanu, G., Barletta, L., Rini, S. *JGRA: Jacobian Geometry Robustness Assessment in NISQ Noise-Aware Quantum Neural Networks.* arXiv:2606.09964v2, June 2026.

**[Sun et al. 2024]** Sun, M., Chen, X., Kolter, J. Z., Liu, Z. *Massive Activations in Large Language Models.* arXiv:2402.17762.

**[Syed et al. 2023]** Syed, A., Rager, C., Conmy, A. *Attribution Patching Outperforms Automated Circuit Discovery.* arXiv:2310.10348.

**[Thamm et al. 2022]** Thamm, M., Staats, M., Rosenow, B. *Random matrix analysis of deep neural network weight matrices.* Physical Review E 106, 054124.

**[Turner et al. 2023]** Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., MacDiarmid, M. *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248.

**[Valeriani et al. 2023]** Valeriani, L., Doimo, D., Cuturello, F., Laio, A., Ansuini, A., Cazzaniga, A. *The geometry of hidden representations of large transformer models.* NeurIPS 2023.

**[Veit et al. 2016]** Veit, A., Wilber, M., Belongie, S. *Residual Networks Behave Like Ensembles of Relatively Shallow Networks.* NeurIPS 2016.

**[Xiao et al. 2023]** Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. *Efficient Streaming Language Models with Attention Sinks.* arXiv:2309.17453.

**[Zou et al. 2023]** Zou, A., Phan, L., Chen, S., et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.

**[Dehaene–Naccache 2001]** Dehaene, S., Naccache, L. *Towards a cognitive neuroscience of consciousness: basic evidence and a workspace framework.* Cognition 79(1–2), 1–37.
