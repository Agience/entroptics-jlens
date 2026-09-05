# entroptics-jlens

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22293929.svg)](https://doi.org/10.5281/zenodo.22293929)

## The claim

**A Jacobian lens carries a large identity component at depth, and a spectral floor derived from
the matrix it is judging is inflated by it. Removing the identity first changes the resolved count
by 2x to 21x, of which 1.15x to 1.87x is attributable to the transport rather than to the floor.**

Eleven lens files, ten of the twenty-two on the Neuronpedia mirror, at each model's deepest
fitted layer, `mp` null at `far=0.05`:

| model | width | identity share | K(J) | K(J−αI) | change |
|---|---|---|---|---|---|
| qwen3-1.7b | 2048 | 0.798 | 3 | 64 | **21.3×** |
| qwen3-4b | 2560 | 0.876 | 7 | 149 | **21.3×** |
| gemma-3-4b | 2560 | 0.703 | 16 | 165 | **10.3×** |
| qwen3.5-4b | 2560 | 0.790 | 25 | 183 | 7.3× |
| gpt2 | 768 | 0.422 | 6 | 39 | 6.5× |
| qwen3.5-0.8b | 1024 | 0.778 | 12 | 68 | 5.7× |
| gemma-3-1b | 1152 | 0.660 | 17 | 64 | 3.8× |
| llama3.1-8b | 4096 | 0.720 | 74 | 269 | 3.6× |
| pythia-70m | 512 | 0.458 | 2 | 4 | 2.0× |
| gemma-3-270m | 640 | 0.107 | 39 | 42 | 1.1× |

**9 of 10 models. Five families. Widths 512 to 4096.** Reproduce it with
[`exp51_the_claim.py`](https://github.com/Agience/entroptics-jlens/blob/main/research/experiments/exp51_the_claim.py).

**Read this table before quoting that one.** A derived floor rises with any added identity,
whatever is underneath it, so most of the raw range is the estimator's response to it. Two
structure-free surrogates measure the part that belongs to the transport, differing in one
invariant -- whether the floor's own variance input (a median row energy) is held fixed
(`research/experiments/exp55_structure_free_control.py`, five draws each, median reported):

| model | change | spectrum surrogate | excess | rotation surrogate | excess |
|---|---|---|---|---|---|
| gemma-3-270m | 1.1x | 1.03x | 1.04x | 1.08x | 1.00x |
| gpt2 | 6.5x | 2.00x | 3.25x | 5.57x | 1.17x |
| pythia-70m | 2.0x | 2.33x | 0.86x | 2.00x | 1.00x |
| gemma-3-1b | 3.8x | 2.69x | 1.40x | 4.00x | 0.94x |
| gemma-3-4b | 10.3x | 4.31x | 2.39x | 9.17x | 1.12x |
| llama3.1-8b | 3.6x | 1.97x | 1.85x | 2.10x | 1.73x |
| qwen3.5-0.8b | 5.7x | 3.00x | 1.89x | 4.00x | 1.42x |
| qwen3.5-4b | 7.3x | 4.06x | 1.80x | 4.82x | 1.52x |
| qwen3-1.7b | 21.3x | 3.67x | 5.82x | 21.33x | 1.00x |
| qwen3-4b | 21.3x | 7.88x | 2.70x | 16.56x | 1.29x |
| **median** | | | **1.87x** | | **1.15x** |

`spectrum` holds the singular values and alpha and redraws both subspaces, so the median row energy
moves and the floor moves with it. `rotation` holds the singular values, alpha **and every row
norm** -- verified to 2e-16, so the floor's variance estimate is identical by construction -- while
destroying the alignment between the transport's input and output directions.

**The excess attributable to the transport lies between 1.15x and 1.87x.** The bracket's width is
how far a surrogate is allowed to move the statistic the floor is made of.

There is an eleventh lens file, a second independent fit of qwen3.5-4b over 417 sequences rather
than 1000. It gives **7.28× against the other's 7.32×**, so the measurement replicates across two
fits of one model as well as across ten models.

### Why it holds

A transformer's residual stream **adds** each layer's output to what came before, so a map from
layer *l* to the final layer contains a copy of the identity by construction — and it grows with
depth, because less transformation is left above it. At the deepest layer of every model here above
1B parameters, **66% to 88% of the transport is that pass-through.**

The identity flattens the spectrum. The noise floor is estimated *from that spectrum*. So the floor
rises with the identity and buries the real modes underneath it. Read qwen3-1.7b's deepest layer as
it stands and you conclude it resolves **3 directions**. It resolves **64**.

### Why it is not a coincidence across models

**The size of the change is predicted by the identity share: Spearman +0.773** across the eleven
lens files, and +0.745 across the ten models. The one lens where the count barely moves —
gemma-3-270m at 1.1× — is the one with almost no identity to remove, at 0.107. A structure-free
surrogate reproduces that same association (§1.2 of the paper), which places it with the floor.

### What this is about

**Measured here:** the published lens files carry the identity — it is in the matrices, at the
shares tabulated above, because a faithful Jacobian of a residual stream contains it — and a
spectral read whose threshold is derived from the matrix it is judging resolves a different count
depending on whether it is removed first.

The lens files are not at issue: they are a correct measurement of the map they claim to measure,
and nothing here inspects
anyone else's code, and this package cannot tell you what a given paper did downstream of the
checkpoint. The claim is about what is in the files and what happens if you read them directly,
which is what the tooling around them does by default.

**Also not claimed: that subtracting `αI` is a new idea.** It is the obvious step once you notice
the identity is there, and `tr(J)/d` is the first thing anyone would write down. The contribution
is the measurement — that it is 66–88% of the matrix at depth, that the resolved count moves by a
factor of 2 to 21, how much of that survives a structure-free surrogate, and that it settles below
relative depth 0.6. Those numbers did not exist before this, and without them there is no reason to bother.

### Why the larger number is the right one

The obvious objection: you removed something and got a bigger number — why is bigger correct?

Because on a matrix of **known** rank, the read on J-alpha*I does not move when you add identity, and
the raw read collapses. Plant rank 20 in a 256-wide matrix, then add `α·I` for growing `α`:

| | α=0 | α=0.5 | α=1.5 | α=3.0 | α=6.0 |
|---|---|---|---|---|---|
| identity share | 0.000 | 0.032 | 0.230 | 0.543 | 0.826 |
| **K(J)** | 20 | 20 | 20 | 17 | **5** |
| **K(J−αI)** | **20** | **20** | **20** | **20** | **20** |

The planted rank is 20 throughout. The raw read reports 5 of it once the identity dominates; the
read on J-alpha*I reports 20 at every identity strength.

Over a sweep of planted ranks 5–40 against five identity strengths, 20 cases: the read on J-alpha*I
recovers the planted rank **exactly in 15 of 20**, mean absolute error **2.50**; the raw read
manages 11 of 20 at mean error **5.70**, and every one of its failures is an *understatement* that
grows with the identity share.

The read on M has five misses, all at planted rank 40, where it saturates at 30 — that is the
`mp` floor's own documented tendency to under-count signal-dense matrices, present with or without
an identity. It is flat across α: 30 at every strength, while the read on J falls 30 → 5.

**So removing αI does not make the number bigger. It makes the number invariant to a component the
architecture guarantees is present.**

### Does it apply to the layer you are reading?

The table above is each model's deepest layer. Across every layer of the five lenses narrow
enough to sweep whole — 81 layers — bucketed by relative depth
(`research/experiments/exp54_sweep_and_depth.py`):

| relative depth | median identity | median change | layers over 1.5× |
|---|---|---|---|
| 0.0 – 0.2 | 0.040 | 0.9× | 0 / 17 |
| 0.2 – 0.4 | 0.017 | 1.0× | 0 / 15 |
| 0.4 – 0.6 | 0.080 | 1.0× | 0 / 16 |
| 0.6 – 0.8 | 0.281 | 1.2× | 5 / 15 |
| **0.8 – 1.0** | **0.474** | **3.6×** | **14 / 18** |

**Below relative depth 0.6 the two reads agree.** Read the raw transport there and you get the
same answer. Above 0.8 the change is large in 14 of 18 layers measured.

And the cut is the identity share, not the depth: over those 81 layers every layer where the
change exceeds 1.5× has an identity share between 0.348 and 0.778, and every layer where it does
not sits between 0.003 and 0.433. The two populations meet at about **0.4** — which `audit` prints in its own
column, so you can read off whether your layer is in the affected range rather than guessing from
its index.

The deep layers are where the change matters, and they are also where claims about a model's
workspace and capacity are made.

### It does not depend on the one free parameter

`K` is read against a noise floor at a false-alarm rate you choose. If the effect only existed at
the value in the table it would be an artefact of that choice, so it was swept across **5.7 orders
of magnitude**:

| far | models over 1.5× | range |
|---|---|---|
| 0.5 | 9/10 | 1.1× – 21.7× |
| 0.05 | 9/10 | 1.1× – 21.3× |
| 0.005 | 9/10 | 1.1× – 21.1× |
| 0.0005 | 9/10 | 1.1× – 21.0× |
| 1e-06 | 9/10 | 1.1× – 20.9× |

Nine of ten at every setting, and the per-model factors barely move — qwen3-1.7b runs 21.7× to
20.3×, gpt2 stays at 6.5×, and gemma-3-270m sits at 1.1× throughout. **The exception is as stable
as the rule.**

### The fix

Exact, one line, nothing fitted:

```python
M = je.decompose(J).residual        # alpha = tr(J)/d, then J - alpha*I
```

`alpha = tr(J)/d` is the orthogonal projection of `J` onto the span of the identity — the unique
least-squares coefficient against a basis element the architecture guarantees is present. No
threshold, no null, no training. That is why this replicates: there is nothing in it to fail.

**In your existing loop**, that is one inserted line and nothing else changes:

```python
for l in layers:
    J = lens.jacobian(l)
    M = je.decompose(J).residual      # <- the line
    rank = your_effective_rank(M)     # everything downstream unchanged
```

Measured on Qwen3.5-4B, the number that loop produces before and after:

| layer | before | after | off by |
|---|---|---|---|
| 0 | 94 | 93 | 1.0× |
| 12 | 203 | 178 | 0.9× |
| 24 | 49 | 131 | 2.7× |
| 30 | 25 | **183** | **7.3×** |

**Apply it unconditionally.** It is exact at every layer, costs `O(d²)` — 8.8 ms at d=2560, against
seconds for the spectrum that follows — and is never wrong to have done. There is no decision to
make and no read to take first.

`screen` is a *separate* convenience, and not a prerequisite. It answers "can I skip the expensive
spectrum on this layer entirely", which is a question about your compute budget rather than about
correctness:

```
entroptics-jlens screen lens.pt

 layer    alpha  identity  needs the full read?
     0    0.181     0.008  no
    19    0.632     0.324  no
    29    1.063     0.742  yes
    30    0.994     0.790  yes
```

So: **remove the identity always; screen only when you want to avoid paying for spectra you do
not need.**

### Which half is which

`decompose` is **not** an entroptics read. It is two lines of numpy —

```python
alpha = float(np.trace(A) / d)      # <J, I> / <I, I>
R = A - alpha * np.eye(d)
```

— and it carries the *domain* knowledge: a residual stream adds, so the transport contains an
identity. What entroptics supplies is the measurement, `noise_floor` under the `mp` provider, which
is the only thing here that turns a matrix into a number.

That matters for reading the claim. The `mp` floor estimates its variance **from the matrix it is
judging**, so the identity raises the floor by the same amount it flattens the spectrum. This is
therefore not a lens fact that happens to use entroptics — it is a correction *to* an entroptics
read, supplied by the domain. A fixed threshold would not have this failure mode.

```bash
pip install "entroptics-jlens[lens]"
entroptics-jlens fetch gpt2                    # 13 MB, transports only
entroptics-jlens audit lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt
```

Under eight seconds on gpt2, no GPU, no corpus, no labels, no forward pass — and it prints both
columns so the gap is visible rather than inferred.

## What it actually measures

Four reads. Three of them need only the checkpoint.

| read | question it answers | needs | cost |
|---|---|---|---|
| `screen` | **which layers even need the expensive read?** | the file | O(d²), no SVD |
| `audit` | at this layer, how much of the map is the architecture rather than the model? how many directions does it resolve above its own noise floor? | the file | O(d³) per layer |
| `compare` | are these two readouts the same map, or has one drifted? | two files | O(d³) per layer |
| `coverage` | how much of a real residual stream does this readout reach, against what a random readout of the same size would reach? | the files, plus streams you collect | O(d³) per layer |

### `screen` — do not pay for a spectrum you do not need

The identity share is `tr(J)/d` and one Frobenius norm. No SVD, `O(d²)` against a spectrum's
`O(d³)` — **8.8 ms against 2.39 s at d=2560** — and it is the quantity that *predicts* the
correction rather than a proxy for it, which is what makes screening on it sound.

```
entroptics-jlens screen qwen3.5-4b.pt

 layer    alpha  identity  needs the full read?
    12    0.280     0.070  no
    20    0.724     0.390  no
    24    1.037     0.583  yes
    30    0.994     0.790  yes

3 of 6 layers carry enough identity to change the answer: 24, 26, 30
  entroptics-jlens audit qwen3.5-4b.pt --layers 24,26,30
```

On Qwen3.5-4B's 31 layers: **auditing everything is 82 s; screening (1.0 s) then auditing the 10
that flagged is 27 s.** The layers it flags are exactly the ones a full audit calls
identity-dominated — asserted end to end in [tests/test_screen.py](https://github.com/Agience/entroptics-jlens/blob/main/tests/test_screen.py), because
arriving cheaply at the wrong layers would be worse than arriving expensively at the right ones.

### `audit` — what a transport is, and what it is not

Here is the real output on the published gpt2 lens — 13 MB, and 7.6 s wall including interpreter start:

```
  d_model 768   fitted layers 11   n_prompts 277
  reading 11 layer(s) at far=0.05

 layer    alpha  identity     PR(J)  PR(J-aI)        K(J)     K(J-aI)  reads as
     0    0.263     0.097      41.6      36.3       71/86      67/104  transport
     1    0.375     0.135      71.8      60.5       69/56       64/80  transport
     2    0.495     0.185      83.0      63.9       63/26       59/51  transport
     3    0.575     0.220      75.1      51.7       55/16       55/36  transport
     4    0.679     0.254      57.8      35.0       46/16       52/32  transport
     5    0.801     0.278      41.6      22.9       41/19       51/33  transport
     6    0.936     0.348      39.5      17.3       31/14       48/26  transport
     7    1.024     0.405      32.7      11.6       24/11       48/22  transport
     8    1.108     0.471      24.0       6.6        12/5       53/18  transport
     9    1.138     0.519      15.3       3.4         6/6       46/13  identity-dominated
    10    1.112     0.422       4.7       1.5         6/7       39/15  transport
```

Reading across a row:

- **`alpha`** is `tr(J)/d`, how much of the transport is a plain pass-through. Watch it climb from 0.26 to 1.14 with depth.
- **`identity`** is the share of the matrix's total energy that pass-through accounts for. At gpt2 layer 9, more than half.
- **`PR(J)`** is the effective rank of the transport as it stands — how many directions it meaningfully uses, with no threshold anywhere in it.
- **`PR(J-aI)`** is the same read after the pass-through is removed.
- **`K(J)` / `K(J-aI)`** are how many directions stand above the matrix's own noise floor, before and after that removal, each shown as `mp/fence`.

`mp` is the calibrated null: it answers *how many modes would noise alone have produced*, at the false-alarm rate you set with `--far`. `fence` is the Tukey outlier fence `Q3 + 1.5·IQR` of the singular spectrum — a useful second view of where the spectrum breaks, but **not a null**. It carries no false-alarm rate, `--far` does not move it at all (measured: sweeping `far` from 0.5 to 1e-6 leaves it at 2.40679 to five decimals while `mp` moves), and because a few dominant directions raise `Q3` and the `IQR` together, it reads *fewer* modes the more structured the spectrum is — 104 against mp's 67 at gpt2 layer 0, but 13 against 46 by layer 9. Read the gap as a description of the spectrum's shape, never as an interval on `K`.

**Now look down the `K` columns.** `K(J)` falls from 71 to 6 — the transport appears to resolve less and less as you go deeper. `K(J-aI)` does not: 67, then 46, then 39. Nearly all of that collapse is the identity, and the mechanism is mechanical rather than mysterious. The identity flattens the spectrum; the default floor estimates its variance *from that spectrum*; so the floor rises with the identity and buries the real modes beneath it. Same matrix, same null, same false-alarm rate.

On Qwen3.5-4B the same effect is larger. At layer 30 the transport resolves **25** directions read on `J` and **183** read on `J-aI`. On the identity-cored test fixture it is total: rank 8 planted, **0** resolved on `J`, exactly **8** on `J-aI`.

### The identity, layer by layer

The claim at the top reads one layer per model. Here is the whole depth profile for one of them,
Qwen3.5-4B, `--layers 0,6,12,18,24,26,28,30`:

| | L0 | L6 | L12 | L18 | L24 | **L26** | L28 | L30 |
|---|---|---|---|---|---|---|---|---|
| identity | 0.008 | 0.026 | 0.070 | 0.236 | 0.583 | 0.629 | 0.706 | 0.790 |
| PR(J) | 12.4 | 69.3 | 88.7 | 240.3 | 987.0 | 1115.2 | 1249.1 | 1328.5 |
| **PR(J-aI)** | 12.3 | 67.8 | 80.3 | 163.2 | 326.1 | **345.0** | 262.6 | 124.4 |

`alpha = tr(J)/d` is the unique least-squares coefficient against a basis element the architecture guarantees is present. Nothing is tuned and nothing is trained.

**If you take one thing from this package: when you read the spectrum of a Jacobian-style transport, subtract the identity first, or you are describing the residual stream rather than the model.**

### `compare` — did this readout drift?

Two independent fits of one model should be the same map. A checkpoint that has moved, or a fit that has degenerated, should not be.

```bash
entroptics-jlens compare fit_a.pt fit_b.pt --k 400
```

There are two published fits of Qwen3.5-4B, one over 417 sequences and one over 1000, so the honest answer is measurable rather than assumed. Over their top 400 directions:

| layer | agree_to | cos_mean | cos_min | verdict |
|---|---|---|---|---|
| 0 | 268 | 0.8587 | 0.0004 | same map |
| 12 | 395 | 0.9909 | 0.5966 | same map |
| 26 | 397 | 0.9960 | 0.7592 | same map |
| 30 | 399 | 0.9985 | 0.8844 | same map |

`cos_min` is the trap. Two genuine fits of one model disagree **completely** on their 400th direction — it sits in the spectral tail where the fit is noise — so a rule reading the minimum would have called two fits of one model different maps at every layer. Read `agree_to` instead: the two agree on 268 of their leading 400 directions at layer 0 and on 399 of them at layer 30. That is the question principal angles exist to answer, and no scalar summary gives it.

The verdict reads the mean, whose two populations are far apart and were both measured: 0.86 to 0.9985 for these fits of one model, 0.17 for two unrelated maps of the same shape.

This read sees the map only. It says whether two readouts are the same object. It cannot say which of them is better — for that you need the model.

### `coverage` — how much of a real stream does it reach?

The other two reads never touch the model. This one does, because the question is about the model's actual activations.

Given a residual stream and the same stream after the readout has acted on it, `coverage` is the overlap of the two resolved subspaces, `||V_s^T V_t||_F^2 / k_s`. What makes it a measurement rather than a number is that it has an **analytic chance level**: a randomly-oriented `k_t`-dimensional subspace covers `k_t/d` of anything at all. So the answer has the shape *"this readout reaches 22% of the stream's structure, where chance is 0.9%"*.

The table prints a random map of the same rank on the same frame beside the real one, every time, because that control is what a read has to beat to mean anything. Collect the streams yourself:

```python
import numpy as np, torch, transformers
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval().float()
tok = transformers.AutoTokenizer.from_pretrained("gpt2")
# At least d_model tokens. A (T, d) frame has rank at most T, so a short prompt caps what
# can be resolved at its own length -- see the note below.
ids = tok(your_long_text, return_tensors="pt").input_ids[:, :1024]
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
np.savez("streams.npz", a=np.stack([h[0].numpy() for h in out.hidden_states]))
```

```bash
entroptics-jlens coverage lens.pt --streams streams.npz --layers 0,5,10
```

```
 layer  k_sig  k_read  coverage   chance  x chance  random map  x chance
     0     13      22    0.1606   0.0286       5.6      0.0286       1.0
     5     20      12    0.1757   0.0156      11.2      0.0157       1.0
    10     15       7    0.2185   0.0091      24.0      0.0095       1.0
```

The real transport reaches 16-22% of the stream's resolved structure, at 6 to 24 times chance. The random map of the same rank sits at **1.0x chance at every layer** — which is what tells you the first column is measuring the transport rather than the frame.

**Give it enough tokens.** `k_sig` is the number of directions the stream itself resolves, and a `(T, d)` frame has rank at most `T`. Feed it a three-token prompt and `k_sig` is 1, and the tool will happily report a one-dimensional overlap at 679 times chance — a spectacular-looking number that is a fact about your prompt. `coverage` says so out loud whenever a stream is shorter than `d`, but it is easier not to do it.

Layer `l`'s transport pairs with hidden state `l+1`, and the tool checks that rather than assuming it — reading `s[l]` instead produces a curve that looks entirely plausible and is off by one.

## How it works

Everything here is linear algebra on the readout's own matrix, plus a null to score it against.

**The rank a matrix resolves.** Take the singular values. A matrix of pure noise has a predictable spectrum, and its largest singular value has a known distribution (Tracy-Widom) — so there is a level above which a singular value is not something noise would produce. Count what stands above it and you have a resolved rank with a stated false-alarm rate, no threshold chosen by anyone. That floor comes from [entroptics](https://pypi.org/project/entroptics/), which this package is a domain wrapper over.

**Two floors, because one is a lower bound.** The default estimator reads the per-cell variance off the whole matrix, signal included, so a matrix carrying a lot of structure lifts its own floor and buries its weakest modes. Measured on planted ranks: 30 modes planted, 19 recovered by the default and 29 by the robust estimator. Both columns are printed; the gap between them is diagnostic.

**Effective rank without any floor at all.** Participation ratio, `(sum s^2)^2 / sum s^4`, equals `r` exactly for a spectrum with `r` equal non-zero values. It needs no null and carries no false-alarm rate, which is what makes it usable on a corpus-averaged transport where there is no noise bulk left for a floor to find an edge of — and what stops it answering "how many of these directions are real". Read it next to `K`, never instead of it. A pure-noise matrix has the flattest spectrum there is and therefore the *highest* participation ratio in the file; `audit` reports its peak only over layers that resolve something, for exactly that reason.

**Subspace overlap with a chance level.** The sum of squared canonical correlations between two subspaces, normalised by the smaller dimension. 1 when one contains the other, 0 when orthogonal, `k_t/d` when the second one is randomly oriented.

### The one thing that will mislead you

The `mp` floor estimates its per-cell variance **from the matrix it is judging**. So anything that concentrates a matrix's energy raises the bar that same matrix then has to clear — and a control that destroys structure can come out *ahead of the real thing*.

This is not hypothetical. Take gpt2's transports and compare each against its own entry-shuffle, which permutes every cell and should be pure noise:

| layer | K(J) | K(shuffled J) | | K(J−αI) | K(shuffled J−αI) |
|---|---|---|---|---|---|
| 7 | 24 | 25 | | 48 | 13 |
| 8 | 12 | 28 | | 53 | 15 |
| 9 | 6 | 34 | | 46 | 21 |
| 10 | 6 | 43 | | 39 | 44–51 |

Shuffles are at seed 0; the layer-10 range is over 8 seeds, drawn because that is the row the argument turns on.

Read on `J`, the real transport resolves **fewer** directions than its own shuffled noise at four consecutive layers. That looks like a devastating result about the lens. It is the identity: the shuffle moves the diagonal off the diagonal, destroying the identity core on the control side while the real side keeps it holding its floor up. Decompose first and the inversion is gone at layers 7–9, robustly across seeds.

It survives at layer 10 — and there the cause is the same mechanism wearing different clothes. `PR(J−αI)` is 1.5 at that layer: the transport is nearly rank-one, and a single dominant direction concentrates energy just as an identity does. Nothing here fixes that one, so it is reported rather than explained away.

**The rule that falls out:** a control must preserve what the architecture guarantees. An entry shuffle does not preserve a residual stream's identity, so it is not a valid null for a raw transport.

### And the obvious fix does not work either

`nulls.py` carries surrogates that keep more of the matrix — `sign_flip` preserves `|J|` entrywise and so every row and column energy exactly, destroying only sign coherence. That is the right instinct, and on a clean planted rank-6 it is exactly right: it finds all 6, valid and calibrated at an exceedance of 0.045 against a nominal 0.05.

Now set **one** cell of that same matrix to 400, against an rms of 1.12:

| surrogate | K, clean | K, with one outlier cell |
|---|---|---|
| `sign_flip` | 6 | 1 |
| `within_col_shuffle` | 6 | 1 |
| `within_row_shuffle` | 6 | 0 |
| `mp` (analytic) | 6 | 7 |

Planted rank is 6. Preserving the magnitude profile *exactly* means the giant cell rides along in every surrogate draw, and one giant cell carries a large top singular value by itself — so the floor climbs above the real structure. A real transport is that second column: removing the identity does not remove the massive activations, and on gpt2's `J−αI` the largest cell runs from 70× the rms at layer 0 to 599× at layer 10, sitting at `(447,138)` throughout. `sign_flip` reads K = 2, 1, 1, 1 across layers 0/5/9/10 where `mp` reads 67, 51, 46, 39.

**Both nulls fail on massive activations, in opposite directions.** `mp` over-counts because a flattened bulk sets its variance too low; the sampled surrogates under-count because one preserved cell sets their floor too high. Nothing here closes that gap, and inventing a winsorised surrogate to close it would be fitting rather than measuring.

So: use the sampled surrogates where energy is not dominated by a few cells, and on a real transport decompose first, read `mp` knowing it is a lower bound, and **read `PR` next to `K`** so a concentrated spectrum is something you can see rather than infer from a surprising count. A sampled-floor `K` of 0 or 1 on a transport is telling you about its largest entry, not its rank.

## Other things it will do

Neither of these is the claim above. Both are real capabilities with measured limits,
and the limits are stated because they decide whether the capability is worth using.

### Pointing it at ordinary weights: what did quantisation change?

Nothing above is specific to a lens. `compare` works on any two matrices of the same shape, so
point it at a model's own weights before and after a change.

gpt2's attention output projections, original against symmetric per-tensor quantisation, showing
how much of each layer's map survived:

| layer | int8 | int4 | int3 |
|---|---|---|---|
| 0 | 0.9991 | 0.8669 | 0.7681 |
| 3 | 0.9981 | 0.6674 | 0.3485 |
| 6 | 0.9993 | 0.8506 | 0.4292 |
| 9 | 0.9985 | 0.7502 | 0.4065 |
| **11** | **0.9621** | **0.3893** | 0.3341 |

**int8 is clean everywhere; at int4 the damage is uneven** — 0.87 at layer 0 against 0.39 at
layer 11 — and layer 11's effective rank collapses from **27.2 to 4.2**, from 27 usable directions
down to 4.

**It does not tell you which layer to protect.** Quantising one layer at a time and measuring what
it costs the model gives Spearman **+0.070** against these scores — no relationship. Layer 0 has
the *highest* agreement and the *highest* cost (+0.368 loss); layer 1 has the lowest agreement and
costs almost nothing (+0.007). The score says which weights moved, not which weights mattered.

So this is an exact description of what changed and a poor predictor of what it will cost. Use it
to answer "did this transformation preserve the map, and where did it not", not to allocate bits.

```python
import entroptics_jlens as je
before = model.transformer.h[11].attn.c_proj.weight.detach().numpy()
after  = quantized.transformer.h[11].attn.c_proj.weight.detach().numpy()
je.principal_angles(before, after, k=64).mean()        # 0.3893
je.participation_ratio(je.energy_spectrum(after))      # 4.2, from 27.2
```

The same comparison answers *did my fine-tune change anything structural, and where?* and *did
distillation preserve the map?* — a weight-norm delta says "the weights moved 0.3%", which is not
an answer to either.

**What this is not:** a replacement for your eval run. These numbers say precisely what changed in
the weights. Whether a given agreement score predicts task loss is an experiment nobody here has
run, and it is the one that would turn a measurement into a decision procedure.

### At inference time: 0.93 microseconds a token

The reads above are offline, on a weight file. **The runtime half is a projection**, because the
SVD is preprocessing — it happens once when you load the model, not per token. Measured on gpt2
layer 6, `d=768`, `k=48`:

| | |
|---|---|
| build the workspace, once per model | 1058 ms |
| **extract, per token** | **0.93 µs** (48 coordinates) |
| inject, per token | 4.48 µs |
| gpt2's own forward pass | 2242 µs/token |

**Extract is 0.054% of the forward pass.** This is affordable on every token of every request.

```python
ws = je.workspace(lens.jacobian(6), layer=6)   # once, at load

coords = ws.extract(h)                          # (T, 48) — where these tokens sit
h2 = ws.inject(h, direction=0, amount=5.0)      # move one coordinate
```

**Inject is surgical.** The basis is orthonormal, so moving direction 0 by 5.0 moves that coordinate by exactly 5.0 and every other one by at most **1.8e-15** — machine precision rather than approximation. That is what makes it a controlled write instead of a perturbation, and the tests assert it.

**Two things to know before building on it.**

*Where you write matters more than what you write.* A computed vector written at the **embedding**
reproduces a real token 100% of the time on a copy task; the same vector written four blocks later
reproduces it **0%** of the time. Depth is where writes stop carrying. `inject` returns a vector
and does not choose a site — you do.

*The basis is barely better than a random one at summarising a stream.* Measured task-free, as the share of a real residual stream's variance that survives projection onto k directions:

| layer | k | random | workspace | stream's own PCA |
|---|---|---|---|---|
| 2 | 59 | 0.0767 | 0.0770 | 0.9409 |
| 6 | 48 | 0.0686 | 0.0820 | 0.8652 |
| 10 | 39 | 0.0511 | 0.0802 | 0.7600 |

**1.0–1.6× a random subspace, and 8–11% of the optimal one.** So the coordinates `extract` returns are not a compressed summary of what the model is carrying — an arbitrary projection of the same width carries nearly as much.

This does not contradict the `coverage` result above, which reads the overlap of *resolved subspaces* and finds the real transport at 6–24× chance. Both hold because the stream's resolved directions are not where its variance is. What it rules out is using these coordinates as a low-dimensional summary. What it does not rule out is that one particular direction in the basis is individually meaningful — that needs a labelled concept and is a different experiment.

## What it measures, and at what level

Every read here characterises a **map** — a transport, a pair of fits, a truncation — at the level
of a frame. That is where the numbers hold and where they are exact: how much of a transport is the
skip connection, whether two independently fitted lenses are the same map, whether a rank
truncation preserved the structure it was meant to keep. None of these needs a predictive link to
behaviour, and every read prints its control beside it.

Scoring an individual token is a different question, and the reads here are frame-level
instruments. `coverage` compares two frames; `audit` describes one matrix; `compare` decides
whether two maps agree. Use them at that level.

**A mean-Jacobian lens is a weak per-token linearisation in the lower half of a model.** Measured
as explained variance of the token-varying part of the final residual, a shared linearisation
accounts for under 5% at and below layer 8 on Qwen3.5-4B, crossing half the variance only at
relative depth 0.90. The same profile appears on Qwen3.5-0.8B at 2.5× less width, so it is a
property of the method rather than of one network. Where a lens is most wanted — early and middle
depth — a linear transport reproduces little of the token-to-token variation, and the paper's §7
gives the whole curve.

**The identity result applies where the identity is.** Below relative depth 0.6 the raw and
residual reads agree, so the screen is worth running before the spectrum: `identity_share` is
O(d²) and answers it in 9.5 ms at d=2560 against 3.5 s for the SVD.

## Using it as a library

The CLI is a thin front on an API you can call directly.

```python
import numpy as np
import entroptics_jlens as je

lens = je.load_lens("lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt")
J = lens.jacobian(5)                     # float64, one layer at a time

# Always first. Everything below reads M, not J.
d = je.decompose(J)
M = d.residual
d.alpha, d.removed_energy, d.identity_dominated       # 0.801, 0.278, False

je.participation_ratio(je.energy_spectrum(J))         # 41.6  -- effective rank, with the identity
je.participation_ratio(je.energy_spectrum(M))         # 22.9  -- and without it
je.transport_spectrum(M, null="mp", far=0.05).K       # 51    -- a lower bound, at this far

# How far into the spectrum two readouts agree. Cosines descend; find where they fall below 0.9.
c = je.principal_angles(M, je.decompose(lens.jacobian(6)).residual, k=64)
int(np.flatnonzero(c < 0.9)[0])                       # 25 of the leading 64 directions

# H is a (T, d) residual stream you collected; give it at least d tokens.
cov = je.coverage(H, H @ M.T)
cov.coverage, cov.null, cov.k_signal                  # 0.1557, 0.0156, 23
```

Those are the real values from that lens, so a copy-paste that disagrees is telling you something.

Two things the API will not stop you doing, and should not be done: reading a spectrum on `J` rather than `M`, and reading `null="robust"` as though it carried a false-alarm rate. Both are covered above.

A ~100-layer lens at `d_model = 5120` is about 5 GB stored and 21 GB upcast, so `LensFile` memory-maps the checkpoint and upcasts one transport on request. Nothing here materialises them all.

### Which file answers which question

Each module's docstring carries the measurement behind its design, including the ones that went wrong.

| | |
|---|---|
| [`decompose.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/decompose.py) | the identity the architecture puts there, and why it comes off first |
| [`transport.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/transport.py) | resolved rank against a derived floor — and why `mp` and the fence are different questions |
| [`spectra.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/spectra.py) | the threshold-free reads: participation ratio, Shannon rank, principal angles |
| [`coverage.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/coverage.py) | subspace overlap and its analytic chance level, with the coverage read it supports |
| [`nulls.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/nulls.py) | distribution-free floors, the magnitude-preserving surrogates, **and where they collapse** |
| [`controls.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/controls.py) | the entropy-matched nulls, and the one caveat behind all of them |
| [`lenses.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/lenses.py) | the transport as an Entroptics `Lens`; `truncated_pair`, and why a plain `pinv` is not a projector |
| [`io.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/io.py) | reading a `lens.pt`, one transport at a time, at float64 |
| [`frames.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/frames.py) | the boundary: what is upcast, and what is refused |
| [`targets.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/targets.py) | comparing a transport against the model's own final residual, on the same footing |
| [`bench.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/bench.py) | the sealed half that opens once |
| [`results.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/results.py) | results files that say whether their run finished |
| [`cli.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/cli.py) · [`catalog.py`](https://github.com/Agience/entroptics-jlens/blob/main/src/entroptics_jlens/catalog.py) | the three commands, and the published-lens table |

`je.Bench` is separate and worth knowing about if you are measuring an effect rather than a matrix: it seals half your items at construction and opens them once, and it records claims as **rates** rather than magnitudes. Rates transferred across sealed halves five times in this work where the corresponding means halved, doubled, and once reversed sign.

## Everything refuses rather than guesses

A missing lens file, a fitting checkpoint mistaken for a lens, a non-square transport, a transport that resolves nothing, an unknown null, streams of the wrong width or depth, a coverage read on a frame that resolves nothing: each stops with a message naming what was found. There are 55 such refusals, and [tests/test_refusals.py](https://github.com/Agience/entroptics-jlens/blob/main/tests/test_refusals.py) exercises the ones nothing else reached — asserting the message rather than the exception type, because a refusal whose text is wrong has failed at the only job it has. Nothing truncates, imputes, or substitutes a plausible value, and no `nan` is ever printed under a real heading. A run that quietly read a subset of layers would report a curve with holes in it as if it were the curve.

The prose is linted like the code: [tests/test_docs.py](https://github.com/Agience/entroptics-jlens/blob/main/tests/test_docs.py) checks every markdown file for literal tabs, stray carriage returns, ragged tables and dead relative links. Every one of those defects has actually happened here.

## Where the numbers come from

- [research/PAPER.pdf](https://github.com/Agience/entroptics-jlens/blob/main/research/PAPER.pdf) — the paper as a PDF.
- [research/PAPER.md](https://github.com/Agience/entroptics-jlens/blob/main/research/PAPER.md) — the same write-up in markdown, and the record of what every number was measured on. §1 is the claim, the four checks that bound it, and what the recovered directions contain; §2 is what Entroptics supplies; §9 is validation against planted truth.
- [research/experiments/](https://github.com/Agience/entroptics-jlens/tree/main/research/experiments) — every figure above, as a script that re-runs it.

The reads themselves come from [entroptics](https://pypi.org/project/entroptics/), which is the engine: the noise floors, the null providers, the screen. This package is the domain wrapper — it knows what a Jacobian lens is, and entroptics does not.

## Citation

> Sessford, I. J. *The identity component of a Jacobian lens inflates the noise floor computed
> from it.* 2026. [doi:10.5281/zenodo.22293929](https://doi.org/10.5281/zenodo.22293929).

That is the concept DOI and it resolves to the newest deposit; each release also gets its own
version DOI, listed on the Zenodo record.
[`CITATION.cff`](https://github.com/Agience/entroptics-jlens/blob/main/CITATION.cff) carries the
machine-readable form, which GitHub's "Cite this repository" reads.

This work is built on **Entroptics**, which supplies every resolved-rank count in it — cite that
too:

> Sessford, I. J. *Entroptics: Reading any 2-D signal as a finite optical aperture at its own
> entropy-matched resolution.* 2026. [doi:10.5281/zenodo.21273400](https://doi.org/10.5281/zenodo.21273400).
> Software: <https://pypi.org/project/entroptics/> (this work uses 0.2.1).

## License

Apache-2.0 — see [LICENSE](https://github.com/Agience/entroptics-jlens/blob/main/LICENSE) and [NOTICE](https://github.com/Agience/entroptics-jlens/blob/main/NOTICE). Contributing: [CONTRIBUTING.md](https://github.com/Agience/entroptics-jlens/blob/main/CONTRIBUTING.md).
