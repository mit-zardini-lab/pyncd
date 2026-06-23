# GNNs as Programs in a `D`-Graded Colored PROP

This note gives a detailed, constructive interpretation of graph neural networks (GNNs) inside the `D`-graded colored PROP framework developed in this repository.

It is best read as a **proposed semantics**: highly plausible and aligned with categorical treatments of typed compositional systems, but still a modeling choice that should be stress-tested on concrete architectures.

---

## 1. Executive idea

A message-passing GNN layer is represented as a morphism in a colored PROP:

```text
Layer : NodeState ⊗ EdgeState ⊗ GraphStruct → NodeState
```

with internal factorization:

```text
message   : NodeState ⊗ NodeState ⊗ EdgeState → Msg
aggregate : Multiset(Msg) → AggMsg
update    : NodeState ⊗ AggMsg → NodeState
```

The graph instance determines **wiring** (which messages flow to which nodes), and PROP composition stacks layers, skip connections, heads, and readouts.

`D`-grading contributes an independent axis of structure (distance, scale, order, time, etc.), so the same typed layer can be lifted/broadcast over that axis.

---

## 2. Why this is a natural fit

GNNs have exactly the ingredients PROPs are built for:

1. **Many-input/many-output operations** (`m → n`) such as residual branches, multi-head channels, and joint node/edge updates.
2. **Parallel composition** (tensor product) for independent streams.
3. **Sequential composition** for stacked layers.
4. **Permutation symmetry** from graph relabelings.
5. **Typing constraints** (node features cannot be silently wired into edge logits without an explicit map).

Colored PROPs model all of these directly.

---

## 3. Object and color design

Choose a set of colors (wire types), e.g.

```text
N   : node feature channel
E   : edge feature channel
G   : global graph feature
M   : message channel
A   : aggregated message channel
Y   : prediction channel
```

Objects are finite tensor words over colors (`N ⊗ E ⊗ G`, etc.).
Morphisms are typed programs on these objects.

For heterogeneous graphs, refine colors by entity/relation:

```text
N_user, N_item, E_click, E_purchase, ...
```

Then illegal mixes are rejected structurally (no morphism unless explicitly provided).

---

## 4. What the grading `D` can mean for GNNs

`D` tracks extra index structure orthogonal to local message functions.

Common choices:

1. **Hop distance grading** (`k = 0, 1, ..., K`): run/sum channels per distance.
2. **Scale grading** (cluster levels or coarsenings): multiresolution GNNs.
3. **Path/order grading** (walk length, motif order): higher-order message channels.
4. **Time grading** (snapshot index): temporal/dynamic graphs.
5. **Frequency grading** (spectral bands): filter-bank style graph signal models.

The lift action (`⊛`) says: "apply the same structural operation across each degree index."

---

## 5. Canonical message-passing decomposition

For directed graph \(G=(V,E)\), one layer:

```text
m_{u→v} = φ_msg(h_u, h_v, e_{u→v})
ā_v     = ⊕_{u∈N(v)} m_{u→v}
h'_v    = φ_upd(h_v, ā_v)
```

In PROP terms:

```text
MsgOp  : N ⊗ N ⊗ E → M
AggOp  : Bag(M) → A
UpdOp  : N ⊗ A → N
Layer  := (wire graph incidences) ; MsgOp ; AggOp ; UpdOp
```

`Bag(M)` is implemented by explicit fan-in wiring + a commutative/associative aggregator (sum/mean/max/attention-weighted sum).

---

## 6. Worked example: vanilla GCN

GCN update:

```text
H' = σ( D̃^{-1/2} Ã D̃^{-1/2} H W )
```

PROP factorization:

1. `Linear : N → N` (right multiplication by `W`)
2. `Diffuse : N → N` (fixed graph-linear map using normalized adjacency)
3. `Act : N → N` (`σ`)
4. `GCNLayer = Linear ; Diffuse ; Act`

Graph dependence is in `Diffuse`'s wiring coefficients; learnable part is mostly `Linear`.

If degree-graded by hop radius `k`, lift to:

```text
[GCNLayer, k] : N⊛k → N⊛k
```

and combine channels:

```text
MergeHop : (N⊛0) ⊗ ... ⊗ (N⊛K) → N
```

---

## 7. Worked example: GAT as typed attention generator

For each edge \((u,v)\):

```text
q_u = W_q h_u,  k_v = W_k h_v,  v_u = W_v h_u
α_{u→v} = softmax_u( score(q_u, k_v, e_{u→v}) )
m_{u→v} = α_{u→v} · v_u
h'_v = φ_upd(h_v, Σ_u m_{u→v})
```

Generators:

```text
QProj, KProj, VProj : N → N
Score               : N ⊗ N ⊗ E → S
Normalize           : StarIn(S) → StarIn(S)    (softmax over in-edges)
WeightValue         : StarIn(S) ⊗ StarIn(N) → StarIn(M)
Reduce              : StarIn(M) → A
Update              : N ⊗ A → N
```

Here `StarIn(·)` denotes "edge-indexed family incident to a target node."
Multi-head GAT is PROP tensor parallelism:

```text
Head1 ⊗ ... ⊗ HeadH ; HeadMerge
```

---

## 8. Worked example: heterogeneous relational GNN (R-GCN style)

With relation set `R`:

```text
h'_v = σ( Σ_{r∈R} Σ_{u∈N_r(v)} (1/c_{v,r}) W_r h_u + W_0 h_v )
```

Colored PROP encoding:

```text
N_entityType, E_relationType
Msg_r : N_src(r) ⊗ E_r → M_r
Agg_r : Bag(M_r) → A_r
FuseRelations : A_r1 ⊗ ... ⊗ A_r|R| ⊗ N_target → N_target
```

`D` can grade by relation families, allowing per-family lift and controlled sharing.

---

## 9. Worked example: dynamic graph neural network

Let time axis `t ∈ D`.
At each step:

```text
h_t' = Layer_t(h_t, E_t, G_t)
h_{t+1} = Transition(h_t, h_t')
```

Two patterns:

1. **Untied-time**: distinct layer per `t` (explicit composition chain).
2. **Tied-time**: one step morphism lifted over time, then scanned/folded along `t`.

This aligns with the `Scan` discussion in this repo: recurrent graph processing is a fold over a distinguished temporal degree.

---

## 10. Equivariance and invariance in this language

Graph relabeling by permutation \(\pi\) should satisfy:

```text
Layer(π·x) = π·Layer(x)          (equivariance)
Readout(π·x) = Readout(x)        (invariance)
```

In PROP terms:

1. Permutations are structural symmetries.
2. Equivariance means layer morphisms commute with those symmetries.
3. Invariant readouts are coinvariant reducers (sum/mean/max/set pooling/attention pooling with symmetric normalization).

This is one of the strongest advantages: symmetry constraints become algebraic equations, not informal implementation wishes.

---

## 11. Readout and task heads as separate morphisms

Separate encoder and head:

```text
Encoder : GraphInput → NodeState
Readout : NodeState ⊗ GraphStruct → GraphState
Head    : GraphState → Y
Model   := Encoder ; Readout ; Head
```

Node classification uses identity/local readout on nodes.
Edge prediction includes pair selection morphisms before a scoring head.
Graph classification uses invariant global readout.

---

## 12. Training semantics

The PROP describes architecture-level composition.
A target algebra (e.g., PyTorch/JAX semantics) interprets each generator as a differentiable map.

Then:

1. A symbolic morphism in the graded PROP becomes executable code.
2. Parameterized generators (`Linear`, `MLP`, `AttentionScore`) become learned tensors.
3. Gradients flow through the interpreted composition.

This is exactly the algebra view already used elsewhere in this repository.

---

## 13. Implementation sketch in pyncd style

At the level of this project's design, a practical route is:

1. **Define colors** for node/edge/global/message channels.
2. **Define primitive generators** (`Msg`, `Agg`, `Upd`, `Readout`, optional `Route`/`Scan` for dynamic settings).
3. **Represent graph incidence** as structural wiring data (acset instance).
4. **Compose** generators with explicit typed boundaries.
5. **Lift over degree** (`⊛`) for hop/time/scale channels.
6. **Compile via algebra** to backend kernels.

Minimal pseudo-DSL shape:

```python
# Pseudocode only
msg = Msg @ IncidenceProject        # (u,v,e) -> m_uv
agg = InEdgeCollect @ SumReduce     # {m_uv}_u -> a_v
upd = NodeResidual @ MLPUpdate      # (h_v, a_v) -> h'_v
layer = msg @ agg @ upd

model = (layer ** L) @ GlobalPool @ Classifier
```

(`@` here means composition in the project's style; actual constructors depend on concrete APIs.)

---

## 14. Design patterns enabled by graded PROP structure

### 14.1 Multi-hop channels without ad-hoc duplication

Instead of manually writing 1-hop/2-hop/3-hop modules:

```text
HopLayer := [BaseLayer, k]
FuseK    := ⊗_{k=1..K} HopLayer ; Merge
```

### 14.2 Shared architecture across scales

Lift one local block across coarsened graph levels, then interleave up/down maps.

### 14.3 Hybrid local-global models

Tensor a local MPNN stream with a global transformer/readout stream, then fuse.

### 14.4 Safety for heterogeneity

Color discipline prevents accidental cross-type message misuse.

---

## 15. Limitations and open points

1. **Aggregator semantics** (`Bag(M)`) must be concretized carefully for sparse batching efficiency.
2. **Attention normalization domains** need explicit incidence objects (which neighborhood? direction? relation subset?).
3. **Dynamic rewiring** (learned edges) may require extra generators beyond static incidence.
4. **Continuous symmetries** and gauge/steerable fields may require richer target categories (`Rep(G)`-valued fibers).
5. **Universality/expressivity** statements must be re-proven in this formulation, not assumed.

---

## 16. Concrete mapping table

| GNN concept | PROP/graded-PROP object |
| --- | --- |
| Node features \(h_v\) | wire color `N` |
| Edge features \(e_{uv}\) | wire color `E` |
| Message function \(φ_{msg}\) | generator `N ⊗ N ⊗ E → M` |
| Neighborhood aggregate \(⊕\) | symmetric reducer `Bag(M) → A` |
| Node update \(φ_{upd}\) | generator `N ⊗ A → N` |
| Multi-head | tensor-parallel branches + merge morphism |
| Skip connection | diagonal/copy + add morphism |
| Graph readout | invariant morphism `N* → G` |
| Layer stack | vertical composition |
| Hop/time/scale channel | degree object in `D` with lift `⊛` |
| Equivariance | commutation with symmetry morphisms |

---

## 17. End-to-end tiny symbolic example

Suppose a toy graph with two node types (`User`, `Item`) and one relation (`rates`).
Colors:

```text
N_u, N_i, E_r, M_i, A_i, Y
```

Morphisms:

```text
MsgRates  : N_u ⊗ N_i ⊗ E_r → M_i
AggRates  : Bag(M_i) → A_i
UpdItem   : N_i ⊗ A_i → N_i
PoolItem  : Bag(N_i) → G
Head      : G → Y
```

Composite:

```text
ItemLayer = MsgRates ; AggRates ; UpdItem
Model     = (ItemLayer ; ItemLayer) ; PoolItem ; Head
```

If we add temporal grading `t`:

```text
TimeLiftedLayer = [ItemLayer, t]
TemporalModel   = Scan(TimeLiftedLayer) ; PoolItem ; Head
```

This gives a typed, compositional, symmetry-aware dynamic recommender GNN skeleton.

---

## 18. Summary

The interpretation is: **GNNs are typed string diagrams whose local primitives are message/aggregate/update generators, and whose global architecture is composition in a colored PROP; `D`-grading lifts this architecture across hop/scale/time/order indices.**

That gives a unifying algebraic language for:

1. homogeneous and heterogeneous message passing,
2. multiscale and temporal variants,
3. explicit equivariance/invariance constraints,
4. backend compilation via PROP algebras.

The main remaining work is empirical and formal: instantiate the generators concretely in this codebase, compile them to tensor kernels, and prove the expected equivariance/expressivity theorems for the chosen `D`.
