---
tags: [layer/caching, concept]
code: notebooks/caching/mamba/, notebooks/sota/KimiK3/delta_rule_scan.py
status: partly implemented
agent: Claude Opus 5.5 (1M context), effort 40, 2026-09-26, curated for the public vault on 2026-10-02
---

# Carrying the State of a Scan Between Passes

A Mamba layer mixes its tokens in two ways. A convolution of a few taps reads the last few tokens, and a selective scan carries a state from each token to the next. This note writes the layer as a morphism, with the scan as a loop whose state is a loop variable on the tape. It then states what `derive_cached_pass` derives on the layer and why it stops at the scan, how the state of the scan is cached between passes, and the changes to the seeds, the cost and the derivation that the state cache implies. `notebooks/caching/mamba/` holds the scan and its evaluation in `torch.float64`. The delta attention of Kimi K3 is a scan of the same form, written by `notebooks/sota/KimiK3/delta_rule_scan.py` after `selective_scan.py`, and `notebooks/website/modern/validate_kimi_k3.py` evaluates it against the reference loop of `fla`, per [[SOTA Model Notebooks]].

The layer is `Mamba` in [mamba_simple.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py) and its scan is `selective_scan_ref` in [selective_scan_interface.py L127-L193](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/selective_scan_interface.py#L127-L193), both at the commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` of `state-spaces/mamba`, read on 2026-09-26. `notebooks/caching/mamba/reference_links.py` pins the line range of every step.

## The Mamba layer computes six steps

The reference's sizes `d_model`, `d_inner`, `d_state`, `dt_rank` and `d_conv` are the axes $m$, $d$, $n$, $r$ and $w$, and the tokens are $x$. Write $s$ for the hidden state the layer reads. Every step but the scan is broadcast over the tokens.

The input projection `in_proj` is one weight whose result the reference cuts in two. Each half is a linear map of its own, $W^{u}$ and $W^{z}$ from $m$ to $d$, per the ruling in [[Representing Models]] that a linear map cut into parts is one linear map per part. The first half $v$ is the branch the convolution reads and the second is the gate $z$.

$$v[i_x, i_d] = \sum_{i_m \in m} W^{u}[i_m, i_d]\, s[i_x, i_m], \qquad z[i_x, i_d] = \sum_{i_m \in m} W^{z}[i_m, i_d]\, s[i_x, i_m]$$

The convolution is depthwise and causal, with a bias per channel, and is followed by the SiLU. Tap $i_w$ of token $i_x$ reads token $i_x - i_w$, and a tap before the first token reads the universal unit, which the sum over the taps ignores, as the reference's padding of `d_conv - 1` zeros adds nothing.

$$u[i_x, i_d] = \mathrm{SiLU}\Big(b^{c}[i_d] + \sum_{i_w \in w} W^{c}[i_w, i_d]\, v[i_x - i_w, i_d]\Big)$$

The reference cuts the result of `x_proj` into a step of rank $\lvert r \rvert$, $B$ and $C$, which are three maps $W^{\Delta}$, $W^{B}$ and $W^{C}$ reading $u$. `dt_proj` raises the step to every channel, and the reference adds its bias and applies the softplus inside the scan.

$$\Delta[i_x, i_d] = \ln\Big(1 + \exp\Big(b^{\delta}[i_d] + \sum_{i_r \in r} W^{\delta}[i_r, i_d] \sum_{i_e \in d} W^{\Delta}[i_e, i_r]\, u[i_x, i_e]\Big)\Big)$$

The decay rates are $A[i_d, i_n] = -e^{A_{\log}[i_d, i_n]}$, from the parameter array $A_{\log}$. The decays and the writes are the following.

$$\bar{A}[i_x, i_d, i_n] = e^{\Delta[i_x, i_d]\, A[i_d, i_n]}, \qquad \bar{B}u[i_x, i_d, i_n] = \Delta[i_x, i_d]\, B[i_x, i_n]\, u[i_x, i_d]$$

The scan writes the state $H[i_x]$ after token $i_x$ from the state after the token before, starting from zero, and reads it out with $C$.

$$H[i_x, i_d, i_n] = \bar{A}[i_x, i_d, i_n]\, H[i_x - 1, i_d, i_n] + \bar{B}u[i_x, i_d, i_n], \qquad H[-1] = 0, \qquad y[i_x, i_d] = \sum_{i_n \in n} C[i_x, i_n]\, H[i_x, i_d, i_n]$$

The layer adds the skip, gates, and projects back to the width of the hidden state.

$$o[i_x, i_m] = \sum_{i_d \in d} W^{o}[i_d, i_m]\, \big(y[i_x, i_d] + D[i_d]\, u[i_x, i_d]\big)\, \mathrm{SiLU}(z[i_x, i_d])$$

## A weight read at the channel of its value is a Linear followed by a diagonal

The taps $W^{c}$ and the skip $D$ are each read at the channel of the value they multiply. Per the ruling in [[Representing Models]] that a contraction against a parameter array is a `Linear` producing the array's axes followed by a diagonal, $W^{c}$ is `Linear((w,), (d,), bias=True)` broadcast over the tokens and the channels, and the view `diag` keeps the entries where the channel it produces is the channel it read. The bias of that `Linear` is the bias of the convolution. $D$ is `Linear((), (d,))` written the same way. $A_{\log}$ passes through $-e^{x}$ before anything reads it, so it is a parameter array, a `Linear` with no inputs.

## The selective scan is a loop whose state is a loop variable

The recurrence reads the state of the token before, so it is not an operator broadcast over the tokens. `selective_scan.selective_scan` writes it as a `cat.Block` repeated $\lvert x \rvert$ times, whose tag names the counter $i_x$, per [[Para Category]]. The state $h$ over $(d, n)$ is a loop variable. A `Para.StreamGrab` of the slot $h$ reads the state the iteration starts from, and a `Para.StreamDrop` writes the state the next iteration starts from. A `Para.Drop` of zeros before the loop starts it, and the initialiser, its drop and the loop stand in one block.

The decays, the writes and $C$ are computed before the loop by operators broadcast over the tokens, so they are arrays over every token when the loop begins. They enter the loop on wires, and iteration $i_x$ reads its token through an `ops.View` whose row has no domain axis and the shift $i_x$, times the identity on the other axes. The view is the read of one index that `move_reads_backwards.index_read` writes, with the counter as the index. A `Para.LoopGrab` was the other candidate. It reads the member of a slot that an iteration wrote with a `Para.LoopDrop`. No iteration writes these arrays, so a `LoopGrab` would need a new convention that the member of a slot is the slice of an array over the tokens, where the view states the read with the meaning a view already has.

Iteration $i_x$ writes $y_t$ at position $i_x$ of an array over the tokens with an `aops.CovariantView` of the same row, which holds the universal unit at every other token. The output $Y$ over $(x, d)$ is a second loop variable, started at zero, which sums the writes, and the loop hands out its value after the last iteration. The loop therefore reads each token through a view at the counter and writes each token through the covariant reading of the same row. `aops.CovariantView` declares the `View` of its row as its reverse derivative.

`notebooks/caching/mamba/evaluate_numerically.py` evaluates the loop with the semantics above. Every iteration reads the loop's domain wires, a stream grab reads the value its slot held when the iteration started, a stream drop's value holds from the next iteration on, and the loop's result is its last iteration's result.

## Deriving the pass stops at the scan and places the cache of the convolution

On the layer as built and on its `ParaWrap` form, `derive_cached_pass` raises `ReadsTheTokenAxisWhole` at a view at the counter. `slide_causal_reads_backwards` passes the loop whole, because the loop receives no read from its results and holds no causal read, and no read reaches its operands. It stopped at the loop with `hypergraph_crawler.RepeatedBlockChangesTheGuide` until 2026-09-28, because the crawl required every repeated block to hand its operands the read its results received, and the loop reads more arrays than it returns. The change was made for the 69 delta layers of Kimi K3, which slide in the same way. Until change 4 below was applied, both raised an `AttributeError` at the first bare `Para.StreamDrop`, because the read crawl had no case for a tape seed. The result of that view is one token's decays and holds no token axis, so the read reaching it has no row onto the tokens, and the crawl takes the view to read every token of its operand. Over its iterations the loop does read every token, one per iteration. The scan is the fold over the earlier tokens that [[Deriving Caches by Dragging the New Tokens]] lists among the cases it does not handle.

On the part of the layer before the scan the derivation succeeds. The convolution composed with the read of the new tokens reads $\lvert x_{\mathrm{old}} \rvert + i_{\mathrm{new}} - i_w$, whose reach is $\lvert w \rvert - 1$, so the crawl places a `Caching` of the result of $W^{u}$ onto $K + x_{\mathrm{new}}$ with $\lvert K \rvert = \lvert w \rvert - 1$, and the convolution reads position $\lvert K \rvert + i_{\mathrm{new}} - i_w$ of it. `placements_by_sliding_caches_back` finds one more placement, on the hidden state, which keeps $\lvert m \rvert$ entries per kept token against $\lvert d \rvert = 2 \lvert m \rvert$ and computes $W^{u}$ again at the kept tokens in every pass. The part after the scan is broadcast over the tokens and is rebuilt over the new tokens with no cache.

The reference keeps `conv_state` over $(d, w)$, the last `d_conv` inputs of the convolution ([L255-L266](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py#L255-L266)). Each step rolls the oldest input out, writes the new input last and convolves all `d_conv` of them ([L214-L221](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py#L214-L221)). The oldest entry is overwritten before a step reads it, so `conv_state` after the roll is the result of the derived `Caching` for one new token, and the entry the reference stores beyond $K$ is never read.

## The state of a scan is the wire of states read one token back

The recurrence reads $H[i_x - 1]$, and the zero the loop starts from is $H[-1]$, the universal unit a read before the first token returns. $H$ is therefore a wire over the tokens that the recurrence reads through a causal read of reach one. By the rule of [[Deriving Caches by Dragging the New Tokens]], the cache of $H$ keeps one earlier token, $H[\lvert x_{\mathrm{old}} \rvert - 1]$. $H$ is prefix-stable in the sense of that note, because $H[i_x]$ depends on the tokens up to $i_x$ alone.

**Lemma.** Write $\Phi_t(h) = \bar{A}_t \odot h + \bar{B}u_t$ for one iteration at token $t$. A loop over the tokens $t_0, \dots, t_{k-1}$ started from $h$ computes $\Phi_{t_{k-1}}(\cdots \Phi_{t_0}(h))$. The loop over a concatenation $x_{\mathrm{old}} + x_{\mathrm{new}}$ started from $h$ is therefore the loop over $x_{\mathrm{new}}$ started from the state the loop over $x_{\mathrm{old}}$ ends with. The loop is a fold in the universal form $F(a, u \mathbin{:} v) = A_F(a, F(a, u), v)$ that [[Design Space]] gives a non-associative operation.

**Theorem.** Let the passes append $x_{\mathrm{new}}^{(1)}, x_{\mathrm{new}}^{(2)}, \dots$. Suppose that at the start of pass $k$ the last entry of the state cache is $H(z)[\lvert x_{\mathrm{old}}^{(k)} \rvert - 1]$, and that the cache is empty when $x_{\mathrm{old}}^{(k)}$ is, so that the grab reads zero. Then the loop of pass $k$ writes $H(z) \circ \iota_{\mathrm{new}}$ and $y(z) \circ \iota_{\mathrm{new}}$, and after the pass the last entry of the cache is $H(z)[\lvert x_{\mathrm{old}}^{(k+1)} \rvert - 1]$.

*Proof.* The inputs the pass's loop reads at $i_{\mathrm{new}}$ are the model's inputs at $\lvert x_{\mathrm{old}} \rvert + i_{\mathrm{new}}$. The operators before the scan are broadcast over the tokens, which is Lemma 1 of [[Deriving Caches by Dragging the New Tokens]], and the convolution reads its cached axis, which is that note's theorem for a kept axis. Iteration $i_{\mathrm{new}}$ starts from $H(z)[\lvert x_{\mathrm{old}} \rvert + i_{\mathrm{new}} - 1]$, by induction on $i_{\mathrm{new}}$, with the cached entry as the base case. It applies $\Phi$ at token $\lvert x_{\mathrm{old}} \rvert + i_{\mathrm{new}}$, so it writes $H(z)$ there, and the read-out writes $y(z)$ there. The model's loop writes $y$ at every token, and the writes of the tokens of $x_{\mathrm{old}}$ fall outside the image of $\iota_{\mathrm{new}}$, so the output of the pass starts from zero over $x_{\mathrm{new}}$. The drop after the loop appends the state after the last new token, which is $H(z)[\lvert x_{\mathrm{old}}^{(k)} \rvert + \lvert x_{\mathrm{new}}^{(k)} \rvert - 1]$. ∎

The scan of a pass therefore starts from the cached state. A `Para.CacheGrab` of the slot $c_{h}$ loads the entry over the kept axis $x_{\mathrm{last}}$ of one token, a view reads position zero of it, and a `Para.Drop` puts it on the slot $h$ where the model drops zeros. The loop runs over $x_{\mathrm{new}}$ and hands out its final state beside the output. A covariant view writes that state at position zero of $x_{\mathrm{last}}$, and a `Para.CacheDrop` appends it to $c_{h}$. Composed with the parts derived before and after the scan, such a pass gives the reference `forward` over every token and the reference `step` for one token, with the kept inputs equal to the last columns of `conv_state` and the last state equal to `ssm_state`. The derivation does not write this scan, and change 3 below states the rule that would.

## A Mamba pass loads the same amount whatever the length of the past

A pass over $\lvert x_{\mathrm{new}} \rvert$ tokens loads $(\lvert w \rvert - 1) \lvert d \rvert$ inputs of the convolution and $\lvert d \rvert \lvert n \rvert$ entries of the state per layer, and appends $\lvert x_{\mathrm{new}} \rvert \lvert d \rvert$ inputs and one state. None of these depends on $\lvert x_{\mathrm{old}} \rvert$. A cache of attention loads every earlier token. The scan can also be written without the loop, as $H[i_x] = \sum_{i_s \le i_x} e^{A\,(S[i_x] - S[i_s])} \odot \bar{B}u[i_s]$ with $S$ the running sum of $\Delta$ over the tokens. Every read of that form is a causal read whose reach grows with the past, so the derivation would cache $\Delta$, $B$ and $u$ at every earlier token, which is correct and grows with the sequence. The loop states the fold, and the state cache follows from it.

## Five changes to the seeds, the cost and the derivation were proposed

The work that wrote this note changed no existing file. Changes 1, 4 and 5 were applied on 2026-09-26 when the work was reviewed, and changes 2 and 3 are open in [[Open Gaps]].

1. `Para.CacheDrop` says that it appends the entries of the tokens of the pass. The state pass appends the entry of its last token alone. A later `Para.CacheGrab` reads the last $\lvert K \rvert$ entries appended, so a drop that appends the last $\lvert K \rvert$ entries of a pass states the same cache. The docstrings of `CacheGrab`, `CacheDrop` and the paragraph on the two in the module docstring of `para/data_structure/Para.py` should say so. The same reading lets a server keep $\lvert w \rvert - 1$ inputs of the convolution, where the reference keeps $\lvert w \rvert$. A pair of seeds for a state, the load of the value the previous pass wrote and the write of the value the next pass starts from, was considered and is not needed, because a state is the window of one token of the wire of states. Applied.
2. `cost_cache_placements.cost_of_a_pass` counts the `Caching` operators of a pass alone, so the load and the append of the state cost nothing in it, and `cache_contents.caches_in_the_order_they_run` raises `RepetitionIsNotAnInteger` at the loop over the new tokens. The count should walk a loop over the tokens once and count every `CacheGrab` and `CacheDrop` outside a `Caching`, the grab at $\lvert K \rvert$ entries and the drop at $\min(\lvert K \rvert, \lvert x_{\mathrm{new}} \rvert)$.
3. `derive_cached_pass.NewTokenCrawler.propagate_category` needs a rule for a repeated block whose repetition is the size of the token axis. It asks every domain array of the loop for the read of the new tokens, rebuilds the body with the token axis and the counter over the new tokens, replaces the start of each loop variable with the load of its cache at position zero of a kept axis of one token, hands the final value out of the loop and appends it with a `CacheDrop`. A loop variable that sums covariant writes over the tokens, as $Y$ does, is rebuilt over the new tokens and started at zero.
4. `move_reads_backwards.ReadCrawler.root_processor` needs a case for a bare tape seed, which reads `input_weaves` and raises `AttributeError` now. A seed whose array holds no token axis passes with no read, and one whose array holds it carries the read of the new tokens, as `NewTokenCrawler.through_para_wrap` does. Applied as `ReadCrawler.through_tape_seed` and `NewTokenCrawler.through_tape_seed`.
5. `factor_of_component` of the reindexing algebra returned a `StrideMorphism` for every factor of a read that is not the identity. A factor whose every row reads one domain axis as itself at unit stride with no shift is a `cat.Rearrangement`, and returning it as one lets tsncd draw the diagonal the derived pass writes after $W^{c}$ and $D$, which it drew as a red hexagon. Applied: `factor_of_component` returns a `pc.Rearrangement` for a factor that only copies or deletes an axis and carries no name.

`torch_compile` cannot evaluate either expression, because it compiles every `ops.View` as the identity, compiles a repeated block only when its repetition is an integer, and has no rule for a tape seed, an `ops.ConstantOp`, an `aops.CovariantView` or a `Caching`. `evaluate_numerically.py` evaluates the operators of this layer alone.

## Open

- The rule of change 3 is not written, so no pass of the layer over new tokens is derived.
- A stack of Mamba layers in a repeated block has not been derived. Each layer would keep its own two caches, as a weight inside a repeated block stands for one weight per layer.
- The placements of the state are not searched. The state could be recomputed from cached inputs, which is the closed form above, and costs a pass that grows with the past.
- The evaluator reads the universal unit as zero, which holds for the sums this layer takes over a guarded axis and not for a maximum or a softmax.

## See also

[[Deriving Caches by Dragging the New Tokens]], [[Caching Between Passes]], [[Para Category]], [[Representing Models]], [[Advanced Axis Dynamics]], [[Design Space]], [[SOTA Model Notebooks]]
