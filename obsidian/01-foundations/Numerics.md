---
tags: [layer/foundations, concept]
code: data_structure/Numeric.py
status: stable
---

# Numerics

## What it is

Symbolic arithmetic over `Term`s: array sizes, loop repetitions and the formulas an
operator carries. `Numeric` is a small closed algebra of free symbols, integers, addition,
multiplication, powers and logarithms, kept symbolic so that a size can be an unknown `|q|`
through every rewrite and be given a value by a consumer alone.

## Where it lives

`data_structure/Numeric.py`.

| name | what it is |
|---|---|
| `FreeNumeric` | a symbol. `FreeNumeric.named(x)` derives its id from the name via `hash_id`, so it is **stable across calls and across processes** |
| `Integer` | a literal |
| `FreeInput` | **the** variable of a formula, printed `x`. No uid — every `FreeInput` is the same input, which is what makes a formula a function of one argument. `nm.x` |
| `Constant`, `ConstantSymbol` | a named mathematical constant. `ConstantSymbol.EULER` prints `e` and `ConstantSymbol.PI` prints `\pi`. `Power(Constant(), x)` is the exponential, `Logarithm(Constant(), x)` the natural log; `Logarithm.template(b, b)` folds to 1. `nm.E`, `nm.PI` |
| `DimensionOfMeasure`, `UnitOfMeasure` | a kind of quantity, and a unit of it as a constant whose value is its scale in the reference units. `3 * GIBIBYTE` is a `Multiplication`. See [[Units of Measure]] |
| `CumulativeGaussian` | the standard normal distribution function, printed `\Phi(argument)`. It has no closed form in the other numerics, so it is a numeric of its own. `nm.standard_normal_density` writes its derivative, `(2 pi)^{-1/2} e^{-u^{2}/2}` |
| `Expandable` | the base of a numeric that the primitive numerics can also spell. Every numeric that is not an `Expandable` is primitive. See [[#Expandable numerics]] below |
| `Sigmoid` | the logistic function, an `Expandable` printed `\sigma(argument)` and expanded to `(1 + e^{-argument})^{-1}`. `nm.logistic_density` writes its derivative as `\sigma(u) (1 - \sigma(u))`. See [[#Writing a derivative through the forward term]] below |
| `RectifiedLinear` | the rectified linear function, an `Expandable` printed `\mathrm{ReLU}(argument)` and expanded to `argument \mathbbm{1}_{argument > 0}`. `ops.ReLU` carries the short name, and a `Term` class name is registered once, so the numeric carries the full one |
| `Clamp` | the argument held between `lower` and `upper`, an `Expandable` whose default bounds are 0 and 1. The default prints between corner brackets alone, `\ulcorner argument \lrcorner`, and other bounds are written on the closing bracket. It expands to `lower + (u - lower) \mathbbm{1}_{u - lower > 0} - (u - upper) \mathbbm{1}_{u - upper > 0}`. See [[#Expandable numerics]] below |
| `Sign` | one above zero, minus one below zero and zero at zero, an `Expandable` printed `\operatorname{sign}(argument)` and expanded to `\mathbbm{1}_{argument > 0} - \mathbbm{1}_{-argument > 0}`. Both indicators differentiate to zero, so the sign does too |
| `AbsoluteValue` | the magnitude of the argument, an `Expandable` printed `\lvert argument \rvert` and expanded to the argument times its sign. Its derivative comes out as `\operatorname{sign}(argument)` times the argument's derivative |
| `LargerOf` | the larger of `first` and `second`, an `Expandable` printed `\max(first, second)` and expanded to `first + (second - first) \mathbbm{1}_{second - first > 0}`. `ops.Maximum` is the reduction operator and a `Term` class name is registered once, so the numeric is named for what it returns |
| `SquareRoot` | the square root of the argument, an `Expandable` printed `\sqrt{argument}` and expanded to the power `argument^{1/2}`, which `Power.to_latex` prints as the same root |
| `IsPositive` | the indicator `\mathbbm{1}_{argument > 0}`, which is one above zero and zero elsewhere |
| `Conjugate` | the complex conjugate of the argument, printed `\overline{argument}`. It is primitive, because every primitive reads a number whole and none of them takes a complex number apart. The table of an inverse rotary embedding holds `\overline{e^{\mathrm{i} x}}` of the angle, which is the factor of the counterclockwise table at the same position and pair turned back |
| `Associative` | base for `Addition` (identity 0) and `Multiplication` (identity 1, absorbing 0) |
| `Power`, `Logarithm`, `division` | the rest |
| `Equality` | a pair asserted equal, which `Operator.sizing_rules` states an axis size with |
| `canonical_order` | the sort key that makes commutative operands comparable |
| `expand`, `expand_to_addition`, `apply_equalities`, `contains_free_input`, `map_direct_subterms`, `expand_every_expandable`, `substitute_subterm`, `outer_function` | the explicit-algebra section. [[#Expandable numerics]] describes the last four |
| `cancel_reciprocal_factors` | a product with every symbol cancelled against its own reciprocal, so `\hat{v} 2^{63} \hat{v}^{-1}` becomes `2^{63}`. It is algebra applied explicitly, per the rule below, and a factor with an integer base is kept as it stands so that a bound written as a power survives the JSON transport. The Engram hash uses it to state that an identifier below `\hat{v}` times a multiplier below `2^{63} / \hat{v}` is below `2^{63}` |
| `dimensions_of`, `convert_to_unit`, `DimensionMismatch` | dimensional analysis, applied explicitly. [[Units of Measure]] describes them |
| `solver.algebra.differentiate_numeric.differentiate` | d/dx of a formula, one rule per class. See [[#Differentiating a formula]] below |

`solver/numeric_solver.py` is where a numeric is solved.
`display/display_numeric.py` renders a numeric.

## Differentiating a formula

`solver/` holds the base-numeric algebra, meaning everything that is genuinely algebra
over `Numeric`s rather than the canonicalisation `Numeric.template` does. It imports
nothing but `Numeric`, so `data_structure.Operators` reaches it without a cycle. It is
laid out in the three subfolders `CLAUDE.md` lists:

| module | contents |
|---|---|
| `solver/data_structure/numeric_derivative_rule.py` | the `Rule` type, the `RULES` table, the `register` decorator and `rule_for`, which walks the method resolution order |
| `solver/registries/numeric_derivative.py` | one rule per `Numeric` class. A class added outside `solver` is differentiated by adding a row rather than by editing the pass. Import the module for its side effect |
| `solver/algebra/differentiate_numeric.py` | `differentiate`, which decides only whether a subterm is constant, through `nm.contains_free_input`, and hands every other case to its rule, beside `natural_logarithm` |

The table sits in `data_structure/` because both of the other two read it and neither may
import the other. A rule differentiates its own subterms, so the registry imports the
algebra, and the algebra looks a rule up through the table.

The rules are:

| class | derivative |
|---|---|
| `Integer`, `FreeNumeric`, `Constant`, `UnitOfMeasure` | 0. A `FreeNumeric` is a size symbol rather than a variable, and a `UnitOfMeasure` is a scale |
| `FreeInput` | 1 |
| `Addition` | the sum of the parts' derivatives |
| `Multiplication` | the product rule, one term per part |
| `Power` | `b^e (e' ln b + e b'/b)`, which collapses to `e^{x} -> e^{x}` |
| `Logarithm` | `a'/(a ln b) - b'/(b ln b) * log_b(a)` |
| `CumulativeGaussian` | `standard_normal_density(u) * u'` |
| `Sigmoid` | `logistic_density(u) * u'` |
| `IsPositive` | 0. The step at zero has no derivative, and zero there is the convention [[Derivatives]] justifies for a ReLU |
| `Conjugate` | the conjugate of the argument's derivative. Conjugation is additive and commutes with multiplication by a real number, so it passes through a derivative taken with respect to a real variable |
| `Expandable`, for a class with no rule of its own | the derivative of `expand_to_primitives()`. `RectifiedLinear` takes it: `u [u > 0]` differentiates to `u' [u > 0] + u \cdot 0`, which `template` folds to `[u > 0] u'`. `Clamp` takes it too, and differentiates to `([u - lower > 0] - [u - upper > 0]) u'` |

Results go through `template`, so `e^{x}` differentiates to literally `e^{x}` and
`x^{-1}` to `-x^{-2}`. The GELU, written `\Phi(x) * x`, differentiates to
`\Phi(x) + x (2 pi)^{-1/2} e^{-x^{2}/2}`, which agrees with `torch.autograd`. The constant-exponent power rule is written out rather than
derived, because collecting `n b^{-1} b^{n}` into `n b^{n-1}` is algebra `template` does
not do.

## Writing a derivative through the forward term

`Sigmoid` is a numeric the other numerics can spell. `(1 + e^{-u})^{-1}` is
the logistic function, and the general power rule differentiates it to
`e^{-u} (1 + e^{-u})^{-2}`. The two terms in that derivative appear nowhere in the
formula it came from, so a pass comparing a backward subterm against a forward one finds
nothing shared.

`nm.logistic_density` writes the derivative as `\sigma(u) (1 - \sigma(u))` instead. The
derivative then holds the same `Sigmoid` term the formula holds, and terms are shared
nodes of one directed acyclic graph, per the *Term sharing* invariant in [[Invariants]],
so the two are one node rather than two equal ones. The SiLU is the case this was added
for, and a model writes it as
`ops.Arithmetic.template(nm.Sigmoid(nm.x) * nm.x, name='SiLU')`, and its derivative is
`x \sigma(x) (1 - \sigma(x)) + \sigma(x)` over the same `\sigma(x)`.

[[Pathway Collapse]]'s `split_off_forward_formulas` uses the sharing. A backward
`Arithmetic<\sigma(x) (1 - \sigma(x))>` on a taped `x`, beside a forward
`Arithmetic<\sigma(x)>` on the same `x`, becomes the forward map followed by
`Arithmetic<x (1 - x)>`, and `migrate_drops` then reads `\sigma(x)` off the tape. The
SiLU's derivative also reads `x` outside `\sigma(x)`, so it is not a function of the
SiLU's value, and it keeps its form.

## Expandable numerics

A numeric that the primitive numerics can also spell derives from `Expandable`, which
declares `expand_to_primitives`. `Sigmoid` expands to `(1 + e^{-u})^{-1}` and
`RectifiedLinear` expands to `u [u > 0]`. Every numeric that is not an `Expandable` is
primitive, including `IsPositive` and `CumulativeGaussian`, which has no closed form.

An `Expandable` stays whole in a formula and in the formula's derivative, so
`\sigma(x)` and `\mathrm{ReLU}(x - 10)` read as the functions they are, and their
derivatives, `\sigma(x) (1 - \sigma(x))` and `[x - 10 > 0]`, hold the same terms. A pass
calls `expand_every_expandable` only to compare two formulas that may be spelled
differently, and keeps both formulas as written when the comparison fails.
[[Pathway Collapse]]'s `split_off_forward_formulas` is that pass. It looks for a forward
formula `g` with `f = h(g)` among the formulas as written, then among their expansions,
and leaves `f` as written when neither search finds one.

Four functions in the explicit-algebra section carry the comparison.
`map_direct_subterms` rebuilds a term through `template`, so every result is canonical,
and returns the term itself when nothing changed. `expand_every_expandable` applies
`expand_to_primitives` to every node of a formula. `substitute_subterm` replaces every
occurrence of a subterm, where a sum or a product also occurs as a sub-multiset of the
operands of a sum or product of its kind. `outer_function(composite, inner)` replaces
`inner` with a fresh symbol, and returns what remains, with the symbol written as `x`,
only when the free input appears nowhere else.

A clamp on one side is written with `RectifiedLinear`. `x - \mathrm{ReLU}(x - L)` clamps
from above at `L`, and differentiates to `1 - \mathbbm{1}_{x - L > 0}`. The clamp
written with a root, `(x + L - \sqrt{(x - L)^2}) / 2`, has the same value, but a reader
has to work out that it is a clamp, and its derivative is `0 \cdot \infty` at `x = L`.

A clamp between two bounds is a `Clamp`, since 2026-09-17. `Clamp(u)` holds `u` between
0 and 1, which is the clamp a ramp takes, and `Clamp(u, lower, upper)` holds it between
any two bounds, with `lower` assumed to be at most `upper`. The expansion is written in
`IsPositive` terms directly. Below `lower` both indicators are zero and the value is
`lower`. Between the bounds the first indicator is one, and `lower + (u - lower)` is
`u`. Above `upper` both are one, the two copies of `u` cancel, and the value is `upper`.
`torch_compile.numeric_torch` evaluates it with `torch.clamp`, and
`para/validate_backward.py` checks the value and the derived gradient of the default
clamp and of a clamp between `-1/2` and `1/2` against `torch.autograd`. The corner
brackets are `\ulcorner` and `\lrcorner`, which KaTeX typesets, and `tsncd` mirrors the
class in `Numeric.ts` with a case in `NumericRenderer.ts`.

A conjugate is a `Conjugate`, since 2026-09-18. The rotary table of DeepSeek-V4.1-Flash
turns a pair of channels counterclockwise by an angle that grows with the position, and
the table the model multiplies its attention output by turns the pair back. The second
table holds the conjugate of every entry of the first, so its last elementwise map is
`\overline{e^{\mathrm{i} x}}` and the operator itself is named under a conjugate bar
through `fd.DynamicNameSettings.overline`.

The sign, the magnitude, the larger of two values and the square root are `Sign`,
`AbsoluteValue`, `LargerOf` and `SquareRoot`, since 2026-09-19. The user asked for them
so that Engram's gate, `\sigma(\operatorname{sign}(d) \sqrt{\max(\lvert d \rvert,
\varepsilon)})`, reads as the reference writes it, where it had been written by hand as
`2 \mathbbm{1}_{d > 0} - 1`, a product with that sign, and `\varepsilon + \mathrm{ReLU}(u
- \varepsilon)`. Each is an `Expandable`, so the formula keeps the symbol and the
expansion exists in `expand_to_primitives` alone. The user ruled the same day that an
inspection box does not spell such a function out, because every one of them is a
function a reader knows, so `notebooks/display/explain_operators.written_formula` writes
the formula as the model wrote it and appends no expansion. The sign of zero is zero, as
`torch.sign` has it, so the gate at a dot product of zero is `\sigma(0)` where the
reference's `copysign` gives `\sigma(\sqrt{\varepsilon})`. `torch_compile.torch_compile`
evaluates the four with `torch.sign`, `torch.abs`, `torch.maximum` and `torch.sqrt`, and
`para/validate_backward.py` checks the value and the derived gradient of each, and of the
gate, against `torch.autograd`. `tsncd` mirrors the four in `Numeric.ts` with a case each
in `NumericRenderer.ts`.

`IsPositive` prints as the indicator `\mathbbm{1}_{u > 0}`, the `bbm` package's command,
ruled on 2026-09-14. `\mathbb{1}` denotes the universal unit in [[The Universal Unit]], so
the indicator uses the other command and the two stay apart in a figure. KaTeX has neither
the command nor a double-struck digit, and `tsncd` supplies `\mathbbm` as a macro in
`display/HTMLRender/katex_options.ts` that overlaps two ones into the doubled stem.

## How a numeric prints

`to_latex` writes a negative term with one minus sign, in the two places a sign can
appear. `is_negative` is the single test both read. An `Integer` below zero is negative,
and a `Multiplication` is negative when an odd number of its factors are. `without_sign`
returns the term with the sign taken off every negative factor, so `-2 * x` and `-2 * -x`
both become `2 * x`. The writer places the sign once, from `is_negative`, and writes the
unsigned term after it. Until 2026-09-17 the test was `negated_part`, which recognised a
constant of `-1` and no other, so a sum printed `a + -2 b`, and a product built with two
negative factors printed `-2 -x`, which reads as a subtraction.
`data_structure/validate_numeric_signs.py` asserts the strings.

`Multiplication` writes a product by juxtaposition, as mathematics writes one, so a stride
reads `3pq`. It writes `-1 * x` as `-x`, `-2 * x` as `-2 x`, and brackets a factor that is a sum. `sep` is the
star that `display.display_numeric` and `display.node_category` write, and it is no longer
the latex. `Addition` joins each term after the first with `signed_latex`, which writes a
negated term as a subtraction, so the derivative of `x \sigma(x)` prints
`x (1 - \sigma(x)) \sigma(x) + \sigma(x)` where it once printed `1 + -\sigma(x)`. An
`ops.Arithmetic` takes its name from the formula's latex, so the spelling is what a box
carries in a diagram and what `agent_display` lists.

`Power` writes a power of one half as a root, `\sqrt{base}`, and `Logarithm` writes a
logarithm to the base $e$ as `\ln(argument)`, since 2026-09-17. The inspection box over an
elementwise map writes the map's formula, per [[Advanced Display]], and the score function
of the DeepSeek-V4.1 router read `log_{e}(1 + e^{x})^{2^{-1}}` there. A logarithm to any
other base is written `log_{b}(argument)` as before. A quotient is still written as a
product with a negative power, so the route scale reads `3 x 2^{-1}`.

`tsncd` renders a numeric of its own for an axis size, a stride and a shift, and writes the
same product and the same subtraction. The juxtaposition was its own until 2026-09-14, when
the star went from the latex here. The root and the natural logarithm are written here
alone, because no size, stride or shift of the package holds either.

## The rule that matters

**Numerics compare structurally, and `template` canonicalises.**

`Addition.template` and `Multiplication.template` flatten nested associatives, drop
identities, fold integer constants, and sort commutative operands into `canonical_order`. So
`x + y` really is `y + x`, and `1 + -1` really is `0`. The canonicalisation is normalisation: it
makes structural equality agree with arithmetic on the cases decidable by looking.

Anything deeper, such as collecting like terms, cancelling `x/x`, or deciding whether two symbolic
expressions denote the same function, is genuine algebra. It lives in the numeric-algebra
ALGEBRA* section at the foot of the file and is **applied explicitly**. Do not put it in
`==`.

> [!warning] The history is the argument
> Equality used to go through a `numeric_hash` taken modulo `2**16-1`, with addition
> summing its parts' hashes and multiplication multiplying them, so `2x + 2x == 4x` fell
> out arithmetically. It bought a little algebra and cost correctness: it never handled
> division (`x/y*y != x`) or powers (`x*x != x**2`), and being a hash modulo a small
> number it reported `Integer(65535) == Integer(0)`, which is a silently wrong answer in a value
> used for tile counts and array sizes.

Because of that history `Numeric` sets `_memoize_hash = False` in the sense that its
equality story is its own. `data_structure/Numeric.py` and [[Terms]] carry it.

## Where numerics show up

- `Axis._size` — every axis has one; `Axis.named('q')` gives it a `FreeNumeric` printed
  with absolute bars, `|q|`.
- `BlockTag.repetition` — a repetition other than 1 means the block **is a loop**
  ([[Hypergraphs]]).

## See also

- [[Terms]] — the base
- [[Stride Category]] — strides and shifts are numerics
- [[Units of Measure]] — a unit is a leaf numeric, and a quantity is a numeric times a unit
