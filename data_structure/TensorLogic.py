from __future__ import annotations
from collections import deque
from dataclasses import dataclass

import data_structure.Term as fd
import data_structure.BroadcastedCategory as bc
import data_structure.ProductCategory as pc
import data_structure.StrideCategory as sc
import data_structure.Numeric as nm
import data_structure.Operators as ops
from data_structure.TensorExpr import TensorRef, IversonBinOp, IversonUnaryOp, _factor_axes, _serialize_iverson


def _iverson_is_materializable(factor: IversonBinOp | IversonUnaryOp) -> bool:
    """True when every axis of this Iverson factor has a concrete integer size.

    Sized Iversons are auto-materialized as constant buffers at compile time
    (ConstructedTensorEquation._factor_slots).  They must not appear as Bool
    domain weaves so that morphism composition sees a pure Reals domain.
    """
    return all(isinstance(ax._size, nm.Integer) for ax in _factor_axes(factor))


# A single tensor logic equation stored as a pyncd Operator.
# Index identity is UID identity: passing the same Axis object in multiple positions
# encodes shared indices. An axis whose UID appears in lhs_indices is retained
# (output); one that appears only in rhs is contracted (summed over).
# Because TensorEquation is a Term, Context.apply() traverses lhs_indices and rhs
# via deep_reconstruct, keeping the equation and its Broadcasted in sync under
# axis unification.
@dataclass(frozen=True)
class TensorEquation(bc.Operator):
    # All fields need defaults: Operator already declares name with a default, and
    # Python dataclasses require that subclass fields without defaults precede those
    # with defaults — satisfied here by giving everything a default.
    lhs_name: fd.DynamicName | None = None
    lhs_indices: fd.Prod[sc.RawAxis] = ()
    rhs: fd.Prod[TensorRef | IversonBinOp | IversonUnaryOp] = ()
    operator: bc.Operator | None = None  # nonlinearity; None means Identity

    def retained_uids(self) -> set[fd.UID]:
        return {ax.uid for ax in self.lhs_indices}

    def contracted_axes(self) -> tuple[sc.RawAxis, ...]:
        retained = self.retained_uids()
        seen: set[fd.UID] = set()
        result = []
        for factor in self.rhs:
            for ax in _factor_axes(factor):
                if ax.uid not in retained and ax.uid not in seen:
                    seen.add(ax.uid)
                    result.append(ax)
        return tuple(result)

    def bc_signature[B: bc.Datatype](
        self,
        signature: str = '',
        datatype: B = bc.Reals(),
        give_names: bool = True,
        array_datatypes: dict[fd.DynamicName, bc.Datatype] | None = None,
    ) -> bc.Broadcasted[B, sc.RawAxis]:
        # `give_names` is accepted for interface compatibility with Operator.bc_signature()
        # (monkey-patched in Operators.py) but ignored — axis names come from UIDs.
        # `signature` must be empty; TensorEquation derives contraction structure from
        # axis UID identity, not from a string.
        #
        # Produces a homogeneously-typed Broadcasted: every weave (input and
        # output) shares the same `datatype`. This is intentional — Broadcasted
        # is parameterised by a single B, and bc_signature() is the path for
        # display, morphism composition, and structurally-typed reasoning.
        #
        # For equations whose tensors have mixed datatypes (e.g. a Natural-valued
        # embedding index alongside Reals-valued weights), use
        # acset.convert.from_tensor_equation() instead. That path produces an
        # SBrInstance where each ArrayRow carries its own datatype_tag, at the
        # cost of losing the Broadcasted type parameter.
        #
        # Converts this equation to a Broadcasted by reading the contraction
        # structure from UID identity. The translation has three parts:
        #
        #   1. degree — the retained axes (lhs_indices), shared as _dom by every
        #      reindexing. Broadcasted.degree() calls iallequals(r.dom() for r in
        #      reindexings), so all reindexings must agree on their domain.
        #
        #   2. weaves — one Weave per input. Each position in the weave is either
        #      WeaveMode.TILED (retained axis: filled from the degree at runtime,
        #      i.e. the output loop index) or a concrete Axis (contracted axis:
        #      kept in place for the operator to sum over). The output weave is
        #      all-TILED because the output shape equals the degree exactly.
        #
        #   3. reindexings — one Rearrangement per input. mapping[i] is the
        #      position within degree of the i-th retained axis of that input,
        #      so Rearrangement.cod() yields the subset of degree axes this input
        #      actually uses. Broadcasted.dom() then calls
        #      weave.imprint_to_degree(reindexing.cod()) to recover the full input
        #      shape by filling TILED slots with those degree axes.
        #
        # operator=self so the full equation is preserved inside the Broadcasted
        # and remains accessible for display and round-trip editing.
        if signature:
            raise ValueError(
                "TensorEquation.bc_signature() derives contraction structure from "
                f"axis UID identity; string signatures are not accepted (got {signature!r})"
            )
        degree = self.lhs_indices
        retained_uid_to_pos = {ax.uid: i for i, ax in enumerate(degree)}
        _dt = array_datatypes or {}

        # Sized-Iverson factors are auto-materialized as constant buffers at
        # compile time and do not need to be caller inputs.  Exclude them from
        # input_weaves / reindexings so the morphism domain is pure Reals and
        # composes cleanly with adjacent morphisms.
        # Unsized Iversons (axes have no concrete size) still appear as Bool
        # weaves — the caller must supply the materialised mask tensor.
        domain_factors = [
            f for f in self.rhs
            if not (isinstance(f, (IversonBinOp, IversonUnaryOp)) and _iverson_is_materializable(f))
        ]

        input_weaves = tuple(
            bc.Weave(
                _dt.get(factor.name, datatype) if isinstance(factor, TensorRef) else bc.Bool(),
                tuple(
                    bc.WeaveMode.TILED if ax.uid in retained_uid_to_pos else ax
                    for ax in _factor_axes(factor)
                ),
                iverson_expr=_serialize_iverson(factor) if isinstance(factor, (IversonBinOp, IversonUnaryOp)) else None,
            )
            for factor in domain_factors
        )
        out_dt = _dt.get(self.lhs_name, datatype) if self.lhs_name else datatype
        output_weave = bc.Weave(
            out_dt,
            tuple(bc.WeaveMode.TILED for _ in degree),
        )
        reindexings = tuple(
            pc.Rearrangement(
                mapping=tuple(
                    retained_uid_to_pos[ax.uid]
                    for ax in _factor_axes(factor)
                    if ax.uid in retained_uid_to_pos
                ),
                _dom=degree,
            )
            for factor in domain_factors
        )
        return bc.Broadcasted(
            operator=self,
            input_weaves=input_weaves,
            output_weaves=(output_weave,),
            reindexings=reindexings,
        )


def _split_nonlinearity(
    eq: TensorEquation,
    array_datatypes: dict | None = None,
    datatype: bc.Datatype = bc.Reals(),
) -> bc.Broadcasted | pc.Composed:
    """Compile one equation, splitting any nonlinearity into a separate step.

    Returns the plain Broadcasted einsum when the equation has no nonlinearity,
    or Composed(einsum, nonlinearity_op) when it does.  The @ composition
    handles autoalignment so axis UIDs are unified across the boundary.
    """
    op = eq.operator
    if op is None or isinstance(op, ops.Identity):
        return eq.bc_signature(datatype=datatype, array_datatypes=array_datatypes)
    # Strip the nonlinearity before building the einsum so ConstructedTensorEquation
    # never sees a non-identity operator.
    bare_eq = TensorEquation(
        lhs_name=eq.lhs_name,
        lhs_indices=eq.lhs_indices,
        rhs=eq.rhs,
        operator=None,
    )
    br = bare_eq.bc_signature(datatype=datatype, array_datatypes=array_datatypes)
    if isinstance(op, ops.SoftMax):
        return br @ ops.SoftMax.template()
    elif isinstance(op, ops.Normalize):
        return br @ ops.Normalize.template()
    elif isinstance(op, ops.Elementwise):   # catches ReLU (subclass)
        return br @ ops.Elementwise.template()
    else:
        raise NotImplementedError(
            f'No base morphism registered for nonlinearity {op!r}; '
            f'compose it as a separate Broadcasted step.'
        )


# Collects a set of TensorEquation objects and converts them to a single
# Composed morphism via to_morphism().
#
# The key problem: each TensorEquation is constructed with its own fresh Axis
# objects. Two equations that both reference a tensor named 'Hidden' will use
# different Python objects for its axes — they need to be unified so the
# resulting Composed is coherent. to_morphism() solves this with a Context:
#
#   1. Topologically sort equations so each is processed after its dependencies.
#
#   2. For each equation, any rhs tensor that was produced by a prior equation
#      is looked up in name_to_axes. The prior output axes are unified pairwise
#      with this equation's rhs axes via ctx.append_iter((prior_ax, eq_ax)).
#      This tells the Context that the two axis objects represent the same index.
#
#   3. ctx.apply(eq) walks the equation's full data structure via deep_reconstruct
#      and replaces every axis whose UID was unified with its canonical
#      representative. The substitution reaches inside both lhs_indices and rhs,
#      so the equation and its resulting Broadcasted are self-consistent.
#
#   4. bc_signature() is called on the substituted equation to produce the
#      Broadcasted for this step.
#
#   5. applied_eq.lhs_indices (post-apply, canonical UIDs) is stored in
#      name_to_axes so that subsequent equations unify against the same
#      canonical axes rather than stale originals.
#
# The result is Composed(content=(br1, br2, ...)) where shared axes across
# adjacent morphisms carry consistent UIDs.
@dataclass(frozen=True)
class TensorProgram(fd.Term):
    equations: fd.Prod[TensorEquation] = ()

    def to_morphism(
        self,
        declarations: dict[fd.DynamicName, tuple[sc.RawAxis, ...]] | None = None,
        array_datatypes: dict[fd.DynamicName, bc.Datatype] | None = None,
    ) -> pc.ThreadedComposed:
        ctx = fd.Context()
        morphisms = []
        name_to_axes: dict[fd.DynamicName | None, fd.Prod[sc.RawAxis]] = {}
        declarations = declarations or {}

        # Pre-collect external tensor names (in topo order of first appearance).
        # Internal names are those with a defining equation.
        internal_names: set[fd.DynamicName | None] = {eq.lhs_name for eq in self.equations}
        equations_sorted = topological_sort(self.equations)
        external_order: list[fd.DynamicName] = []
        external_name_set: set[fd.DynamicName] = set()
        for eq in equations_sorted:
            for factor in eq.rhs:
                if isinstance(factor, TensorRef) and factor.name not in internal_names:
                    if factor.name not in external_name_set:
                        external_order.append(factor.name)
                        external_name_set.add(factor.name)
        n_external = len(external_order)
        ext_idx: dict[fd.DynamicName, int] = {name: i for i, name in enumerate(external_order)}

        external_axes: dict[fd.DynamicName, tuple[sc.RawAxis, ...]] = {}
        produced_idx: dict[fd.DynamicName | None, int] = {}
        step_routing: list[tuple[int, ...]] = []

        for eq in equations_sorted:
            # Unify declaration axes with lhs axes to propagate concrete sizes.
            if eq.lhs_name in declarations:
                decl_axes = declarations[eq.lhs_name]
                if len(decl_axes) != len(eq.lhs_indices):
                    raise ValueError(
                        f"declaration rank mismatch for {eq.lhs_name!r}: "
                        f"expected {len(eq.lhs_indices)}, got {len(decl_axes)}"
                    )
                for decl_ax, eq_ax in zip(decl_axes, eq.lhs_indices):
                    ctx.append_iter((decl_ax, eq_ax))
            seen_in_eq: set[fd.DynamicName | None] = set()
            for factor in eq.rhs:
                if not isinstance(factor, TensorRef):
                    continue
                tensor_name = factor.name
                input_axes = factor.axes
                if tensor_name not in internal_names:
                    # External tensor: unify axes with prior occurrence if any.
                    if tensor_name in external_axes:
                        if tensor_name not in seen_in_eq:
                            seen_in_eq.add(tensor_name)
                            for prior_ax, eq_ax in zip(external_axes[tensor_name], input_axes):
                                ctx.append_iter((prior_ax, eq_ax))
                    else:
                        external_axes[tensor_name] = input_axes
                elif tensor_name in name_to_axes and tensor_name not in seen_in_eq:
                    # Internal tensor: unify with the prior equation's output axes.
                    seen_in_eq.add(tensor_name)
                    prior_axes = name_to_axes[tensor_name]
                    if len(prior_axes) != len(input_axes):
                        raise ValueError(
                            f"axis rank mismatch for intermediate {tensor_name!r}: "
                            f"defined with rank {len(prior_axes)}, consumed with rank {len(input_axes)}"
                        )
                    for prior_ax, eq_ax in zip(prior_axes, input_axes):
                        ctx.append_iter((prior_ax, eq_ax))
                # Subsequent occurrences of the same intermediate tensor are
                # self-joins: skip unification so each reference keeps its own
                # axis UIDs.
            applied_eq = ctx.apply(eq)

            # Compute routing for this step from the equation's domain factors.
            domain_factors = [
                f for f in applied_eq.rhs
                if not (isinstance(f, (IversonBinOp, IversonUnaryOp))
                        and _iverson_is_materializable(f))
            ]
            route: list[int] = []
            for factor in domain_factors:
                if isinstance(factor, TensorRef):
                    name = factor.name
                    if name in ext_idx:
                        route.append(ext_idx[name])
                    else:
                        route.append(n_external + produced_idx[name])
                # Unsized Iverson (Bool-typed) — not yet in external_order;
                # skip for now (no caller slot assigned).
            step_routing.append(tuple(route))

            # Each equation is one step (einsum + optional inner nonlinearity).
            result = _split_nonlinearity(applied_eq, array_datatypes=array_datatypes)
            morphisms.append(result)
            name_to_axes[eq.lhs_name] = applied_eq.lhs_indices
            produced_idx[eq.lhs_name] = len(produced_idx)

        return pc.ThreadedComposed(
            content=tuple(morphisms),
            routing=tuple(step_routing),
            n_external=n_external,
        )


def topological_sort(
    equations: fd.Prod[TensorEquation],
) -> list[TensorEquation]:
    # Kahn's algorithm. Only intra-program names (those with a defining equation)
    # count as dependencies; external tensors (weights, inputs) are ignored.
    name_to_eq: dict[fd.DynamicName | None, TensorEquation] = {}
    for eq in equations:
        if eq.lhs_name in name_to_eq:
            raise ValueError(f"duplicate lhs_name {eq.lhs_name!r} in TensorProgram")
        name_to_eq[eq.lhs_name] = eq
    deps: dict[fd.DynamicName | None, set[fd.DynamicName | None]] = {
        eq.lhs_name: {
            factor.name for factor in eq.rhs
            if isinstance(factor, TensorRef) and factor.name in name_to_eq
        }
        for eq in equations
    }
    result: list[TensorEquation] = []
    ready: deque[TensorEquation] = deque(
        eq for eq in equations if not deps[eq.lhs_name]
    )
    processed_names: set[fd.DynamicName | None] = set()
    ready_names: set[fd.DynamicName | None] = {eq.lhs_name for eq in ready}
    while ready:
        eq = ready.popleft()
        result.append(eq)
        processed_names.add(eq.lhs_name)
        ready_names.discard(eq.lhs_name)
        for other in equations:
            if other.lhs_name not in processed_names and other.lhs_name not in ready_names:
                deps[other.lhs_name].discard(eq.lhs_name)
                if not deps[other.lhs_name]:
                    ready.append(other)
                    ready_names.add(other.lhs_name)
    # A short result means at least one cycle exists — raise rather than silently
    # producing an incomplete Composed whose dom()/cod() would raise IndexError.
    if len(result) < len(equations):
        cyclic = [eq.lhs_name for eq in equations if eq.lhs_name not in processed_names]
        raise ValueError(f"TensorProgram has cyclic equation dependencies: {cyclic!r}")
    return result
