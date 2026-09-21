'''The operator overloads an expression is written with.

Importing the package installs `@`, `*` and `>>` on the categories defined in
`data_structure`. The methods on `ProductCategory.Morphism` raise
`NotImplementedError` until it is imported, so a module that builds an expression
imports it for that effect alone, as `import construction_helpers as ch`.

  composition     `@`, sequential composition. The left morphism's codomain and
                  the right morphism's domain are separately constructed axes
                  with different UIDs, so `align_axes` identifies them by
                  position and `add_excess_lift` lifts the shorter side.
  product         `*`, the parallel product. Two objects give an object, a
                  morphism on either side gives a morphism, and a nested tuple
                  of either flattens.
  lift            `>>`, broadcasting an array or a morphism over a product of
                  axes.
  einops          The `'q d, x d -> q x'` signature strings, read into weaves,
                  reindexings and contraction groups.
  signature       The shorter signature the generic operators take, in which a
                  symbol on the right of the arrow is degree and a symbol on the
                  left alone is absorbed.
  simple_helper   `make_composed` and `make_product`, which flatten a nested
                  `Composed` or `ProductOfMorphisms` and drop identities.
  factorize       No implementation.

`obsidian/02-categories/Construction Helpers.md` is the full account.
'''
from construction_helpers.composition import composition as comp
from construction_helpers.lift import dynamic_object_lift
from construction_helpers.product import object_product, morphism_product, general_product

__all__ = [
    'comp',
    'dynamic_object_lift',
    'object_product',
    'morphism_product',
    'general_product',
]
