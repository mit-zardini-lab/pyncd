import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm

### SOME NUMERIC ALGEBRA
import itertools
import utilities.utilities as util

def expand_to_addition(expr: nm.Numeric) -> fd.Prod[nm.Numeric]:
    match expr:
        case nm.Addition(content=terms):
            return util.concat(expand_to_addition(term) for term in terms)
        case nm.Multiplication(content=terms):
            subterms = [expand_to_addition(term) for term in terms]
            return tuple(
                nm.Multiplication.template(*combination)
                for combination in itertools.product(*subterms)
            )
        case _:
            return (expr,)

def expand(expr: nm.Numeric) -> nm.Numeric:
    return nm.Addition.template(*expand_to_addition(expr))

# def replace(
#     old: Numeric,
#     new: Numeric,
#     target: Numeric
# ):
#     if target == old:
#         return new
#     if isinstance(target, Associative) and isinstance(new, Associative) and type(target) == type(new):
#         if 
#     target = expand(target)
#     target = target.reconstruct(
#         **{k: replace(old, new, v)
#             for k, v in target.dict().items()}
#     )

        