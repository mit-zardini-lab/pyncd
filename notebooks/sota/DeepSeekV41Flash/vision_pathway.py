# Claude Opus 5, effort high.
'''The image pathway of the released DeepSeek-V4.1-Flash, which stands between the
embedding and the expansion into the four residual streams.

Written by Claude Fable 5.1, reasoning effort 80, and moved into this package by Claude
Opus 5, effort high, on 2026-09-19, when the user asked for every mechanism to be stated
in its newest form and the write at token positions was split into
`write_at_token_positions`.

The patches of one image pass through the vision encoder, the three-by-three pixel
unshuffle and the projector, and the feature of each cell is written into the residual
at the token position the cell occupies. `image_pathway` returns those four steps as
one block. `write_span_delimiters` is the write of the three learned delimiter vectors
of the image span, and `image_pathway_with_delimiters` composes the two.

    encode_patches   the vision encoder, the one `ops.GenericOperator` of the module
    unshuffle_pixels the patch grid read as a grid of cells of three by three patches
    project_cells    the unshuffle and the two biased linear maps of the projector
    write_cell_features
                     the feature of every cell written at the token the cell occupies
    delimiter_vectors, write_span_delimiters
                     the three learned delimiter vectors and their write

The view hands a cell out in the order (U, V, M), so the pixel unshuffle is the product
of one group view per grid axis and the identity on the features. The released
`F.unfold` lays a cell out in the order (M, U, V), so the rows of the first weight of
the projector are in a different order from the released weight. Both linear maps of the
projector carry a bias, as `nn.Linear` does unless told otherwise.

`ViT` is the one generic operator. The vision encoder is a second model of 32 layers.
Its rotary embedding takes two positions, the row and the column of a patch, and pairs
channel `i` with channel `i + 32`. `dst.Rotary` takes one position and pairs adjacent
channels, so it does not state that rotation.

The expression states one image whose patch grid is a whole number of cells. The
released code pads the grid with zeros to a multiple of three, and repeats the pathway
for every image of the prompt.

`OPERATOR_ROLES` says what each weight of this module is for, `ARITHMETIC_ROLES` and
`ARITHMETIC_REFERENCES` what the GELU is, and `GENERIC_OPERATOR_EXPLANATIONS` what the
vision encoder is, in the form `operator_explanations` holds its own. The explanation
tables of the omissions notebook and of the integrated model both take those rows from
here.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.display.explain_operators import OperatorExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, m
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (
    CELL_FEATURES, IMAGE_POSITIONS, PATCHES, PATCH_FEATURES, STATE, H, Hp, M, U, V, W,
    Wp, generic_operator)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    image_processor_lines, inference_config_lines, model_lines, vision_lines)
from notebooks.sota.DeepSeekV41Flash.write_at_token_positions import write_at_positions
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

VISION_ENCODER_NAME = '\\mathrm{ViT}'
GENERIC_OPERATOR_NAMES: tuple[str, ...] = (VISION_ENCODER_NAME,)
CELL_VIEW_NAME = '\\mathrm{cell}'
GROUP_VIEW_NAME = 'grp'
GELU_NAME = '\\mathrm{GELU}'
FIRST_PROJECTOR_WEIGHT = 'W^{A1}'
SECOND_PROJECTOR_WEIGHT = 'W^{A2}'
DELIMITER_VECTORS_NAME = '\\mathrm{delim}'

PATHWAY_COLOUR = '#F1F4C1'
ENCODER_COLOUR = '#C5BEDF'
PROJECTOR_COLOUR = '#D9E7F5'
CELL_WRITE_COLOUR = '#B8D8CE'
DELIMITER_WRITE_COLOUR = '#EFEFF4'

UNIT_STRIDE = nm.Integer(1)
NO_SHIFT = nm.Integer(0)
CELL_ROWS = sc.StrideMorphism(
    _dom=(Hp, U),
    _cod_stride_shift=((H, (U.local_size(), UNIT_STRIDE), NO_SHIFT),),
    name=fd.DynamicName(GROUP_VIEW_NAME))
CELL_COLUMNS = sc.StrideMorphism(
    _dom=(Wp, V),
    _cod_stride_shift=((W, (V.local_size(), UNIT_STRIDE), NO_SHIFT),),
    name=fd.DynamicName(GROUP_VIEW_NAME))
CELL_INDICES_BEFORE_THE_OFFSETS = route((0, 2, 1, 3, 4), (Hp, Wp, U, V, M))
CELLS_OF_THE_PATCH_GRID = (
    CELL_INDICES_BEFORE_THE_OFFSETS
    @ (CELL_ROWS * CELL_COLUMNS * cat.ProdObject((M,)).identity()))

DELIMITERS = fd.DynamicName('\\Delta', code_form='delimiters').capture(
    cat.RawAxis(_size=Hp.local_size() + nm.Integer(2)))
DELIMITER_KIND = cat.Natural.template('\\delta')
DELIMITER_POSITIONS = cat.Array(IMAGE_POSITIONS.datatype, (DELIMITERS,))
DELIMITER_KINDS = cat.Array(DELIMITER_KIND, (DELIMITERS,))
DELIMITER_VECTORS = cat.Array(R, (DELIMITERS, m))


def encode_patches() -> cat.Block:
    '''The vision encoder as one generic operator. Its attention reads every patch of
    the image, so the whole grid is in the target and nothing is broadcast over.'''
    return cat.Block.template(
        generic_operator(VISION_ENCODER_NAME, (PATCHES,), (PATCH_FEATURES,)),
        title=text.ENCODER_TITLE, fill_color=ENCODER_COLOUR,
        description=text.ENCODE_PATCHES_DESCRIPTION,
        references=(vision_lines(87, 103), vision_lines(8, 21), vision_lines(46, 60),
                    vision_lines(63, 71), inference_config_lines(53, 58),
                    inference_config_lines(65), image_processor_lines(125, 128)))


def unshuffle_pixels() -> cat.Broadcasted:
    '''The patch grid read as a grid of cells of three by three patches. The patch row
    is cut into a cell row and an offset, the patch column into a cell column and an
    offset, and the features are read unchanged, so the reindexing is the product of
    two group views and one identity. The product hands a cell out as
    `(H', U, W', V, M)`, and the rearrangement before it brings the two cell indices to
    the front, which leaves the target `(U, V, M)` that `W^{A1}` reads.'''
    return ops.View.template(reindexing=(CELLS_OF_THE_PATCH_GRID,), name=CELL_VIEW_NAME)


def gelu() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.x * nm.CumulativeGaussian(nm.x), name=GELU_NAME)


def project_cells() -> cat.Block:
    '''From the feature of every patch to one vector of the residual width for every
    cell. Both linear maps carry a bias, as `nn.Linear` does unless told otherwise.'''
    return cat.Block.template(
        unshuffle_pixels()
        @ over((Hp, Wp),
               ops.Linear.template((U, V, M), (m,), FIRST_PROJECTOR_WEIGHT, bias=True)
               @ gelu()
               @ ops.Linear.template((m,), (m,), SECOND_PROJECTOR_WEIGHT, bias=True)),
        title=text.PROJECTOR_TITLE, fill_color=PROJECTOR_COLOUR,
        formula=('\\mathrm{cell}(f)[i_{H\'}, i_{W\'}, i_{U}, i_{V}, i_{M}] = '
                 'f[\\lvert U \\rvert i_{H\'} + i_{U}, '
                 '\\lvert V \\rvert i_{W\'} + i_{V}, i_{M}]'),
        description=text.PROJECT_CELLS_DESCRIPTION,
        references=(vision_lines(106, 119), inference_config_lines(57, 58)))


def write_cell_features() -> cat.Block:
    '''From the residual, the token position of every cell and the feature of every
    cell to the residual with each cell's feature at the cell's token.'''
    return cat.Block.template(
        write_at_positions(IMAGE_POSITIONS, CELL_FEATURES),
        title=text.CELL_WRITE_TITLE, fill_color=CELL_WRITE_COLOUR,
        formula=(
            'h\'[i_{x}, i_{m}] = h[i_{x}, i_{m}]\\, \\big(1 - '
            '\\mathrm{Inject}(\\mathrm{pos}, 1)[i_{x}]\\big) + '
            '\\mathrm{Inject}(\\mathrm{pos}, g)[i_{x}, i_{m}]'),
        description=text.WRITE_CELL_FEATURES_DESCRIPTION,
        references=(model_lines(1238, 1239), model_lines(1253, 1256),
                    image_processor_lines(1, 10)))


def delimiter_vectors() -> cat.BroadcastedCategory:
    '''The learned vector of every delimiter of the span, chosen by the kind of the
    delimiter. A `Linear` whose input is an index selects one of its weights, and the
    weight holds the three vectors. The number of kinds is the named symbol `\\delta`,
    which the released code sets to three, because
    `para.processing.show_grabbed_parameters.selected_axis` names the axis of the
    weight after that symbol.'''
    return over((DELIMITERS,), ops.Linear.template(
        (), (m,), DELIMITER_VECTORS_NAME, datatype=DELIMITER_KIND, output_datatype=R))


def write_span_delimiters() -> cat.Block:
    '''From the residual, the token position of every delimiter and the kind of every
    delimiter to the residual with the learned vector of each delimiter at its token.'''
    return cat.Block.template(
        (hold(STATE) * hold(DELIMITER_POSITIONS) * delimiter_vectors())
        @ write_at_positions(DELIMITER_POSITIONS, DELIMITER_VECTORS),
        title=text.DELIMITER_WRITE_TITLE, fill_color=DELIMITER_WRITE_COLOUR,
        description=text.WRITE_SPAN_DELIMITERS_DESCRIPTION,
        references=(model_lines(1220, 1222), model_lines(1235, 1237),
                    image_processor_lines(22, 23), image_processor_lines(132, 137)))


def image_pathway() -> cat.Block:
    '''From the embedded tokens, the token position of every cell and the patches of
    one image to the embedded tokens with the image written in.'''
    return cat.Block.template(
        (hold(STATE) * hold(IMAGE_POSITIONS) * (encode_patches() @ project_cells()))
        @ write_cell_features(),
        title=text.PATHWAY_TITLE, fill_color=PATHWAY_COLOUR,
        description=text.IMAGE_PATHWAY_DESCRIPTION,
        references=(model_lines(1224, 1226), model_lines(1228, 1239),
                    model_lines(1253, 1258)))


def image_pathway_with_delimiters() -> cat.BroadcastedCategory:
    '''`image_pathway` followed by the write of the delimiter vectors, which reads two
    further operands: the token position and the kind of every delimiter.'''
    return ((image_pathway() * hold(DELIMITER_POSITIONS) * hold(DELIMITER_KINDS))
            @ write_span_delimiters())


def table_key(name: str) -> str:
    '''The text of a name, which is what the explanation tables are keyed by.'''
    return fd.DynamicName.from_str(name).to_bodies()


ENCODER_EXPLANATION = OperatorExplanation(
    title=r'\text{Vision Encoder}',
    formula=(
        r'\mathrm{ViT}(f) = \mathrm{RMSNorm}(g_{32}), \quad '
        r'g_{0}[i_{H}, i_{W}] = W^{\mathrm{pe}} f[i_{H}, i_{W}] + b^{\mathrm{pe}}, '
        r"\quad g'_{k} = g_{k} + \mathrm{Attn}_{k}\big(\mathrm{RMSNorm}(g_{k})\big), "
        r"\quad g_{k+1} = g'_{k} + W^{2}_{k}\Big( \mathrm{SiLU}\big(W^{1g}_{k}\, "
        r"\mathrm{RMSNorm}(g'_{k})\big) \odot W^{1u}_{k}\, \mathrm{RMSNorm}(g'_{k}) "
        r'\Big)'),
    description=text.VISION_ENCODER_DESCRIPTION,
    references=(
        vision_lines(87, 103), vision_lines(8, 21), vision_lines(24, 34),
        vision_lines(37, 43), vision_lines(46, 60), vision_lines(63, 84),
        image_processor_lines(125, 128), model_lines(1224, 1226),
        inference_config_lines(53, 58), inference_config_lines(65)))

GENERIC_OPERATOR_EXPLANATIONS: dict[str, OperatorExplanation] = {
    VISION_ENCODER_NAME: ENCODER_EXPLANATION,
}

ARITHMETIC_ROLES: dict[str, str] = {
    GELU_NAME: (
        text.GELU_ROLE),
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    GELU_NAME: (vision_lines(119),),
}

OPERATOR_ROLES: dict[str, OperatorRole] = {
    table_key(FIRST_PROJECTOR_WEIGHT): OperatorRole(
        role=(
            text.PROJECTOR_FIRST_MAP_ROLE),
        references=(vision_lines(110, 111), vision_lines(118, 119))),
    table_key(SECOND_PROJECTOR_WEIGHT): OperatorRole(
        role=(
            text.PROJECTOR_SECOND_MAP_ROLE),
        references=(vision_lines(112), vision_lines(119))),
    table_key(DELIMITER_VECTORS_NAME): OperatorRole(
        role=(
            text.DELIMITER_VECTORS_ROLE),
        references=(model_lines(1220, 1222), model_lines(1235, 1237))),
}
