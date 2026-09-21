'''
Figures for a notebook that demonstrates attention.

Two kinds of picture accompany such a notebook, and they answer different
questions.

  * The **circuit diagram** - what the expression *is*. Drawn by tsncd in a
    browser, reached here through `show_diagram`, which wraps the
    capture/send/skip choice so the notebook does not repeat it at every call.
  * The **numeric figures** - what the expression *does* to actual arrays.
    Plain matplotlib, no browser, so they survive `nbconvert --execute` on a
    machine with nothing else installed.

The second kind is the reason this module exists. Attention is a soft lookup,
and that is a claim about numbers: the attention matrix is where each query
looks, its rows are distributions, and the output is a mixture of value rows.
Each of those is one figure.

Colour follows one rule, applied everywhere:

    signed quantity      -> diverging, blue (negative) to red (positive)
                            through a neutral grey, symmetric about zero
    probability          -> sequential, one hue, light to dark
    a categorical series -> a fixed slot, never a value ramp

so a blue-to-red picture always means "this has a sign" and a monotone blue one
always means "this is a magnitude in [0, 1]". Cells are labelled only where the
number is the point, as in the attention matrix. A raw input is left
since their individual entries carry nothing.
'''

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


#############
## PALETTE ##
#############

@dataclass(frozen=True)
class Surface:
    '''The mode-dependent half of the palette: everything that is not data.'''
    background: str
    primary: str
    secondary: str
    muted: str
    grid: str
    axis: str
    neutral: str      # the diverging midpoint - "nothing here"


LIGHT = Surface(
    background='#fcfcfb', primary='#0b0b0b', secondary='#52514e',
    muted='#898781', grid='#e1e0d9', axis='#c3c2b7', neutral='#f0efec')

DARK = Surface(
    background='#1a1a19', primary='#ffffff', secondary='#c3c2b7',
    muted='#898781', grid='#2c2c2a', axis='#383835', neutral='#383835')

SURFACES = {'light': LIGHT, 'dark': DARK}

# Categorical slots, in fixed order. They are never cycled and never
# assigned by rank, because the
# first three are the only ones any figure here needs, and they are the three
# that separate under colour-vision deficiency with every pair on screen.
SERIES = {
    'light': ('#2a78d6', '#eb6834', '#1baf7a'),
    'dark': ('#3987e5', '#d95926', '#199e70'),
}

# One hue, light to dark. Magnitude in [0, 1].
_BLUE_RAMP = (
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
    '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
    '#0d366b',
)

# The warm arm of the diverging pair, stepped to mirror the cool one.
_RED_RAMP = (
    '#fbdad9', '#f7c4c3', '#f3adac', '#ef9695', '#eb7f7e', '#e76463',
    '#e34948', '#d43c3b', '#c23230', '#ad2a28', '#961f1e', '#7d1817',
    '#661212',
)

_current: Surface = LIGHT
_current_mode: str = 'light'


def surface() -> Surface:
    '''The surface the last `use_style` selected.'''
    return _current


def series(index: int) -> str:
    '''Categorical slot `index` (0-based) for the current mode.'''
    return SERIES[_current_mode][index]


def sequential() -> LinearSegmentedColormap:
    '''Probabilities: one hue, light to dark, lightest meaning near zero.'''
    return LinearSegmentedColormap.from_list('pyncd_sequential', _BLUE_RAMP)


def diverging() -> LinearSegmentedColormap:
    '''Signed values: blue and red poles about a neutral grey midpoint.

    The midpoint is the surface's neutral rather than a hue, so zero reads as
    "nothing" instead of as a third category.
    '''
    return LinearSegmentedColormap.from_list(
        'pyncd_diverging',
        (*reversed(_BLUE_RAMP), _current.neutral, *_RED_RAMP))


def use_style(mode: str = 'light') -> Surface:
    '''Select a mode and install it into matplotlib's defaults.

    Dark is a selection rather than an inversion, with its own ramps against
    surface. Call once at the top of a notebook.
    '''
    global _current, _current_mode
    _current, _current_mode = SURFACES[mode], mode
    s = _current
    mpl.rcParams.update({
        'figure.facecolor': s.background,
        'figure.dpi': 130,
        'savefig.facecolor': s.background,
        'savefig.bbox': 'tight',
        'axes.facecolor': s.background,
        'axes.edgecolor': s.axis,
        'axes.labelcolor': s.secondary,
        'axes.titlecolor': s.primary,
        'axes.titlesize': 9,
        'axes.titleweight': 'normal',
        'axes.titlelocation': 'left',
        'axes.titlepad': 8,
        'axes.labelsize': 8,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.color': s.grid,
        'grid.linewidth': 0.6,
        'grid.linestyle': '-',           # never dashed: a grid is not a threshold
        'xtick.color': s.muted,
        'ytick.color': s.muted,
        'xtick.labelcolor': s.secondary,
        'ytick.labelcolor': s.secondary,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'legend.frameon': False,
        'legend.fontsize': 8,
        'lines.linewidth': 2.0,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8,
        'text.color': s.primary,
    })
    return s


use_style('light')


##############
## PRIMITIVE ##
##############

def _symmetric(data: np.ndarray) -> Normalize:
    '''Limits centred on zero, so the neutral midpoint really is zero.'''
    extent = float(np.abs(data).max()) or 1.0
    return Normalize(vmin=-extent, vmax=extent)


def _ink(image, value: float) -> str:
    '''Whichever of the two inks reads on the cell this value lands on.

    Measured off the colour the ramp produced rather than off the value:
    "large" means a dark cell on a sequential ramp but a light one at the
    midpoint of a diverging ramp, and guessing from the value gets the
    near-zero cells of a sequential heatmap wrong in exactly the way that
    makes them unreadable.
    '''
    red, green, blue = image.cmap(image.norm(value))[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return _current.background if luminance < 0.55 else _current.primary


def _clean(text: str) -> str:
    '''`-0.0` is a rounding artefact rather than a number a caller means.'''
    return text[1:] if text.startswith('-') and float(text) == 0 else text


def heatmap(
    ax,
    data: np.ndarray,
    *,
    signed: bool = True,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    row_title: str = '',
    col_title: str = '',
    title: str = '',
    annotate: bool = False,
    fmt: str = '{:.2f}',
    vmax: float | None = None,
):
    '''One matrix as a grid of cells, rows down and columns across.

    `signed` picks the encoding: diverging about zero, or sequential from zero
    up to `vmax`. `annotate` writes the value into each cell, which is the
    table view for a matrix small enough to have one, so it is opt-in, and
    only worth it where the numbers are what the figure is about.
    '''
    s = _current
    if signed:
        image = ax.imshow(data, cmap=diverging(), norm=_symmetric(data),
                          aspect='auto', interpolation='nearest')
    else:
        image = ax.imshow(data, cmap=sequential(), vmin=0.0,
                          vmax=vmax if vmax is not None else float(data.max()),
                          aspect='auto', interpolation='nearest')

    ax.set_xticks(range(data.shape[1]), col_labels or range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]), row_labels or range(data.shape[0]))
    if col_title:
        ax.set_xlabel(col_title)
    if row_title:
        ax.set_ylabel(row_title)
    if title:
        ax.set_title(title)

    # A 2px surface gap between cells rather than a border around each one.
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which='minor', color=s.background, linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, _clean(fmt.format(data[i, j])),
                        ha='center', va='center', fontsize=6.5,
                        color=_ink(image, float(data[i, j])))
    return image


def _colorbar(fig, image, ax, label: str, orientation: str = 'vertical'):
    horizontal = orientation == 'horizontal'
    bar = fig.colorbar(image, ax=ax, orientation=orientation,
                       fraction=0.05 if horizontal else 0.045,
                       pad=0.16 if horizontal else 0.03,
                       aspect=40 if horizontal else 20)
    bar.set_label(label, fontsize=7.5, color=_current.secondary)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=_current.muted, length=0)
    labels = (bar.ax.get_xticklabels() if horizontal
              else bar.ax.get_yticklabels())
    for text in labels:
        text.set_color(_current.secondary)
    return bar


#############
## FIGURES ##
#############

def plot_inputs(Q, K, V, *, query_labels, key_labels, figsize=(8.4, 2.6)):
    '''The three inputs, side by side, so the shared axes are visible.

    Q and K are the same width, and that width is the axis the first contraction
    consumes. K and V are the same height, and that height is the axis the softmax
    normalises over and the second contraction consumes. V's width is shared
    with nothing, which is why it ends up as the width of the output.
    '''
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    fig, axes = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={'width_ratios': [Q.shape[1], K.shape[1], V.shape[1]]})

    heatmap(axes[0], Q, row_labels=query_labels, row_title='q  queries',
            col_title='d', title=f'Q : R[q, d]   {Q.shape}')
    heatmap(axes[1], K, row_labels=key_labels, row_title='x  keys',
            col_title='d', title=f'K : R[x, d]   {K.shape}')
    image = heatmap(axes[2], V, row_labels=key_labels, row_title='x  keys',
                    col_title='e', title=f'V : R[x, e]   {V.shape}')
    _colorbar(fig, image, axes[2], 'value')
    fig.set_facecolor(_current.background)
    fig.tight_layout()
    return fig


def plot_attention(
    scores, weights, *, query_labels, key_labels, figsize=(7.6, 3.2)
):
    '''Scores, then the same matrix after the softmax.

    Left is signed and unnormalised, and right is a probability per row. The row
    sums printed down the right-hand edge are the check that the softmax ran
    along `x` and not along `q` - every row is 1, no column has to be.
    '''
    scores, weights = np.asarray(scores), np.asarray(weights)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    raw = heatmap(axes[0], scores, row_labels=query_labels,
                  col_labels=key_labels, row_title='q  queries',
                  col_title='x  keys', title='QKᵀ   scores',
                  annotate=True, fmt='{:.1f}')
    _colorbar(fig, raw, axes[0], 'score', orientation='horizontal')

    soft = heatmap(axes[1], weights, signed=False, row_labels=query_labels,
                   col_labels=key_labels, col_title='x  keys',
                   title='softmax over x   attention weights',
                   annotate=True, vmax=1.0)
    _colorbar(fig, soft, axes[1], 'weight', orientation='horizontal')

    # The row sums live on the right edge, which the horizontal colour bar
    # leaves free. They are the check that the softmax ran along `x`: every
    # row is 1, and no column has to be.
    for i, total in enumerate(weights.sum(axis=1)):
        axes[1].text(weights.shape[1] - 0.3, i, f'Σ {total:.2f}',
                     ha='left', va='center', fontsize=7, clip_on=False,
                     color=_current.secondary)
    fig.set_facecolor(_current.background)
    fig.tight_layout()
    return fig


def plot_lookup(
    weights, V, output, query, *, query_labels, key_labels, figsize=(8.0, 2.9)
):
    '''One query, end to end: where it looks, and what that returns.

    Left, the row of the attention matrix as bars, a distribution over key
    positions. Right, the output row beside every value row, each drawn at the
    opacity its weight earned, which is the picture of a convex combination:
    the output cannot leave the hull the value rows span.
    '''
    weights, V, output = np.asarray(weights), np.asarray(V), np.asarray(output)
    row, s = weights[query], _current
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [1.15, 1]})

    axes[0].bar(range(len(row)), row, color=series(0), width=0.62)
    axes[0].set_xticks(range(len(row)), key_labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('x  keys')
    axes[0].set_ylabel('weight')
    axes[0].set_title(f'where {query_labels[query]} looks')
    axes[0].grid(axis='y')
    axes[0].set_axisbelow(True)
    axes[0].spines['left'].set_visible(False)
    for j, weight in enumerate(row):
        if weight >= 0.05:                       # label the ones a reader needs
            axes[0].text(j, weight + 0.03, f'{weight:.2f}', ha='center',
                         fontsize=7, color=s.secondary)

    positions = np.arange(V.shape[1])
    for j, value_row in enumerate(V):
        axes[1].plot(positions, value_row, color=s.muted, marker='o',
                     markersize=3.5, linewidth=1.2,
                     alpha=0.15 + 0.85 * float(row[j]), zorder=2,
                     label='value rows' if j == 0 else None)
        if row[j] >= 0.1:          # only the rows that actually contribute
            axes[1].annotate(key_labels[j], (positions[-1], value_row[-1]),
                             textcoords='offset points', xytext=(7, 0),
                             ha='left', va='center', fontsize=7,
                             color=s.secondary)
    axes[1].plot(positions, output, color=series(1), marker='o', markersize=6,
                 markeredgecolor=s.background, markeredgewidth=1.5,
                 linewidth=2.4, zorder=3, label='output')
    axes[1].set_xticks(positions, [f'e{j}' for j in positions])
    axes[1].set_xlim(positions[0] - 0.15, positions[-1] + 0.7)
    axes[1].set_xlabel('e')
    axes[1].set_title(f'output row {query_labels[query]}  =  Σₓ weight · V[x]')
    axes[1].grid(axis='y')
    axes[1].set_axisbelow(True)
    axes[1].legend(loc='lower left')
    fig.set_facecolor(_current.background)
    fig.tight_layout()
    return fig


def plot_scaling(dims, unscaled, scaled, *, figsize=(5.6, 3.0)):
    '''Why the scores get divided by the square root of their length.

    `unscaled` and `scaled` are the mean largest softmax weight at each
    dimension. A dot product of `d` independent terms has a standard deviation
    that grows like the square root of `d`, so without the correction the
    softmax sharpens as the head gets wider, eventually onto one key, where
    its gradient vanishes. Dividing restores a `d`-independent spread.
    '''
    dims = np.asarray(dims)
    fig, ax = plt.subplots(figsize=figsize)

    for values, colour, label in (
        (np.asarray(unscaled), series(1), 'QKᵀ'),
        (np.asarray(scaled), series(0), 'QKᵀ / √d'),
    ):
        ax.plot(dims, values, color=colour, marker='o', markersize=4,
                markeredgecolor=_current.background, markeredgewidth=1.2,
                label=label)
        ax.annotate(label, (dims[-1], values[-1]), textcoords='offset points',
                    xytext=(6, 0), va='center', fontsize=8, color=colour)

    ax.set_xscale('log', base=2)
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_ylim(-0.06, 1.02)
    ax.set_xlabel('d   key/query width')
    ax.set_ylabel('mean largest attention weight')
    ax.set_title('An unscaled softmax sharpens as d grows')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='lower center', ncol=2)
    ax.set_xlim(dims[0] * 0.9, dims[-1] * 1.45)
    fig.set_facecolor(_current.background)
    fig.tight_layout()
    return fig


def plot_permutation(
    weights, shuffled, difference, order, *,
    query_labels, key_labels, figsize=(8.6, 3.0)
):
    '''Shuffling the keys and values permutes the columns and nothing else.

    Third panel is the output difference, on the same symmetric scale the score
    matrix used, so a panel that reads as flat neutral states that attention
    has no idea what order its keys arrived in.
    '''
    weights, shuffled = np.asarray(weights), np.asarray(shuffled)
    difference = np.asarray(difference)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    shuffled_labels = [key_labels[i] for i in order]

    heatmap(axes[0], weights, signed=False, row_labels=query_labels,
            col_labels=key_labels, row_title='q  queries', col_title='x  keys',
            title='weights', vmax=1.0)
    heatmap(axes[1], shuffled, signed=False, row_labels=query_labels,
            col_labels=shuffled_labels, col_title='x  keys, shuffled',
            title='weights, keys and values shuffled', vmax=1.0)
    image = heatmap(axes[2], difference, row_labels=query_labels,
                    col_title='e', title='output difference')
    image.set_norm(Normalize(vmin=-1, vmax=1))
    _colorbar(fig, image, axes[2], 'difference')
    axes[2].text(
        0.5, -0.34,
        f'largest difference {np.abs(difference).max():.2e}',
        transform=axes[2].transAxes, ha='center', fontsize=7.5,
        color=_current.secondary)
    fig.set_facecolor(_current.background)
    fig.tight_layout()
    return fig


##############
## DIAGRAMS ##
##############

# Diagrams go through `notebook_diagrams`, which holds the one mechanism every
# notebook shares. A notebook declares its settings at the top of its setup
# cell and passes them to every call:
#
#     import notebooks.display.attention_figures as figs
#     DIAGRAMS = figs.DiagramSettings(mode=figs.DiagramMode.OFF)
#     await figs.show_diagram(attention, 'Basic attention.', settings=DIAGRAMS)
#
# `notebook_diagrams` states how a notebook is executed from the command line.
from notebooks.display.notebook_diagrams import (
    DiagramMode as DiagramMode,
    DiagramSettings as DiagramSettings,
    SETTINGS as SETTINGS,
    show_diagram as show_diagram,
)
