# Claude Opus 5.5 (1M context), effort 40.
'''The selective scan of one Mamba layer as a loop over the tokens, and the evaluation of
its operators in `torch.float64`. The delta-rule scan of `notebooks/sota/KimiK3/` is
written and evaluated in the same way.

The layer is the mixer `Mamba` of `state-spaces/mamba`.

    reference_links         the pinned links into `mamba_simple.py` and
                            `selective_scan_interface.py`
    declared_axes           the axes, the arrays between the steps, the counter of the
                            loop over the tokens, and the sizes the layer is evaluated at
    selective_scan          the scan as a loop over the tokens, with the state and the
                            output as loop variables on the tape
    evaluate_numerically    the operators of the layer and of its pass evaluated in
                            `torch.float64`, with caches kept between passes

`obsidian/08-caching/Carrying the State of a Scan Between Passes.md` states the
mathematics.
'''
