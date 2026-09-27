'''The three classic neural circuit diagrams of https://zardini.mit.edu/diagrams/, each
written as a morphism in Br and drawn from the expression.

Written by Claude Opus 5.5 (1M context), reasoning effort 40.

The page holds three diagrams drawn by hand: the encoder-decoder transformer of
*Attention Is All You Need*, Mixtral-8x7B and DeepSeek-V3. A module here builds the
expression of each. A module states the mechanisms of one model, and a wording file beside it holds the titles and the descriptions of its
blocks, which carry the annotations of the hand-drawn diagram into the inspection boxes
of the interactive page. `reference_links.py` writes the pinned links into the papers
and the released code each model is checked against, and `shared_mechanisms.py` holds
the mechanisms two of the models share.

The notebooks under `notebooks/website/classic/` draw the models for the lab website,
and a validator beside each notebook checks every claim of the notebook. The modules
that build their further forms sit here, each named after the form it builds. A
`quantised_` module writes the quantisations of the released code onto a model, a
`cached_` module derives the pass of the model over the new tokens, and a
`_page_variants` module lists the forms the page of the model switches between.
DeepSeek-V3 is drawn there at the sizes of its released checkpoint, which
`released_deepseek_v3.py` builds from the parts of `deepseek_v3.py`, with the sentences
that name a size in `released_deepseek_v3_wording.json`.
'''
