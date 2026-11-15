"""The composer: an LLM that writes wsp programs from mined components.

The LLM reads the atlas and the catalog, writes programs in the wsp
language, compiles them on Modal, reads practice scores, and revises. It
has no tool that reads or writes raw weight values. See docs/composer.md.
"""
