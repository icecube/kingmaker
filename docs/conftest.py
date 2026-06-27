"""
Run every ``.. code-block:: python`` example in the docs as a pytest test.

Each .rst file is one Sybil "document": its code blocks execute in order,
sharing one namespace, so later blocks can rely on names defined by earlier
ones (matching how a reader would actually follow the page top to bottom).
"""

from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser


def _use_headless_matplotlib(namespace):
    import matplotlib

    matplotlib.use("Agg")


pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser()],
    patterns=["*.rst"],
    setup=_use_headless_matplotlib,
).pytest()
