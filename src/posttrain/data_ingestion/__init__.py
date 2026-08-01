"""Dataset ingestion and normalization.

Everything downstream (env, reward, eval) depends only on `Problem` — never on
raw HuggingFace rows. That indirection is what makes the dataset a swappable
axis (docs/README.md).
"""

from posttrain.data_ingestion.schema import Problem, TestCase

__all__ = ["Problem", "TestCase"]
