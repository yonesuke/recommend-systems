import dataclasses
import polars as pl

@dataclasses.dataclass(frozen=True)
class Dataset:
    """Dataset for recommendation system
    
    Args:
    train (pl.DataFrame): Training data
    test (pl.DataFrame): Test data
    test_user2items (dict[int, list[int]]): Test data for each user
    item_content (pl.DataFrame): Item content data
    """
    train: pl.DataFrame
    test: pl.DataFrame
    test_user2items: dict[int, list[int]]
    item_content: pl.DataFrame
    
@dataclasses.dataclass(frozen=True)
class RecommendResult:
    """Recommend result
    
    Args:
    rating (pl.DataFrame): Rating data
    user2items (dict[int, list[int]]): Recommended items for each user
    """
    rating: pl.DataFrame
    user2items: dict[int, list[int]]
    
@dataclasses.dataclass(frozen=True)
class Metrics:
    """Metrics for recommendation system
    
    Args:
    rsme (float): RSME
    precision_at_k (float): Precision@K
    recall_at_k (float): Recall@K
    """
    rsme: float
    precision_at_k: float
    recall_at_k: float
    
    def __repr__(self) -> str:
        return f'RSME: {self.rsme:.4f}, Precision@K: {self.precision_at_k:.4f}, Recall@K: {self.recall_at_k:.4f}'