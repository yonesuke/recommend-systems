import dataclasses
import pandas as pd

@dataclasses.dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    test_user2items: dict[int, list[int]]
    item_content: pd.DataFrame
    
@dataclasses.dataclass(frozen=True)
class RecommendResult:
    rating: pd.DataFrame
    user2items: dict[int, list[int]]
    
@dataclasses.dataclass(frozen=True)
class Metrics:
    rsme: float
    precision_at_k: float
    recall_at_k: float
    
    def __repr__(self) -> str:
        return f'RSME: {self.rsme:.4f}, Precision@K: {self.precision_at_k:.4f}, Recall@K: {self.recall_at_k:.4f}'