from abc import ABC, abstractmethod
from utils.data_loader import DataLoader
from utils.metric_calculator import MetricCalculator
from utils.models import Dataset, RecommendResult

class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass
    
    def run_sample(self) -> None:
        dataset = DataLoader(n_user=1000, n_test_items=5, data_path='../data/ml-10M100K').load()
        recommend_result = self.recommend(dataset)
        metrics = MetricCalculator().calc(
            dataset.test.get_column('rating').to_list(),
            recommend_result.rating.get_column('pred_rating').to_list(),
            dataset.test_user2items,
            recommend_result.user2items,
            k=10
        )
        print(metrics)