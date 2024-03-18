import numpy as np
from sklearn.metrics import mean_squared_error
from utils.models import Metrics

class MetricCalculator:
    def calc(
        self,
        true_rating: list[float],
        pred_rating: list[float],
        true_user2items: dict[int, list[int]],
        pred_user2items: dict[int, list[int]],
        k: int
    ) -> Metrics:
        rsme = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)
        return Metrics(rsme, precision_at_k, recall_at_k)
    
    def _precision_at_k(self, true_item: list[int], pred_item: list[int], k: int) -> float:
        if k == 0:
            return 0.0
        
        return len(set(true_item) & set(pred_item[:k])) / k
    
    def _recall_at_k(self, true_item: list[int], pred_item: list[int], k: int) -> float:
        if k == 0:
            return 0.0
        
        return len(set(true_item) & set(pred_item[:k])) / len(true_item)
    
    def _calc_rmse(self, true_rating: list[float], pred_rating: list[float]) -> float:
        return np.sqrt(mean_squared_error(true_rating, pred_rating))
    
    def _calc_precision_at_k(
        self, true_user2items: dict[int, list[int]], pred_user2items: dict[int, list[int]], k: int
    ) -> float:
        scores = []
        for user_id in true_user2items.keys():
            true_item = true_user2items[user_id]
            pred_item = pred_user2items[user_id]
            scores.append(self._precision_at_k(true_item, pred_item, k))
        return np.mean(scores)
    
    def _calc_recall_at_k(
        self, true_user2items: dict[int, list[int]], pred_user2items: dict[int, list[int]], k: int
    ) -> float:
        scores = []
        for user_id in true_user2items.keys():
            true_item = true_user2items[user_id]
            pred_item = pred_user2items[user_id]
            scores.append(self._recall_at_k(true_item, pred_item, k))
        return np.mean(scores)