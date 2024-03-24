import sys
sys.path.append('..')

from utils.models import RecommendResult, Dataset
from algorithms.base_recommender import BaseRecommender
import polars as pl

class PopularityRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 評価値の閾値
        minimum_num_rating = kwargs.get('minimum_num_rating', 200)
        # 各映画ごとの平均の評価値を計算し、その平均評価値を予測値として利用
        df_pred_rating = (
            dataset.test.select('user_id', 'movie_id')
            .join(
                dataset.train
                .group_by('movie_id')
                .agg(pl.col('rating').mean().alias('pred_rating')),
                on='movie_id', how='left'
            )
            .fill_null(0.0)
        )
        # 各ユーザーにおすすめする映画を選択
        # ここでは、各ユーザーに対してまだ評価していない映画の中から評価数が多い映画10作品を選ぶ
        user2recommends = (
            dataset.train.select('movie_id', 'rating')
            .group_by('movie_id')
            .agg(pl.col('rating').mean(), pl.len().alias('count'))
            .filter(pl.col('count') >= minimum_num_rating)
            .join(dataset.train.select('user_id').unique(), how='cross')
            .join(
                dataset.train.select('user_id', 'movie_id', pl.lit(True).alias('already_rated')),
                on=['user_id', 'movie_id'], how='left'
            )
            .filter(pl.col('already_rated').is_null())
            .sort(by='rating', descending=True)
            .group_by('user_id', maintain_order=True).agg(pl.col('movie_id'))
            .with_columns(pl.col('movie_id').list.head(10))
        )
        dict_user2items = {col[0]: col[1] for col in user2recommends.iter_rows()}
        return RecommendResult(df_pred_rating, dict_user2items)
        
    
if __name__ == '__main__':
    minimum_num_ratings = [1, 10, 100, 200]
    for minimum_num_rating in minimum_num_ratings:
        PopularityRecommender().run_sample(minimum_num_rating=minimum_num_rating)