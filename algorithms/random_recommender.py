import sys
sys.path.append('..')

from utils.models import RecommendResult, Dataset
from algorithms.base_recommender import BaseRecommender
import polars as pl
import numpy as np

rng = np.random.default_rng(0)

class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # RMSEを計算するために、テストデータに出てくるユーザーとアイテムごとのレーティング予測評価値を格納
        # 各映画の評価はランダムに生成
        df_pred_rating = (
            dataset.test.select('user_id', 'movie_id')
            .with_columns(
                pl.Series(
                    name='pred_rating',
                    values=np.random.uniform(0.5, 5.0, dataset.test.shape[0])
                )
            )
        )
        # 各ユーザーにおすすめする映画をランダムに選択
        # ここでは、各ユーザーに対してまだ評価していない映画の中からランダムに10作品選ぶ
        movie_ids = dataset.train.get_column('movie_id').unique().to_list()
        user2recommends = (
            dataset.train
            .group_by('user_id').agg(
                pl.col('movie_id').alias('already_rated'),
                pl.lit(movie_ids).alias('all_movie_ids')
            )
            .with_columns(
                pl.col('all_movie_ids')
                .list.set_difference(pl.col('already_rated'))
                .list.sample(10)
                .alias('recommendations')
            )
            .select('user_id', 'recommendations')
        )
        dict_user2items = {col[0]: col[1] for col in user2recommends.iter_rows()}
        return RecommendResult(df_pred_rating, dict_user2items)
        
if __name__ == '__main__':
    RandomRecommender().run_sample()