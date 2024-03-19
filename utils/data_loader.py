import polars as pl
import os
from utils.models import Dataset

class DataLoader:
    """Data loader for recommendation system
    
    Args:
    n_user (int): Number of users
    n_test_items (int): Number of test items
    data_path (str): Path to the data
    """
    def __init__(
        self, n_user: int = 1000, n_test_items: int = 5, data_path: str = '../data/ml-10M100K'
    ) -> None:
        self.n_user = n_user
        self.n_test_items = n_test_items
        self.data_path = data_path
        
    def load(self) -> Dataset:
        """Load the dataset
        
        Returns:
            Dataset: Dataset for recommendation system
        """
        movielens, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(movielens)
        # ranking用の評価データは、各ユーザーの評価値が4以上の映画だけを正解とする
        # key: user_id, value: list of item_id
        user2items = (
            movielens_test
            .filter(pl.col('rating') >= 4.0)
            .group_by('user_id').agg(pl.col('movie_id'))
        )
        movielens_test_user2items = {col[0]: col[1] for col in user2items.iter_rows()}
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)
    
    def _split_data(self, df_movielens: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        df_movielens = (
            df_movielens.with_columns(
                pl.col('timestamp').rank(method='ordinal', descending=True)
                .over('user_id')
                .alias('rating_order')
            )
        )
        df_train = df_movielens.filter(pl.col('rating_order') > self.n_test_items)
        df_test = df_movielens.filter(pl.col('rating_order') <= self.n_test_items)
        return df_train, df_test
    
    def _load(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        # 映画の情報の読み込み（10197作品）
        df_movies = (
            pl.read_csv(
                os.path.join(self.data_path, 'movies.dat'),
                has_header=False,
                truncate_ragged_lines=True
            )
            .with_columns(pl.col('column_1').str.split('::'))
            .with_columns(
                pl.col('column_1').list[0].alias('movie_id'),
                pl.col('column_1').list[1].alias('title'),
                (
                    pl.when(pl.col('column_1').list.len() > 2)
                    .then(pl.col('column_1').list[2].str.split('|'))
                    .otherwise(pl.lit([]))
                    .alias('genres')
                )
            )
            .drop('column_1')
        )
        
        # ユーザーが付与した映画のタグ情報の読み込み
        df_tags = (
            pl.read_csv(
                os.path.join(self.data_path, 'tags.dat'),
                has_header=False,
            )
            .with_columns(pl.col('column_1').str.split('::'))
            .with_columns(
                pl.col('column_1').list[0].alias('user_id'),
                pl.col('column_1').list[1].alias('movie_id'),
                pl.col('column_1').list[2].str.to_lowercase().alias('tag'),
                pl.from_epoch(pl.col('column_1').list[3].cast(pl.Int32)).alias('timestamp')
            )
            .drop('column_1')
        )
        
        # tag情報を結合
        df_movies = df_movies.join(
            df_tags.group_by('movie_id').agg(pl.col('tag')),
            on='movie_id', how='left'
        )
        
        # 評価データの読み込み
        df_ratings = (
            pl.read_csv(
                os.path.join(self.data_path, 'ratings.dat'),
                has_header=False,
            )
            .with_columns(pl.col('column_1').str.split('::'))
            .with_columns(
                pl.col('column_1').list[0].alias('user_id'),
                pl.col('column_1').list[1].alias('movie_id'),
                pl.col('column_1').list[2].cast(pl.Float64).alias('rating'),
                pl.from_epoch(pl.col('column_1').list[3].cast(pl.Int32)).alias('timestamp')
            )
            .drop('column_1')
        )
        
        # user数をn_userに制限
        valid_user_ids = (
            df_ratings.get_column('user_id')
            .unique(maintain_order=True)
            .to_list()
            [:self.n_user]
        )
        df_ratings = df_ratings.filter(pl.col('user_id').is_in(valid_user_ids))
        
        # 上記のデータを結合
        df_movielens = df_ratings.join(df_movies, on='movie_id', how='left')
        
        return df_movielens, df_movies