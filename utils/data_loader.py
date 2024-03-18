import pandas as pd
import os
from utils.models import Dataset

class DataLoader:
    def __init__(
        self, n_user: int = 1000, n_test_items: int = 5, data_path: str = '../data/ml-10M100K'
    ) -> None:
        self.n_user = n_user
        self.n_test_items = n_test_items
        self.data_path = data_path
        
    def load(self) -> Dataset:
        movielens, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(movielens)
        # ranking用の評価データは、各ユーザーの評価値が4以上の映画だけを正解とする
        # key: user_id, value: list of item_id
        movielens_test_user2items = (
            movielens_test[movielens_test['rating'] >= 4]
            .groupby('userId')['movieId']
            .apply(list)
            .to_dict()
        )
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)
    
    def _split_data(self, movielens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        movielens['rating_order'] = (
            movielens.groupby('userId')['timestamp']
            .rank(method='first', ascending=False)
        )
        movielens_train = movielens[movielens['rating_order'] > self.n_test_items]
        movielens_test = movielens[movielens['rating_order'] <= self.n_test_items]
        return movielens_train, movielens_test
    
    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 映画の情報の読み込み（10197作品）
        m_cols = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(
            os.path.join(self.data_path, 'movies.dat'),
            sep='::',
            encoding='latin-1',
            names=m_cols,
            engine='python'
        )
        movies['genres'] = movies['genres'].str.split('|')
        
        # ユーザーが付与した映画のタグ情報の読み込み
        t_cols = ['user_id', 'movie_id', 'tag', 'timestamp']
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, 'tags.dat'),
            sep='::',
            names=t_cols,
            engine='python'
        )
        user_tagged_movies['tag'] = user_tagged_movies['tag'].str.lower()
        movie_tags = user_tagged_movies.groupby('movie_id')['tag'].apply(list).reset_index()
        
        # tag情報を結合
        movies = movies.merge(movie_tags, on='movie_id', how='left')
        
        # 評価データの読み込み
        r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(
            os.path.join(self.data_path, 'ratings.dat'),
            sep='::',
            names=r_cols,
            engine='python'
        )
        
        # user数をn_userに制限
        valid_user_ids = sorted(ratings['user_id'].unique()[:self.n_user])
        ratings = ratings[ratings['user_id'].isin(valid_user_ids)]
        
        # 上記のデータを結合
        movielens = ratings.merge(movies, on='movie_id')
        
        return movielens, movies