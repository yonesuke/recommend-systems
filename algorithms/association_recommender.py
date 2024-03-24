import sys
sys.path.append('..')

from utils.models import RecommendResult, Dataset
from algorithms.base_recommender import BaseRecommender
import polars as pl
from efficient_apriori import apriori

def get_association_rules(
    dataset: Dataset,
    min_rating: float = 4.0,
    min_support: float = 0.1,
    min_threshold: float = 0.1
) -> pl.DataFrame:
    _, rules = apriori(
        (
            dataset.train
            .filter(pl.col('rating')>=min_rating)
            .group_by('user_id')
            .agg(pl.col('movie_id'))
            .get_column('movie_id').to_list()
        ),
        min_support=min_support, min_confidence=0.0
    )
    num_user = dataset.train.get_column('user_id').n_unique()
    return (
        pl.DataFrame([
            pl.Series(name='lhs', values=[rule.lhs for rule in rules]),
            pl.Series(name='rhs', values=[rule.rhs for rule in rules]),
            pl.Series(name='lift', values=[rule.lift for rule in rules]),
            pl.Series(name='confidence', values=[rule.confidence for rule in rules]),
            pl.Series(name='lhs_support', values=[rule.count_lhs/num_user for rule in rules]),
            pl.Series(name='rhs_support', values=[rule.count_rhs/num_user for rule in rules])
        ])
        .filter(pl.col('lift')>=min_threshold)
        .sort(by='lift', descending=True)
    )

class AssociationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        min_support = kwargs.get('min_support', 0.1)
        min_threshold = kwargs.get('min_threshold', 0.1)
        min_rating = kwargs.get('min_rating', 4.0)
        # association rules
        df_rules = get_association_rules(
            dataset,
            min_rating=min_rating,
            min_support=min_support,
            min_threshold=min_threshold
        )
        
        # recommend 10 movies based on association rules
        # which are not yet rated
        
        # 5 latest rated movies
        recent_user2items = (
            dataset.train.filter(pl.col('rating')>=4.0)
            .sort(by='timestamp', descending=True)
            .group_by('user_id').agg(pl.col('movie_id').head(5))
        )
        dict_recent_user2items = {col[0]: col[1] for col in recent_user2items.iter_rows()}
        # previous all rated movies
        rated_user2items = dataset.train.group_by('user_id').agg(pl.col('movie_id'))
        dict_rated_user2items = {col[0]: col[1] for col in rated_user2items.iter_rows()}
        # all user ids
        user_ids = dataset.train.get_column('user_id').unique().to_list()
        # RECOMMEND!!
        dict_recommend_user2items = dict()
        for user_id in user_ids:
            try:
                recent_items: list[str] = dict_recent_user2items[user_id]
                rated_items: list[str] = dict_rated_user2items[user_id]
            except:
                continue
            movie_ids = (
                df_rules
                .with_columns(pl.col('lhs').list.set_intersection(recent_items).alias('intersection'))
                .filter(pl.col('intersection').list.len()>0)
                .get_column('rhs').explode().value_counts(sort=True)
                .with_columns(pl.col('rhs').is_in(rated_items).not_().alias('not_yet_rated'))
                .filter(pl.col('not_yet_rated'))
                .head(10)
                .get_column('rhs').to_list()
            )
            dict_recommend_user2items[user_id] = movie_ids
        # return
        return RecommendResult(
            dataset.test.with_columns(pl.col('rating').alias('pred_rating')),
            dict_recommend_user2items
        )
    
if __name__ == '__main__':
    minimum_supports = [0.06, 0.08, 0.10, 0.12, 0.14]
    for minimum_support in minimum_supports:
        AssociationRecommender().run_sample(min_support=minimum_support)
