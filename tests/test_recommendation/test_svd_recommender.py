"""Tests for SVD recommender."""

import pandas as pd
import pytest

from src.recommendation.svd_recommender import SVDRecommender


@pytest.fixture
def interactions_df():
    """Create sample interactions for SVD tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
            "recipe_id": [101, 102, 103, 104, 105, 101, 103, 105, 102, 103, 104, 105, 101, 102, 104],
            "rating": [5, 4, 3, 5, 2, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4],
        }
    )


@pytest.fixture
def recipes_df():
    """Create sample recipes for SVD tests."""
    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 105],
            "name": ["Pizza", "Salad", "Pasta", "Soup", "Cake"],
        }
    )


@pytest.fixture
def fitted_recommender(tmp_path, interactions_df, recipes_df):
    """Create a fitted SVD recommender."""
    int_path = tmp_path / "interactions.csv"
    rec_path = tmp_path / "recipes.csv"
    interactions_df.to_csv(int_path, index=False)
    recipes_df.to_csv(rec_path, index=False)

    recommender = SVDRecommender()
    recommender.fit(str(int_path), str(rec_path), k=2)
    return recommender


class TestSVDRecommender:
    """Test cases for SVDRecommender."""

    def test_init(self):
        """Test initialization."""
        recommender = SVDRecommender()
        assert recommender.is_fitted is False
        assert recommender.U is None

    def test_fit(self, fitted_recommender):
        """Test fitting the model."""
        assert fitted_recommender.is_fitted is True
        assert fitted_recommender.U is not None
        assert fitted_recommender.S is not None
        assert fitted_recommender.Vt is not None

    def test_predict_user_ratings_before_fit(self):
        """Test prediction before fitting raises error."""
        recommender = SVDRecommender()
        with pytest.raises(ValueError, match="must be fitted"):
            recommender.predict_user_ratings(0)

    def test_predict_user_ratings(self, fitted_recommender):
        """Test user rating prediction."""
        predictions = fitted_recommender.predict_user_ratings(0)
        assert predictions is not None
        assert len(predictions) > 0

    def test_recommend_for_user_personalized(self, fitted_recommender):
        """Test personalized recommendations."""
        result = fitted_recommender.recommend_for_user(
            1, top_n=3, min_historical_threshold=3
        )

        assert "user_id" in result
        assert "svd_recommendations" in result
        assert "fallback_used" in result
        assert result["recommendation_type"] == "personalized"

    def test_recommend_for_user_fallback(self, fitted_recommender):
        """Test fallback to global popular when insufficient history."""
        result = fitted_recommender.recommend_for_user(
            1, top_n=3, min_historical_threshold=100
        )

        assert result["fallback_used"] is True
        assert result["recommendation_type"] == "global_popular"

    def test_recommend_for_nonexistent_user(self, fitted_recommender):
        """Test recommendation for non-existent user."""
        result = fitted_recommender.recommend_for_user(99999, top_n=3)

        assert result["fallback_used"] is True
        assert result["recommendation_type"] == "global_popular"

    def test_recommend_before_fit(self):
        """Test recommendation before fitting raises error."""
        recommender = SVDRecommender()
        with pytest.raises(ValueError, match="must be fitted"):
            recommender.recommend_for_user(1)

    def test_compute_svd_before_data(self):
        """Test SVD computation before loading data raises error."""
        recommender = SVDRecommender()
        with pytest.raises(ValueError, match="must be loaded"):
            recommender.compute_svd()
