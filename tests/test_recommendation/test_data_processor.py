"""Tests for recommendation data processor."""

import pandas as pd
import pytest

from src.recommendation.data_processor import DataProcessor


@pytest.fixture
def interactions_df():
    """Create sample interactions for recommendation tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
            "recipe_id": [101, 102, 103, 101, 104, 102, 103, 104, 105, 101],
            "rating": [5, 4, 3, 5, 2, 4, 5, 3, 4, 5],
        }
    )


@pytest.fixture
def recipes_df():
    """Create sample recipes for recommendation tests."""
    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 105],
            "name": ["Pizza", "Salad", "Pasta", "Soup", "Cake"],
        }
    )


@pytest.fixture
def processor(tmp_path, interactions_df, recipes_df):
    """Create a DataProcessor with loaded data."""
    int_path = tmp_path / "interactions.csv"
    rec_path = tmp_path / "recipes.csv"
    interactions_df.to_csv(int_path, index=False)
    recipes_df.to_csv(rec_path, index=False)

    dp = DataProcessor()
    dp.load_data(str(int_path), str(rec_path))
    return dp


class TestDataProcessor:
    """Test cases for DataProcessor."""

    def test_init(self):
        """Test initialization."""
        dp = DataProcessor()
        assert dp.df_interactions is None
        assert dp.df_recipes is None
        assert dp.sparse_matrix is None

    def test_load_data(self, processor):
        """Test data loading."""
        assert processor.df_interactions is not None
        assert processor.df_recipes is not None

    def test_create_sparse_matrix(self, processor):
        """Test sparse matrix creation."""
        processor.create_sparse_matrix()

        assert processor.sparse_matrix is not None
        assert processor.user_cat is not None
        assert processor.recipe_cat is not None

    def test_create_sparse_matrix_not_loaded(self):
        """Test sparse matrix creation before loading data."""
        dp = DataProcessor()
        with pytest.raises(ValueError, match="Interaction data must be loaded"):
            dp.create_sparse_matrix()

    def test_is_ready(self, processor):
        """Test readiness check."""
        assert not processor.is_ready()

        processor.create_sparse_matrix()
        assert processor.is_ready()

    def test_get_user_index(self, processor):
        """Test user index lookup."""
        processor.create_sparse_matrix()
        idx, uid = processor.get_user_index(1)

        assert isinstance(idx, int)
        assert uid == 1

    def test_get_user_index_not_found(self, processor):
        """Test user index lookup for missing user."""
        processor.create_sparse_matrix()
        with pytest.raises(ValueError, match="not found"):
            processor.get_user_index(999)

    def test_get_user_by_index(self, processor):
        """Test user retrieval by index."""
        processor.create_sparse_matrix()
        user_id = processor.get_user_by_index(0)

        assert user_id is not None

    def test_get_user_by_index_out_of_bounds(self, processor):
        """Test user retrieval with invalid index."""
        processor.create_sparse_matrix()
        with pytest.raises(IndexError):
            processor.get_user_by_index(9999)

    def test_get_user_historical_recipes(self, processor):
        """Test getting user's historical recipes."""
        processor.create_sparse_matrix()
        result = processor.get_user_historical_recipes(1, top_n=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_get_user_historical_recipes_empty(self, processor):
        """Test getting history for user with no data."""
        processor.create_sparse_matrix()
        result = processor.get_user_historical_recipes(999, top_n=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_global_popular_recipes(self, processor):
        """Test getting globally popular recipes."""
        processor.create_sparse_matrix()
        result = processor.get_global_popular_recipes(top_n=3, min_ratings=1)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_get_user_interaction_count(self, processor):
        """Test interaction count for user."""
        count = processor.get_user_interaction_count(1)
        assert count == 3  # user 1 has 3 interactions

    def test_get_user_rated_recipes(self, processor):
        """Test getting rated recipe IDs."""
        rated = processor.get_user_rated_recipes(1)
        assert isinstance(rated, set)
        assert 101 in rated

    def test_get_recipe_names(self, processor):
        """Test getting recipe names."""
        result = processor.get_recipe_names([101, 102])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_recipe_ids_list(self, processor):
        """Test getting recipe ID list."""
        processor.create_sparse_matrix()
        ids = processor.get_recipe_ids_list()
        assert isinstance(ids, list)
        assert len(ids) > 0
