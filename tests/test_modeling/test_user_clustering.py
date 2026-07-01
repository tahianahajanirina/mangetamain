"""Tests for user clustering model."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.user_clustering import UserClusterer


@pytest.fixture
def user_features_df():
    """Create sample user features for clustering tests."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame(
        {
            "user_id": range(1, n + 1),
            "n_interactions": np.random.randint(5, 100, n),
            "avg_rating": np.random.uniform(2.0, 5.0, n),
            "avg_minutes": np.random.uniform(10, 120, n),
            "std_minutes": np.random.uniform(5, 40, n),
            "avg_n_steps": np.random.uniform(3, 15, n),
            "avg_n_ingredients": np.random.uniform(3, 15, n),
            "avg_calories": np.random.uniform(100, 800, n),
        }
    )


class TestUserClusterer:
    """Test cases for UserClusterer."""

    def test_init_default(self):
        """Test default initialization."""
        clusterer = UserClusterer()
        assert clusterer.pca_variance == 0.95
        assert clusterer.n_clusters is None
        assert clusterer.random_state == 42

    def test_init_custom(self):
        """Test custom initialization."""
        clusterer = UserClusterer(n_clusters=4, pca_variance=0.90, random_state=123)
        assert clusterer.n_clusters == 4
        assert clusterer.pca_variance == 0.90
        assert clusterer.random_state == 123

    def test_load_features(self, tmp_path, user_features_df):
        """Test loading user features from file."""
        path = tmp_path / "users.csv"
        user_features_df.to_csv(path, index=False)

        clusterer = UserClusterer(n_clusters=3)
        result = clusterer.load_features(str(path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(user_features_df)
        assert clusterer.users is not None

    def test_prepare_features(self, user_features_df):
        """Test feature preparation (scaling + PCA)."""
        clusterer = UserClusterer(n_clusters=3)
        clusterer.users = user_features_df

        scaled, pca = clusterer.prepare_features()

        assert scaled is not None
        assert pca is not None
        assert len(scaled) == len(user_features_df)

    def test_fit(self, tmp_path, user_features_df):
        """Test complete fit pipeline."""
        path = tmp_path / "users.csv"
        user_features_df.to_csv(path, index=False)

        clusterer = UserClusterer(n_clusters=3)
        result = clusterer.fit(features_path=str(path))

        assert result is clusterer
        assert clusterer.kmeans is not None
        assert "cluster" in clusterer.users.columns
        assert clusterer.users["cluster"].nunique() == 3

    def test_name_clusters(self, tmp_path, user_features_df):
        """Test cluster naming."""
        path = tmp_path / "users.csv"
        user_features_df.to_csv(path, index=False)

        clusterer = UserClusterer(n_clusters=3)
        clusterer.fit(features_path=str(path))
        names = clusterer.name_clusters()

        assert isinstance(names, dict)
        assert len(names) == 3
        assert "cluster_name" in clusterer.users.columns

    def test_get_stats(self, tmp_path, user_features_df):
        """Test statistics retrieval."""
        path = tmp_path / "users.csv"
        user_features_df.to_csv(path, index=False)

        clusterer = UserClusterer(n_clusters=3)
        clusterer.fit(features_path=str(path))
        stats = clusterer.get_stats()

        assert "n_clusters" in stats
        assert "n_users" in stats
        assert "silhouette" in stats
        assert stats["n_clusters"] == 3

    def test_get_stats_before_fit_raises(self):
        """Test that get_stats before fit raises error."""
        clusterer = UserClusterer()
        with pytest.raises(ValueError, match="Clustering non effectué"):
            clusterer.get_stats()

    def test_save_results(self, tmp_path, user_features_df):
        """Test saving clustering results."""
        features_path = tmp_path / "users.csv"
        user_features_df.to_csv(features_path, index=False)

        clusterer = UserClusterer(n_clusters=3)
        clusterer.fit(features_path=str(features_path))

        output_path = tmp_path / "users_clustered.csv"
        clusterer.save_results(str(output_path))

        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert "cluster" in saved_df.columns
