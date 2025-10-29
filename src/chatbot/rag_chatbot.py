"""
Simple RAG Chatbot using Google Gemini Flash 2.5

This module provides a straightforward RAG (Retrieval Augmented Generation) chatbot
that can answer questions about recipes using Google's Gemini model.
"""

import logging
from typing import Dict, List, Optional

import google.generativeai as genai
import numpy as np
import pandas as pd

from src.utils.data_cache import DataCache

logger = logging.getLogger(__name__)


class RecipeRAGChatbot:
    """Simple RAG chatbot for recipe dataset using Gemini Flash 2.5"""

    def __init__(self, api_key: str, recipes_df: pd.DataFrame):
        """
        Initialize the RAG chatbot.

        Args:
            api_key: Google API key for Gemini
            recipes_df: DataFrame containing recipe data
        """
        self.api_key = api_key
        self.recipes_df = recipes_df

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Prepare recipe context
        self._prepare_recipe_context()

        logger.info(f"RAG Chatbot initialized with {len(recipes_df)} recipes")

    def _prepare_recipe_context(self):
        """Prepare a searchable recipe context from the dataset"""
        # Create a simplified text representation of each recipe
        self.recipe_texts = []
        self.recipe_ids = []

        for idx, row in self.recipes_df.iterrows():
            # Create a text summary of the recipe
            recipe_text = f"""
Recipe: {row['name']}
ID: {row['id']}
Cooking Time: {row['minutes']} minutes
Steps: {row['n_steps']}
Ingredients: {row['n_ingredients']}
"""
            # Add nutrition if available
            if "calories" in row and pd.notna(row["calories"]):
                recipe_text += f"Calories: {row['calories']:.0f}\n"

            # Add tags if available
            if "tags" in row and isinstance(row["tags"], list):
                recipe_text += f"Tags: {', '.join(row['tags'][:5])}\n"

            self.recipe_texts.append(recipe_text)
            self.recipe_ids.append(row["id"])

    def _search_relevant_recipes(self, query: str, top_k: int = 5) -> List[str]:
        """
        Simple keyword-based search for relevant recipes.

        Args:
            query: User's question
            top_k: Number of recipes to retrieve

        Returns:
            List of relevant recipe texts
        """
        # Simple keyword matching
        query_lower = query.lower()
        scores = []

        for recipe_text in self.recipe_texts:
            recipe_lower = recipe_text.lower()
            # Count keyword matches
            score = sum(1 for word in query_lower.split() if word in recipe_lower)
            scores.append(score)

        # Get top-k recipes by score
        top_indices = np.argsort(scores)[-top_k:][::-1]
        relevant_recipes = [self.recipe_texts[i] for i in top_indices if scores[i] > 0]

        return relevant_recipes if relevant_recipes else self.recipe_texts[:top_k]

    def chat(self, user_query: str, chat_history: Optional[List[Dict]] = None) -> str:
        """
        Process a user query and generate a response using RAG.

        Args:
            user_query: The user's question
            chat_history: Optional chat history for context

        Returns:
            The chatbot's response
        """
        try:
            # Search for relevant recipes
            relevant_recipes = self._search_relevant_recipes(user_query, top_k=5)

            # Build context from relevant recipes
            context = "\n---\n".join(relevant_recipes)

            # Build prompt for Gemini
            prompt = f"""You are a helpful recipe assistant. Answer the user's question based on the recipe information provided below.

Recipe Database Context:
{context}

User Question: {user_query}

Please provide a helpful, accurate, and concise answer based on the recipes provided. If the information is not available in the recipes, say so politely."""

            # Add chat history if available
            if chat_history:
                history_text = "\n".join(
                    [
                        f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                        for msg in chat_history[-3:]  # Last 3 exchanges
                    ]
                )
                prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"

            # Generate response
            response = self.model.generate_content(prompt)

            return response.text

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return (
                f"I apologize, but I encountered an error: {str(e)}. Please try again."
            )

    def get_recipe_stats(self) -> Dict[str, any]:
        """Get statistics about the recipe dataset"""
        stats = {
            "total_recipes": len(self.recipes_df),
            "avg_time": self.recipes_df["minutes"].mean(),
            "median_time": self.recipes_df["minutes"].median(),
            "avg_steps": self.recipes_df["n_steps"].mean(),
            "avg_ingredients": self.recipes_df["n_ingredients"].mean(),
        }

        if "calories" in self.recipes_df.columns:
            stats["avg_calories"] = self.recipes_df["calories"].mean()

        return stats


def create_chatbot(api_key: str, recipes_path: str) -> RecipeRAGChatbot:
    """
    Factory function to create a RAG chatbot instance.

    Args:
        api_key: Google API key for Gemini
        recipes_path: Path to the recipes CSV file

    Returns:
        Initialized RecipeRAGChatbot instance
    """
    # Load recipes using cache
    logger.info(f"Loading recipes from {recipes_path}")
    recipes_df = DataCache.get_recipes(path=str(recipes_path), optimize_dtypes=True)

    # Parse tags if they exist
    if "tags" in recipes_df.columns:
        import ast

        recipes_df["tags"] = recipes_df["tags"].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
        )

    return RecipeRAGChatbot(api_key, recipes_df)
