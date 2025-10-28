# MangeTaMain - Your Personal Recipe Discovery Platform

Welcome to **MangeTaMain**, an intelligent recipe discovery platform that helps you find the perfect recipes based on your taste, dietary needs, and cooking preferences!

---

## What is MangeTaMain?

MangeTaMain is a smart cooking companion that analyzes over 230,000 recipes and 1 million user reviews to help you discover recipes you'll love. Whether you're looking for quick weeknight dinners, healthy meals, or exploring new cuisines, MangeTaMain has you covered!

### Key Features

- **Personalized Suggestions**: Get recipe recommendations tailored to your taste
- **Smart Search**: Find recipes similar to ones you already love
- **Time Estimation**: Know how long a recipe will take before you start
- **Nutrition Information**: Understand the nutritional profile of any recipe
- **Recipe Categories**: Browse recipes organized by cooking time, complexity, and health profile
- **User Reviews Analysis**: See what other home cooks think about recipes

---

## Getting Started

### Installation Requirements

Before you begin, make sure you have:
- Python 3.8 or higher installed on your computer
- Basic familiarity with running commands in a terminal

### Step 1: Download the Project

```bash
# Clone the project from GitHub
git clone https://github.com/[your-organization]/mangetamain.git
cd mangetamain
```

### Step 2: Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

### Step 3: Prepare Your Data

1. Download the recipe dataset from [Kaggle - Food.com Recipes](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)
2. Place the following files in the `data/` folder:
   - `RAW_recipes.csv` - Contains all recipe information
   - `RAW_interactions.csv` - Contains user ratings and reviews

### Step 4: Launch the Application

```bash
# Start the web application
streamlit run streamlit_app_final.py
```

The application will open in your web browser automatically!

---

## How to Use MangeTaMain

### Discovering New Recipes

**Option 1: Get Personalized Suggestions**
1. Navigate to "Personalized Recommendations"
2. Enter your User ID (or try one of the sample IDs provided)
3. Choose how many recipe suggestions you want (5-20)
4. Apply filters if you have specific needs:
   - Maximum cooking time
   - High protein recipes
   - Low calorie options
5. Click "Get Recommendations"

**Option 2: Find Similar Recipes**
1. Go to "Recipe-Based Recommendations"
2. Search for a recipe you like (e.g., "chocolate cake", "pasta carbonara")
3. Select the recipe from the search results
4. Click "Find Similar Recipes" to discover comparable options

### Understanding Cooking Time

Want to know how long a recipe will take?

1. Navigate to "Cooking Time Prediction"
2. **Enter New Recipe**: Input your recipe details (steps, ingredients, cooking methods)
3. **Search Existing Recipe**: Find a recipe from our database
4. Get an instant time estimate to help with meal planning

### Checking Nutritional Information

1. Go to "Nutrition Information"
2. **Enter Values**: Input nutritional data for any recipe
3. **Search Recipe**: Find an existing recipe in the database
4. See instant categorization:
   - High/Low Calorie
   - High Protein
   - Low Fat
   - Low Sugar
   - Overall Health Score

### Exploring Recipe Categories

1. Visit "Recipe Categories"
2. **Classify Your Recipe**: Enter recipe characteristics to find its category
3. **View Overview**: Explore different recipe types:
   - Quick & Simple meals
   - Healthy & Balanced options
   - Elaborate dishes
   - Desserts & Treats
   - Complex recipes

### Reading User Reviews

1. Navigate to "Review Analysis"
2. **Write Your Own**: Get instant sentiment analysis on your review
3. **Analyze Existing**: Enter a Recipe ID to see what others think
4. View overall sentiment (Positive, Neutral, Negative)

### Exploring the Dataset

Visit "Dataset Insights" to:
- See overall statistics about our recipe collection
- Explore cooking time distributions
- Understand recipe complexity patterns
- View nutritional trends

---

## Understanding Your Results

### Recipe Cards

Each recipe recommendation shows:
- **Recipe Name**: The dish you'll be making
- **Time**: How long it takes to prepare
- **Steps**: Number of cooking steps
- **Ingredients**: How many ingredients you'll need
- **Calories**: Per serving
- **Categories**: Health tags and recipe type

### Recommendation Types

- **Personalized**: Based on your past preferences and similar users
- **Popular**: Highly-rated recipes loved by the community
- **Content-Based**: Similar to recipes you already enjoy

### Health Categories

- **High Protein**: Great for fitness and muscle building
- **Low Calorie**: Perfect for weight management
- **Low Fat**: Heart-healthy options
- **Low Sugar**: Diabetic-friendly or sugar-conscious choices
- **Healthy Recipe**: Overall nutritious and balanced

### Recipe Categories

- **Quick & Simple**: Fast meals for busy days (15-30 minutes)
- **Healthy & Balanced**: Nutritious everyday options
- **Elaborate Meals**: Special occasion cooking (60+ minutes)
- **Complex Dishes**: Advanced techniques for cooking enthusiasts

---

## Tips for Best Results

### Getting Better Recommendations

1. **Be Active**: The more recipes you rate, the better your suggestions become
2. **Use Filters**: Narrow down results based on your needs (time, nutrition)
3. **Try Both Modes**: User-based for variety, recipe-based for similar dishes
4. **Explore Categories**: Find recipes that match your cooking style

### Planning Your Meals

1. **Check Time Estimates**: Use the cooking time prediction before shopping
2. **Review Nutrition**: Ensure recipes fit your dietary goals
3. **Read Reviews**: See what challenges others faced
4. **Start Simple**: Try quick recipes first, then explore complex ones

### Finding the Right Recipe

1. **Search Broadly**: Use general terms like "chicken" or "pasta"
2. **Apply Filters**: Add time or nutrition constraints
3. **Compare Options**: Look at multiple suggestions before deciding
4. **Check Ingredients**: Make sure you have what you need

---

## Dataset Information

MangeTaMain is powered by comprehensive recipe data:

- **230,000+ Recipes**: From Food.com community
- **1,000,000+ Reviews**: Real user feedback and ratings
- **18 Years of Data**: Recipe trends from 2000-2018
- **Detailed Information**: Ingredients, steps, nutrition, cooking time

### What Each Recipe Includes

- Ingredient list and quantities
- Step-by-step cooking instructions
- Preparation and cooking time
- Nutritional information (calories, protein, fat, sugar, etc.)
- User ratings and reviews
- Recipe tags and categories

---

## Frequently Asked Questions

**Q: Do I need to create an account?**
A: Currently, you can explore recipes using existing User IDs from our dataset. Sample IDs are provided in the app.

**Q: How accurate are the time predictions?**
A: Our predictions are typically within 15 minutes of actual cooking time, based on recipe complexity and methods.

**Q: Can I add my own recipes?**
A: The current version works with existing recipes. Custom recipe support may be added in future updates.

**Q: What if I have dietary restrictions?**
A: Use the nutrition filters to find recipes that match your needs (low calorie, high protein, etc.).

**Q: How are recommendations generated?**
A: We analyze your preferences and compare them with similar users who enjoyed recipes you haven't tried yet.

**Q: Can I save my favorite recipes?**
A: This feature is planned for future releases. Currently, note down Recipe IDs you like.

---

## Troubleshooting

### Application Won't Start

1. Check that Python 3.8+ is installed: `python --version`
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure data files are in the correct location

### No Recommendations Showing

1. Try a different User ID
2. Remove strict filters (time, nutrition)
3. Increase the number of recommendations requested
4. Check that data files are loaded correctly

### Slow Performance

1. Reduce the number of recommendations (try 5-10 instead of 20)
2. Disable enrichment temporarily
3. Close other applications to free up memory

---

## Project Structure

```
mangetamain/
├── data/                       # Recipe and user data
│   ├── RAW_recipes.csv        # All recipe information
│   └── RAW_interactions.csv   # User ratings and reviews
├── src/                       # Application source code
│   ├── integration/           # Core recommendation system
│   ├── modeling/              # Prediction and analysis
│   └── sentiment_analysis/    # Review analysis
├── outputs/                   # Saved results
├── streamlit_app_final.py    # Main application
├── requirements.txt          # Python dependencies
└── README.md                 # This guide
```

---

## Support and Feedback

Having issues or suggestions? Here's how to get help:

1. **Check this guide**: Most common questions are answered here
2. **Review error messages**: They often point to the solution
3. **Open an issue**: Report bugs or request features on GitHub
4. **Contact the team**: Reach out to the project maintainers

---

## What's Next?

MangeTaMain is continuously improving! Planned features include:

- User account creation
- Personal recipe collections
- Shopping list generation
- Meal planning calendar
- Social sharing features
- Mobile app version

---

## Credits

MangeTaMain is built on:
- **Dataset**: Food.com Recipes and Interactions (Kaggle)
- **Technology**: Python, Streamlit, and open-source tools
- **Community**: Contributions from home cooks and developers

---

## License and Terms

This project is for educational and personal use. Recipe data is courtesy of Food.com and Kaggle.

---

**Last Updated**: 2025-10-28

**Version**: 2.0.0

---

Enjoy cooking with MangeTaMain! Happy cooking! 👨‍🍳👩‍🍳
