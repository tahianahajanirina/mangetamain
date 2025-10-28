# Tests Documentation

## 📋 Structure des Tests

```
tests/
├── conftest.py                          # Fixtures partagées
├── test_data_loader.py                  # Tests preprocessing
├── test_feature_engineering/
│   ├── test_recipe_features.py          # Tests features recettes
│   ├── test_nutrition_features.py       # Tests features nutrition
│   └── test_time_features.py            # Tests features temps
├── test_modeling/
│   ├── test_recipe_clustering.py        # Tests clustering
│   ├── test_nutrition_classifier.py     # Tests classification
│   └── test_time_predictor.py           # Tests prédiction temps
└── test_scripts/
    └── test_clean_data.py               # Tests scripts de nettoyage
```

## 🚀 Lancer les Tests

### Tous les tests
```bash
pytest tests/
```

### Tests spécifiques
```bash
# Un module
pytest tests/test_data_loader.py

# Une classe
pytest tests/test_data_loader.py::TestRecipeDataLoader

# Un test précis
pytest tests/test_data_loader.py::TestRecipeDataLoader::test_load_data
```

### Avec couverture
```bash
pytest tests/ --cov=src --cov-report=html
```

### Tests verbeux
```bash
pytest tests/ -v
```

## 📊 Coverage

Après avoir lancé les tests avec coverage:

```bash
# Ouvrir le rapport HTML
open htmlcov/index.html

# Voir le rapport dans le terminal
pytest tests/ --cov=src --cov-report=term-missing
```

## 🎯 Fixtures Disponibles

Dans `conftest.py`:

- `sample_recipes_df` - DataFrame de recettes pour tests
- `sample_interactions_df` - DataFrame d'interactions pour tests
- `sample_recipes_with_features` - Recettes avec features engineerées
- `temp_data_dir` - Dossier temporaire pour données de test
- `temp_output_dir` - Dossier temporaire pour outputs de test

### Exemple d'utilisation:

```python
def test_my_feature(sample_recipes_df):
    # sample_recipes_df est automatiquement fourni
    result = my_function(sample_recipes_df)
    assert len(result) > 0
```

## ✍️ Écrire de Nouveaux Tests

### Template de base:

```python
"""Tests for my_module."""

import pytest
import pandas as pd
from src.my_module import MyClass


class TestMyClass:
    """Test cases for MyClass."""
    
    def test_initialization(self):
        """Test class initialization."""
        obj = MyClass()
        assert obj is not None
    
    def test_main_functionality(self, sample_recipes_df):
        """Test main functionality with fixtures."""
        obj = MyClass()
        result = obj.process(sample_recipes_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_error_handling(self):
        """Test error handling."""
        obj = MyClass()
        
        with pytest.raises(ValueError):
            obj.process(None)
```

## 🏷️ Markers

Utiliser des markers pour organiser les tests:

```python
@pytest.mark.slow
def test_long_running():
    """Test qui prend du temps."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Test d'intégration."""
    pass
```

Lancer les tests par marker:
```bash
# Exclure les tests lents
pytest -m "not slow"

# Seulement les tests d'intégration
pytest -m "integration"
```

## 🔧 Configuration pytest

Dans `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--cov=src",
    "--cov-report=term-missing",
]
```

## 🐛 Debugging Tests

### Afficher les prints
```bash
pytest tests/ -s
```

### Arrêter au premier échec
```bash
pytest tests/ -x
```

### Debugger avec pdb
```bash
pytest tests/ --pdb
```

### Dans le code:
```python
def test_my_function():
    result = my_function()
    import pdb; pdb.set_trace()  # Point d'arrêt
    assert result == expected
```

## 📈 Bonnes Pratiques

### ✅ À faire:

1. **Un test = une assertion principale**
   ```python
   def test_length():
       result = process_data()
       assert len(result) == 5
   ```

2. **Noms descriptifs**
   ```python
   def test_removes_duplicates_from_recipes():
       # Clair et précis
       pass
   ```

3. **Arrange-Act-Assert**
   ```python
   def test_calculation():
       # Arrange
       data = create_test_data()
       calculator = Calculator()
       
       # Act
       result = calculator.compute(data)
       
       # Assert
       assert result == expected_value
   ```

4. **Tests isolés**
   - Chaque test doit être indépendant
   - Utiliser des fixtures pour le setup

### ❌ À éviter:

1. **Tests qui dépendent d'autres tests**
2. **Tests qui modifient des fichiers réels**
3. **Tests trop longs ou complexes**
4. **Assertions multiples non liées**

## 🔍 Mock et Patch

Pour tester sans dépendances externes:

```python
from unittest.mock import Mock, patch

def test_with_mock():
    # Mock un objet
    mock_loader = Mock()
    mock_loader.load_data.return_value = pd.DataFrame({'id': [1, 2, 3]})
    
    # Utiliser le mock
    result = process_with_loader(mock_loader)
    assert len(result) == 3

@patch('src.module.external_api_call')
def test_with_patch(mock_api):
    # Patch une fonction
    mock_api.return_value = {'status': 'ok'}
    
    result = function_that_calls_api()
    assert result['status'] == 'ok'
```

## 📊 CI/CD Integration

Les tests s'exécutent automatiquement sur GitHub Actions:

- ✅ À chaque push
- ✅ Sur Python 3.9, 3.10, 3.11
- ✅ Avec rapport de couverture

Voir `.github/workflows/ci.yml` pour les détails.

## 🆘 Problèmes Fréquents

### Import errors
```bash
# Installer le package en mode dev
pip install -e ".[dev,test]"
```

### Fixtures non trouvées
```bash
# Vérifier que conftest.py est au bon endroit
tests/conftest.py
```

### Tests lents
```bash
# Utiliser des données plus petites
# Ou marker comme @pytest.mark.slow
```

## 📚 Ressources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
