"""
Script de test pour vérifier les optimisations mémoire pour Streamlit Cloud.
Simule le chargement et déchargement des DataFrames.
"""

import sys
from pathlib import Path
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_cache import DataCache
from config.config import MEMORY_CONFIG

def format_memory(bytes_value):
    """Format bytes to MB"""
    return f"{bytes_value / 1024**2:.2f} MB"

def test_memory_optimization():
    """Test des optimisations mémoire"""
    
    print("=" * 80)
    print("TEST DES OPTIMISATIONS MÉMOIRE POUR STREAMLIT CLOUD")
    print("=" * 80)
    
    print(f"\n📋 Configuration actuelle:")
    for key, value in MEMORY_CONFIG.items():
        print(f"   {key}: {value}")
    
    # Test 1: Chargement initial
    print("\n" + "=" * 80)
    print("TEST 1: Chargement des DataFrames")
    print("=" * 80)
    
    print("\n📥 Chargement de recipes...")
    recipes = DataCache.get_recipes(
        path="data/processed/processed_recipes.csv",
        optimize_dtypes=True
    )
    recipes_memory = recipes.memory_usage(deep=True).sum()
    print(f"   ✅ Recipes chargé: {len(recipes):,} lignes")
    print(f"   💾 RAM utilisée: {format_memory(recipes_memory)}")
    
    print("\n📥 Chargement de interactions...")
    interactions = DataCache.get_interactions(
        path="data/processed/interactions_clean.csv",
        optimize_dtypes=True
    )
    interactions_memory = interactions.memory_usage(deep=True).sum()
    print(f"   ✅ Interactions chargé: {len(interactions):,} lignes")
    print(f"   💾 RAM utilisée: {format_memory(interactions_memory)}")
    
    total_memory = recipes_memory + interactions_memory
    print(f"\n💾 RAM TOTALE: {format_memory(total_memory)}")
    
    # Test 2: Vérification du cache
    print("\n" + "=" * 80)
    print("TEST 2: Vérification du Cache")
    print("=" * 80)
    
    print(f"\n   Recipes en cache: {'✅' if DataCache.is_cached('recipes') else '❌'}")
    print(f"   Interactions en cache: {'✅' if DataCache.is_cached('interactions') else '❌'}")
    
    cache_info = DataCache.get_cache_info()
    print(f"\n   Total RAM en cache: {cache_info['total_memory_mb']:.2f} MB")
    
    # Test 3: Déchargement de interactions (simulation page Home)
    print("\n" + "=" * 80)
    print("TEST 3: Simulation Navigation Home (décharge interactions)")
    print("=" * 80)
    
    if MEMORY_CONFIG.get("aggressive_mode", False):
        print("\n🧹 Mode agressif activé, déchargement de interactions...")
        DataCache.clear_cache(clear_recipes=False, clear_interactions=True)
        gc.collect()
        
        print(f"   Recipes en cache: {'✅' if DataCache.is_cached('recipes') else '❌'}")
        print(f"   Interactions en cache: {'✅' if DataCache.is_cached('interactions') else '❌'}")
        
        cache_info = DataCache.get_cache_info()
        print(f"   RAM en cache: {cache_info['total_memory_mb']:.2f} MB")
        print(f"   💾 RAM libérée: {format_memory(interactions_memory)}")
    else:
        print("\n⚠️  Mode agressif désactivé, pas de déchargement")
    
    # Test 4: Rechargement de interactions (simulation page Recommendations)
    print("\n" + "=" * 80)
    print("TEST 4: Simulation Navigation Recommendations (recharge interactions)")
    print("=" * 80)
    
    print("\n📥 Rechargement de interactions...")
    import time
    start = time.time()
    interactions = DataCache.get_interactions(
        path="data/processed/interactions_clean.csv",
        optimize_dtypes=True
    )
    reload_time = time.time() - start
    
    print(f"   ✅ Interactions rechargé: {len(interactions):,} lignes")
    print(f"   ⏱️  Temps de rechargement: {reload_time:.2f}s")
    
    cache_info = DataCache.get_cache_info()
    print(f"   💾 RAM totale: {cache_info['total_memory_mb']:.2f} MB")
    
    # Test 5: Déchargement complet
    print("\n" + "=" * 80)
    print("TEST 5: Déchargement Complet (simulation arrêt)")
    print("=" * 80)
    
    print("\n🧹 Déchargement de tous les caches...")
    DataCache.clear_cache(clear_recipes=True, clear_interactions=True)
    gc.collect()
    
    cache_info = DataCache.get_cache_info()
    print(f"   Recipes en cache: {'✅' if DataCache.is_cached('recipes') else '❌'}")
    print(f"   Interactions en cache: {'✅' if DataCache.is_cached('interactions') else '❌'}")
    print(f"   💾 RAM en cache: {cache_info['total_memory_mb']:.2f} MB")
    
    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES TESTS")
    print("=" * 80)
    
    print(f"""
📊 Résultats:
   • RAM recipes: {format_memory(recipes_memory)}
   • RAM interactions: {format_memory(interactions_memory)}
   • RAM totale: {format_memory(total_memory)}
   • Temps rechargement interactions: {reload_time:.2f}s
   
🎯 Pour Streamlit Cloud (limite 1GB):
   • Peak RAM attendu: ~{format_memory(total_memory + 500*1024**2)} (avec sentiment model)
   • Status: {'✅ Devrait fonctionner' if total_memory + 500*1024**2 < 1024*1024**2 else '⚠️ Risque de crash'}
   
⚙️ Mode actuel:
   • aggressive_mode: {MEMORY_CONFIG.get('aggressive_mode', False)}
   • unload_interactions: {MEMORY_CONFIG.get('unload_interactions_after_use', False)}
   • unload_sentiment_model: {MEMORY_CONFIG.get('unload_sentiment_model', False)}
   
💡 Recommandations:
   • Pour Streamlit Cloud: aggressive_mode = True ✅
   • Pour local: aggressive_mode = False (meilleure UX)
   • Trade-off: +{reload_time:.1f}s pour 25% des navigations
    """)
    
    print("=" * 80)
    print("TEST TERMINÉ ✅")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_memory_optimization()
    except Exception as e:
        print(f"\n❌ ERREUR lors du test: {e}")
        import traceback
        traceback.print_exc()
