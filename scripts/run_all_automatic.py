"""
ORCHESTRATEUR AUTOMATIQUE - Exécution Complète du Projet

Ce script exécute automatiquement toutes les étapes du projet :
1. Vérification des données
2. Entraînement LSTM (si pas déjà fait)
3. Entraînement XGBoost (si pas déjà fait) 
4. Comparaison complète de tous les modèles
5. Génération des visualisations
6. Création du rapport final

Usage:
    python scripts/run_all_automatic.py

Author: Climate Prediction Team
Date: December 2025
"""

import subprocess
import sys
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_banner(text):
    """Affiche un banner formaté."""
    logger.info("\n" + "=" * 80)
    logger.info(f"  {text}")
    logger.info("=" * 80)


def check_file_exists(filepath):
    """Vérifie si un fichier existe."""
    return Path(filepath).exists()


def run_command(command, description, check_exit=True):
    """Exécute une commande et affiche le résultat."""
    logger.info(f"\n🚀 {description}")
    logger.info(f"   Commande: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            check=check_exit,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCÈS")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            logger.error(f"❌ {description} - ÉCHEC")
            if result.stderr:
                print(result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - ERREUR")
        print(e.stderr)
        return False
    except FileNotFoundError:
        logger.error(f"❌ Commande introuvable: {command[0]}")
        return False


def main():
    """Fonction principale d'orchestration."""
    print_banner("🤖 ORCHESTRATEUR AUTOMATIQUE - PROJET CLIMAT")
    
    base_path = Path(__file__).parent.parent
    
    # Statistiques
    start_time = time.time()
    steps_completed = 0
    steps_failed = 0
    
    # ÉTAPE 1: Vérification des données
    print_banner("📂 ÉTAPE 1/5 - VÉRIFICATION DES DONNÉES")
    
    data_files = {
        'train': base_path / 'data' / 'processed' / 'splits' / 'train.parquet',
        'val': base_path / 'data' / 'processed' / 'splits' / 'val.parquet',
        'test': base_path / 'data' / 'processed' / 'splits' / 'test.parquet',
        'scaler': base_path / 'data' / 'processed' / 'splits' / 'scaler_new.pkl',
        'imputer': base_path / 'data' / 'processed' / 'splits' / 'imputer_new.pkl'
    }
    
    all_exist = True
    for name, filepath in data_files.items():
        if check_file_exists(filepath):
            logger.info(f"✅ {name}: {filepath.name}")
        else:
            logger.error(f"❌ {name}: MANQUANT - {filepath}")
            all_exist = False
    
    if not all_exist:
        logger.error("\n❌ Certains fichiers de données sont manquants!")
        logger.info("💡 Exécutez d'abord les scripts de preprocessing.")
        return
    
    logger.info("\n✅ Toutes les données sont présentes")
    steps_completed += 1
    
    # ÉTAPE 2: LSTM
    print_banner("🧠 ÉTAPE 2/5 - ENTRAÎNEMENT LSTM (DEEP LEARNING)")
    
    lstm_model_path = base_path / 'models' / 'lstm' / 'lstm_model.h5'
    lstm_metrics_path = base_path / 'models' / 'lstm' / 'lstm_metrics.csv'
    
    if check_file_exists(lstm_model_path) and check_file_exists(lstm_metrics_path):
        logger.info("ℹ️  Modèle LSTM déjà entraîné")
        logger.info(f"   Modèle: {lstm_model_path}")
        logger.info(f"   Métriques: {lstm_metrics_path}")
        steps_completed += 1
    else:
        logger.info("🚀 Entraînement du LSTM en cours...")
        logger.info("⏱️  Temps estimé: 30-60 minutes")
        logger.info("⚠️  Cette étape peut être longue, soyez patient...")
        
        success = run_command(
            [sys.executable, str(base_path / 'src' / 'models' / 'lstm_model_complete.py')],
            "Entraînement LSTM",
            check_exit=False
        )
        
        if success and check_file_exists(lstm_model_path):
            logger.info("✅ LSTM entraîné avec succès")
            steps_completed += 1
        else:
            logger.warning("⚠️  LSTM non entraîné (peut continuer sans)")
            steps_failed += 1
    
    # ÉTAPE 3: XGBoost
    print_banner("🌳 ÉTAPE 3/5 - ENTRAÎNEMENT XGBOOST")
    
    xgb_model_path = base_path / 'models' / 'xgboost' / 'xgboost_model.pkl'
    xgb_metrics_path = base_path / 'models' / 'xgboost' / 'xgboost_metrics.csv'
    
    if check_file_exists(xgb_model_path) and check_file_exists(xgb_metrics_path):
        logger.info("ℹ️  Modèle XGBoost déjà entraîné")
        logger.info(f"   Modèle: {xgb_model_path}")
        logger.info(f"   Métriques: {xgb_metrics_path}")
        steps_completed += 1
    else:
        logger.info("🚀 Entraînement du XGBoost en cours...")
        logger.info("⏱️  Temps estimé: 10-15 minutes")
        
        success = run_command(
            [sys.executable, str(base_path / 'src' / 'models' / 'xgboost_model.py')],
            "Entraînement XGBoost",
            check_exit=False
        )
        
        if success and check_file_exists(xgb_model_path):
            logger.info("✅ XGBoost entraîné avec succès")
            steps_completed += 1
        else:
            logger.warning("⚠️  XGBoost non entraîné (peut continuer sans)")
            steps_failed += 1
    
    # ÉTAPE 4: Comparaison des modèles
    print_banner("📊 ÉTAPE 4/5 - COMPARAISON COMPLÈTE DES MODÈLES")
    
    logger.info("🚀 Génération des comparaisons et visualisations...")
    
    success = run_command(
        [sys.executable, str(base_path / 'scripts' / 'complete_model_comparison.py')],
        "Comparaison des modèles",
        check_exit=False
    )
    
    if success:
        logger.info("✅ Comparaison générée avec succès")
        steps_completed += 1
    else:
        logger.error("❌ Échec de la comparaison")
        steps_failed += 1
    
    # ÉTAPE 5: Rapport final
    print_banner("📄 ÉTAPE 5/5 - RAPPORT FINAL")
    
    results_dir = base_path / 'results' / 'model_comparison'
    report_path = results_dir / 'model_comparison_report.md'
    
    if check_file_exists(report_path):
        logger.info(f"✅ Rapport final généré: {report_path}")
        logger.info(f"\n📁 Tous les résultats dans: {results_dir}")
        
        # Lister les fichiers générés
        if results_dir.exists():
            logger.info("\n📊 Fichiers générés:")
            for file in sorted(results_dir.glob('*')):
                logger.info(f"   - {file.name}")
        
        steps_completed += 1
    else:
        logger.warning("⚠️  Rapport final non trouvé")
        steps_failed += 1
    
    # RÉSUMÉ FINAL
    elapsed_time = time.time() - start_time
    total_steps = 5
    
    print_banner("📊 RÉSUMÉ DE L'EXÉCUTION")
    
    logger.info(f"\n✅ Étapes complétées: {steps_completed}/{total_steps}")
    if steps_failed > 0:
        logger.info(f"⚠️  Étapes échouées: {steps_failed}/{total_steps}")
    logger.info(f"⏱️  Temps total: {elapsed_time/60:.1f} minutes")
    
    if steps_completed == total_steps:
        logger.info("\n🎉 PROJET COMPLÈTEMENT TERMINÉ!")
        logger.info(f"📁 Résultats disponibles dans: {results_dir}")
        logger.info("\n📊 Prochaines étapes:")
        logger.info("   1. Consulter le rapport: model_comparison_report.md")
        logger.info("   2. Visualiser les graphiques PNG")
        logger.info("   3. Analyser les métriques CSV")
    elif steps_completed >= 3:
        logger.info("\n✅ Projet majoritairement terminé")
        logger.info("⚠️  Certaines étapes DL peuvent avoir échoué (normal si TensorFlow pose problème)")
    else:
        logger.warning("\n⚠️  Projet incomplet")
        logger.info("💡 Vérifiez les erreurs ci-dessus et relancez")
    
    print_banner("✅ FIN DE L'ORCHESTRATION")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interruption utilisateur (Ctrl+C)")
        logger.info("💡 Vous pouvez relancer le script, il reprendra là où il s'est arrêté")
    except Exception as e:
        logger.error(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
