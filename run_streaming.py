"""
Script d'exécution automatique du pipeline streaming
Lance automatiquement le producer et le consumer
"""
import subprocess
import time
import sys
from pathlib import Path

def main():
    print("\n🚀 === PIPELINE AUTOMATIQUE ===\n")
    
    # 1. Lancer le producer en arrière-plan
    print("📤 Démarrage du Producer...")
    producer_process = subprocess.Popen(
        [sys.executable, "src/streaming/kafka_producer.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Attendre que le producer envoie quelques messages
    time.sleep(5)
    print("✅ Producer actif\n")
    
    # 2. Lancer le consumer (affiche les résultats)
    print("📥 Démarrage du Consumer...\n")
    print("═" * 60)
    
    try:
        subprocess.run(
            [sys.executable, "src/streaming/demo_consumer.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
    finally:
        # Arrêter le producer
        producer_process.terminate()
        producer_process.wait()
    
    print("═" * 60)
    print("\n✅ PIPELINE TERMINÉ!\n")

if __name__ == "__main__":
    main()
