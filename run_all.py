#!/usr/bin/env python3
"""
ROADMAP EXECUTION AUTOMATIQUE
===============================

Exécute toutes les étapes dans l'ordre optimal:
1. Classification adaptative
2. Entraînement LSTM
3. Évaluation complète
4. Génération rapports
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("="*80)
print("🚀 EXECUTION COMPLETE - SYSTEME PREDICTION EVENEMENTS EXTREMES")
print("="*80)

steps = [
    {
        'name': 'Classification Adaptative',
        'script': 'scripts/06_complete_implementation_PRO.py',
        'description': 'Classification 5 classes + ontologie + architecture LSTM',
        'duration_estimate': '2-3 min'
    },
    {
        'name': 'Entraînement LSTM',
        'script': 'scripts/07_train_lstm_FINAL.py',
        'description': 'Training Bidirectional LSTM + Focal Loss + évaluation',
        'duration_estimate': '30-60 min'
    }
]

print(f"\n📋 Plan d'exécution: {len(steps)} étapes\n")
for i, step in enumerate(steps, 1):
    print(f"{i}. {step['name']}")
    print(f"   📄 Script: {step['script']}")
    print(f"   📝 {step['description']}")
    print(f"   ⏱️  Durée: {step['duration_estimate']}")
    print()

print("="*80)
input("Appuyez sur ENTREE pour démarrer...")

results = []

for i, step in enumerate(steps, 1):
    print("\n" + "="*80)
    print(f"ETAPE {i}/{len(steps)}: {step['name'].upper()}")
    print("="*80)
    
    script_path = Path(step['script'])
    
    if not script_path.exists():
        print(f"❌ ERREUR: Script {script_path} introuvable!")
        results.append({
            'step': step['name'],
            'status': 'FAILED',
            'error': 'Script not found'
        })
        continue
    
    print(f"\n🚀 Lancement: {script_path}")
    print(f"⏱️  Durée estimée: {step['duration_estimate']}\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"\n✅ {step['name']} COMPLETE en {duration:.1f} min")
        
        results.append({
            'step': step['name'],
            'status': 'SUCCESS',
            'duration_minutes': duration
        })
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"\n❌ {step['name']} ECHOUE après {duration:.1f} min")
        print(f"   Exit code: {e.returncode}")
        
        results.append({
            'step': step['name'],
            'status': 'FAILED',
            'duration_minutes': duration,
            'exit_code': e.returncode
        })
        
        response = input("\nContinuer malgré l'erreur? (o/n): ")
        if response.lower() != 'o':
            print("\n❌ Exécution interrompue par utilisateur")
            break

print("\n" + "="*80)
print("RESUME EXECUTION")
print("="*80)

total_duration = sum(r.get('duration_minutes', 0) for r in results)

for r in results:
    symbol = "✅" if r['status'] == 'SUCCESS' else "❌"
    duration = f"{r.get('duration_minutes', 0):.1f} min"
    print(f"{symbol} {r['step']:30} - {r['status']:10} ({duration})")

print(f"\n⏱️  Durée totale: {total_duration:.1f} min")

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f"📊 Succès: {success_count}/{len(results)}")

if success_count == len(results):
    print("\n🎉 TOUS LES SCRIPTS EXECUTES AVEC SUCCES!")
else:
    print(f"\n⚠️  {len(results) - success_count} étape(s) échouée(s)")

print("\n" + "="*80)
print("✅ EXECUTION TERMINEE")
print("="*80)
