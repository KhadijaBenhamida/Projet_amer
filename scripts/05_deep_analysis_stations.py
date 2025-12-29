"""
ANALYSE APPROFONDIE EVENEMENTS EXTREMES PAR STATION
Identification des événements réels dans les données 2015-2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("ANALYSE APPROFONDIE EVENEMENTS EXTREMES PAR STATION")
print("="*80)

# Configuration stations avec événements typiques attendus
STATIONS_INFO = {
    722020: {
        "name": "JFK",
        "city": "New York",
        "zone": "Humid Continental",
        "expected_extremes": [
            "Canicules estivales (>35°C)",
            "Vagues de froid hivernales (<-10°C)",
            "Blizzards/tempêtes de neige",
            "Orages violents"
        ]
    },
    722590: {
        "name": "ORD",
        "city": "Chicago",
        "zone": "Continental",
        "expected_extremes": [
            "Froid polaire hivernal (<-20°C)",
            "Canicules humides (>35°C)",
            "Tornades (printemps)",
            "Blizzards"
        ]
    },
    722780: {
        "name": "MIA",
        "city": "Miami",
        "zone": "Tropical",
        "expected_extremes": [
            "Canicules tropicales (>35°C)",
            "Humidité extrême",
            "Ouragans (vent >120 km/h)",
            "Pas de froid (<10°C rare)"
        ]
    },
    722950: {
        "name": "PHX",
        "city": "Phoenix",
        "zone": "Desert",
        "expected_extremes": [
            "Canicules extrêmes (>45°C)",
            "Tempêtes de poussière (haboob)",
            "Sécheresse prolongée",
            "Amplitudes thermiques jour/nuit"
        ]
    },
    725300: {
        "name": "DFW",
        "city": "Dallas",
        "zone": "Humid Subtropical",
        "expected_extremes": [
            "Canicules (>38°C)",
            "Tornades violentes",
            "Gel hivernal rare mais sévère",
            "Orages de grêle"
        ]
    },
    725650: {
        "name": "DEN",
        "city": "Denver",
        "zone": "Semi-arid",
        "expected_extremes": [
            "Variations température extrêmes (20°C en 24h)",
            "Froid sec (<-20°C)",
            "Neige abondante",
            "Blizzards"
        ]
    },
    727930: {
        "name": "LAX",
        "city": "Los Angeles",
        "zone": "Mediterranean",
        "expected_extremes": [
            "Vagues de chaleur (>35°C)",
            "Sécheresse",
            "Vents Santa Ana (incendies)",
            "Températures douces (peu d'extrêmes)"
        ]
    },
    744860: {
        "name": "SEA",
        "city": "Seattle",
        "zone": "Oceanic",
        "expected_extremes": [
            "Pluie persistante",
            "Vent maritime fort",
            "Neige rare mais paralysante",
            "Températures modérées (10-25°C)"
        ]
    }
}

# Charger données
print("\n1. Chargement données complètes...")
df = pd.read_parquet('data/processed/splits/train.parquet')
print(f"   Total: {len(df):,} samples (2015-2024)")
print(f"   Stations: {df['station_id'].nunique()}")

# ============================================================================
# ANALYSE DETAILLEE PAR STATION
# ============================================================================

print("\n" + "="*80)
print("ANALYSE DETAILLEE PAR STATION")
print("="*80)

results = {}

for station_id in sorted(df['station_id'].unique()):
    station_data = df[df['station_id'] == station_id].copy()
    info = STATIONS_INFO.get(station_id, {})
    
    print(f"\n{'='*80}")
    print(f"{info.get('name', station_id)} - {info.get('city', '')} ({info.get('zone', '')})")
    print(f"{'='*80}")
    print(f"Échantillons: {len(station_data):,}")
    
    # Statistiques température
    print(f"\n📊 TEMPERATURE:")
    print(f"   Min:    {station_data['temperature'].min():7.1f}°C")
    print(f"   P01:    {station_data['temperature'].quantile(0.01):7.1f}°C")
    print(f"   P05:    {station_data['temperature'].quantile(0.05):7.1f}°C")
    print(f"   Moyenne:{station_data['temperature'].mean():7.1f}°C")
    print(f"   P95:    {station_data['temperature'].quantile(0.95):7.1f}°C")
    print(f"   P99:    {station_data['temperature'].quantile(0.99):7.1f}°C")
    print(f"   Max:    {station_data['temperature'].max():7.1f}°C")
    print(f"   Std:    {station_data['temperature'].std():7.1f}°C")
    
    # Amplitudes
    daily_range = station_data.groupby(['year', 'month', 'day'])['temperature'].agg(['min', 'max'])
    daily_range['amplitude'] = daily_range['max'] - daily_range['min']
    print(f"\n   Amplitude jour/nuit:")
    print(f"      Moyenne: {daily_range['amplitude'].mean():.1f}°C")
    print(f"      Max:     {daily_range['amplitude'].max():.1f}°C")
    
    # Événements extrêmes CHAUDS
    print(f"\n🔥 CHALEUR EXTREME:")
    hot_35 = (station_data['temperature'] >= 35).sum()
    hot_40 = (station_data['temperature'] >= 40).sum()
    hot_45 = (station_data['temperature'] >= 45).sum()
    print(f"   T >= 35°C: {hot_35:6,} heures ({hot_35/len(station_data)*100:5.2f}%)")
    print(f"   T >= 40°C: {hot_40:6,} heures ({hot_40/len(station_data)*100:5.2f}%)")
    print(f"   T >= 45°C: {hot_45:6,} heures ({hot_45/len(station_data)*100:5.2f}%)")
    
    if hot_40 > 0:
        print(f"   📌 Températures >40°C:")
        extreme_hot = station_data[station_data['temperature'] >= 40].nsmallest(5, 'temperature')
        for _, row in extreme_hot.iterrows():
            print(f"      {row['year']}-{row['month']:02d}-{row['day']:02d}: {row['temperature']:.1f}°C")
    
    # Événements extrêmes FROIDS
    print(f"\n❄️  FROID EXTREME:")
    cold_0 = (station_data['temperature'] <= 0).sum()
    cold_minus10 = (station_data['temperature'] <= -10).sum()
    cold_minus20 = (station_data['temperature'] <= -20).sum()
    print(f"   T <= 0°C:   {cold_0:6,} heures ({cold_0/len(station_data)*100:5.2f}%)")
    print(f"   T <= -10°C: {cold_minus10:6,} heures ({cold_minus10/len(station_data)*100:5.2f}%)")
    print(f"   T <= -20°C: {cold_minus20:6,} heures ({cold_minus20/len(station_data)*100:5.2f}%)")
    
    if cold_minus10 > 0:
        print(f"   📌 Températures <-10°C:")
        extreme_cold = station_data[station_data['temperature'] <= -10].nlargest(5, 'temperature')
        for _, row in extreme_cold.iterrows():
            print(f"      {row['year']}-{row['month']:02d}-{row['day']:02d}: {row['temperature']:.1f}°C")
    
    # Vent (si disponible)
    if 'wind_speed' in station_data.columns:
        print(f"\n💨 VENT:")
        print(f"   Moyenne: {station_data['wind_speed'].mean():5.1f} km/h")
        print(f"   P95:     {station_data['wind_speed'].quantile(0.95):5.1f} km/h")
        print(f"   P99:     {station_data['wind_speed'].quantile(0.99):5.1f} km/h")
        print(f"   Max:     {station_data['wind_speed'].max():5.1f} km/h")
        
        wind_strong = (station_data['wind_speed'] >= 60).sum()
        wind_violent = (station_data['wind_speed'] >= 90).sum()
        print(f"   Vent fort (>=60 km/h):   {wind_strong:6,} heures ({wind_strong/len(station_data)*100:5.2f}%)")
        print(f"   Vent violent (>=90 km/h): {wind_violent:6,} heures ({wind_violent/len(station_data)*100:5.2f}%)")
    
    # Événements attendus vs réels
    print(f"\n✅ EVENEMENTS ATTENDUS (zone {info.get('zone', '')}):")
    for event in info.get('expected_extremes', []):
        print(f"   - {event}")
    
    # Sauvegarder résultats
    results[station_id] = {
        'name': info.get('name', str(station_id)),
        'zone': info.get('zone', 'Unknown'),
        'samples': len(station_data),
        'temp_min': float(station_data['temperature'].min()),
        'temp_max': float(station_data['temperature'].max()),
        'temp_mean': float(station_data['temperature'].mean()),
        'temp_std': float(station_data['temperature'].std()),
        'temp_p01': float(station_data['temperature'].quantile(0.01)),
        'temp_p05': float(station_data['temperature'].quantile(0.05)),
        'temp_p95': float(station_data['temperature'].quantile(0.95)),
        'temp_p99': float(station_data['temperature'].quantile(0.99)),
        'amplitude_mean': float(daily_range['amplitude'].mean()),
        'amplitude_max': float(daily_range['amplitude'].max()),
        'hot_35_pct': float(hot_35/len(station_data)*100),
        'hot_40_pct': float(hot_40/len(station_data)*100),
        'cold_0_pct': float(cold_0/len(station_data)*100),
        'cold_minus10_pct': float(cold_minus10/len(station_data)*100),
    }

# ============================================================================
# COMPARAISON ENTRE STATIONS
# ============================================================================

print("\n" + "="*80)
print("COMPARAISON ENTRE STATIONS")
print("="*80)

df_results = pd.DataFrame(results).T

print("\n🔥 CANICULES (% heures T >= 40°C):")
print(df_results[['name', 'hot_40_pct']].sort_values('hot_40_pct', ascending=False).to_string())

print("\n❄️  FROID EXTREME (% heures T <= -10°C):")
print(df_results[['name', 'cold_minus10_pct']].sort_values('cold_minus10_pct', ascending=False).to_string())

print("\n🌡️  AMPLITUDE THERMIQUE (moyenne °C/jour):")
print(df_results[['name', 'amplitude_mean']].sort_values('amplitude_mean', ascending=False).to_string())

# ============================================================================
# RECOMMANDATIONS CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("🎯 RECOMMANDATIONS CLASSIFICATION")
print("="*80)

print("\n1. SEUILS ADAPTATIFS PAR STATION (percentiles):")
print("   ✅ P99 (top 1%) pour canicule extrême")
print("   ✅ P95 (top 5%) pour forte chaleur")
print("   ✅ P01 (bottom 1%) pour froid extrême")
print("   ✅ P05 (bottom 5%) pour froid prolongé")

print("\n2. CLASSES A CREER:")
print("   0 = Normal (85-90%)")
print("   1 = Canicule extrême (T > P99)")
print("   2 = Forte chaleur (P95 < T <= P99)")
print("   3 = Froid extrême (T < P01)")
print("   4 = Froid prolongé (P01 <= T < P05)")
print("   5 = Tempête (vent > P99) - stations côtières")

print("\n3. SPECIFICITES PAR STATION:")
for sid, res in results.items():
    name = res['name']
    zone = res['zone']
    
    events = []
    if res['hot_40_pct'] > 1.0:
        events.append(f"Canicules fréquentes (>40°C: {res['hot_40_pct']:.1f}%)")
    if res['cold_minus10_pct'] > 5.0:
        events.append(f"Froid sévère (<-10°C: {res['cold_minus10_pct']:.1f}%)")
    if res['amplitude_max'] > 25:
        events.append(f"Chocs thermiques (amplitude max {res['amplitude_max']:.0f}°C)")
    
    if events:
        print(f"\n   {name} ({zone}):")
        for e in events:
            print(f"      - {e}")

# Sauvegarder analyse
import json
Path('models/analysis').mkdir(parents=True, exist_ok=True)

with open('models/analysis/extreme_events_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("✅ ANALYSE TERMINEE")
print("="*80)
print("\nRésultats sauvegardés: models/analysis/extreme_events_analysis.json")
print("\n🚀 Prochaine étape: Adapter classification avec ces insights")
