"""
ONTOLOGIE CLIMATIQUE - Ingénierie des connaissances

Système de connaissances pour événements climatiques extrêmes:
- Définition ontologie (concepts, relations, règles)
- Moteur d'inférence pour génération alertes automatiques
- Règles type: "IF Température > 42°C pendant 3 jours THEN Canicule niveau 4"

Conforme cahier des charges: "Ontologie climatique avec moteur de règles"
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

# ============================================================================
# ONTOLOGIE: CONCEPTS ET HIERARCHIE
# ============================================================================

class EventType(Enum):
    """Types d'événements climatiques"""
    NORMAL = 0
    CANICULE = 1
    VAGUE_FROID = 2
    SECHERESSE = 3
    TEMPETE = 4
    PRECIPITATION_INTENSE = 5

class SeverityLevel(Enum):
    """Niveaux de sévérité"""
    NORMAL = 0
    FAIBLE = 1
    MODERE = 2
    SEVERE = 3
    EXTREME = 4

class AlertLevel(Enum):
    """Niveaux d'alerte (Vigilance Météo France)"""
    VERT = 0    # Pas de vigilance particulière
    JAUNE = 1   # Soyez attentifs
    ORANGE = 2  # Soyez très vigilants
    ROUGE = 3   # Vigilance absolue

# ============================================================================
# ONTOLOGIE CLIMATIQUE (Knowledge Graph)
# ============================================================================

CLIMATE_ONTOLOGY = {
    "meta": {
        "version": "1.0",
        "created": "2025-01-01",
        "description": "Ontologie événements climatiques extrêmes",
        "author": "System"
    },
    
    "concepts": {
        "Canicule": {
            "description": "Période de températures très élevées",
            "parent": "EvenementClimatique",
            "aliases": ["Heatwave", "Chaleur_extreme", "Forte_chaleur"],
            "properties": {
                "temperature_seuil_min": 33,
                "temperature_seuil_severe": 37,
                "temperature_seuil_extreme": 42,
                "duree_min_heures": 72,
                "humidite_facteur": True
            },
            "impacts": [
                "Risque_sante_publique",
                "Surmortalite",
                "Deshydratation",
                "Incendies_foret",
                "Pics_consommation_energie"
            ],
            "populations_vulnerables": [
                "Personnes_agees",
                "Enfants",
                "Malades_chroniques",
                "Travailleurs_exterieurs"
            ]
        },
        
        "VagueFroid": {
            "description": "Période de températures très basses",
            "parent": "EvenementClimatique",
            "aliases": ["Cold_wave", "Grand_froid"],
            "properties": {
                "temperature_seuil_max": -5,
                "temperature_seuil_severe": -10,
                "temperature_seuil_extreme": -20,
                "duree_min_heures": 72,
                "vent_facteur": True,
                "wind_chill": True
            },
            "impacts": [
                "Hypothermie",
                "Gel_infrastructures",
                "Accidents_route",
                "Pics_consommation_energie"
            ],
            "populations_vulnerables": [
                "Sans_abri",
                "Personnes_isolees",
                "Enfants"
            ]
        },
        
        "Secheresse": {
            "description": "Déficit prolongé en précipitations",
            "parent": "EvenementClimatique",
            "aliases": ["Drought"],
            "properties": {
                "precipitation_seuil": 2.5,  # mm/jour
                "duree_min_jours": 30,
                "evapotranspiration_facteur": True
            },
            "impacts": [
                "Restrictions_eau",
                "Pertes_agricoles",
                "Incendies_foret",
                "Ecosystemes_fragilises"
            ]
        },
        
        "PrecipitationIntense": {
            "description": "Pluies très importantes en courte durée",
            "parent": "EvenementClimatique",
            "aliases": ["Heavy_rain", "Pluie_diluvienne"],
            "properties": {
                "precipitation_seuil_1h": 40,  # mm en 1h
                "precipitation_seuil_24h": 100,  # mm en 24h
                "duree_min_heures": 1
            },
            "impacts": [
                "Inondations",
                "Glissements_terrain",
                "Debordements_cours_eau",
                "Perturbations_transports"
            ]
        }
    },
    
    "relations": {
        "precede": {
            "description": "Un événement précède un autre",
            "examples": [
                ("Secheresse", "Canicule"),
                ("Canicule", "Incendie_foret")
            ]
        },
        "aggrave": {
            "description": "Un événement aggrave un autre",
            "examples": [
                ("Vent_fort", "Canicule"),  # Vent chaud
                ("Humidite_haute", "Canicule"),  # Sensation chaleur
                ("Vent_fort", "VagueFroid")  # Wind chill
            ]
        },
        "favorise": {
            "description": "Un événement favorise l'apparition d'un autre",
            "examples": [
                ("Secheresse", "Incendie"),
                ("Chaleur", "Orage")
            ]
        }
    }
}

# ============================================================================
# REGLES D'INFERENCE
# ============================================================================

class ClimateRule:
    """
    Règle d'inférence pour détection événements
    
    Format: IF conditions THEN conclusion WITH confidence
    """
    
    def __init__(self, 
                 name: str,
                 conditions: List[Tuple[str, str, float]],
                 conclusion: Dict,
                 confidence: float = 1.0,
                 description: str = ""):
        """
        Args:
            name: Nom de la règle
            conditions: Liste (feature, operator, value)
                operators: '>', '<', '>=', '<=', '==', 'between'
            conclusion: Dict avec 'event_type', 'severity', 'alert_level'
            confidence: Confiance dans la règle (0-1)
            description: Description humaine
        """
        self.name = name
        self.conditions = conditions
        self.conclusion = conclusion
        self.confidence = confidence
        self.description = description
    
    def evaluate(self, data: Dict) -> Tuple[bool, float]:
        """
        Évalue si règle s'applique aux données
        
        Returns:
            (is_applicable, confidence)
        """
        for feature, operator, value in self.conditions:
            if feature not in data:
                return False, 0.0
            
            feature_value = data[feature]
            
            # Évaluer condition
            if operator == '>':
                if not feature_value > value:
                    return False, 0.0
            elif operator == '<':
                if not feature_value < value:
                    return False, 0.0
            elif operator == '>=':
                if not feature_value >= value:
                    return False, 0.0
            elif operator == '<=':
                if not feature_value <= value:
                    return False, 0.0
            elif operator == '==':
                if not feature_value == value:
                    return False, 0.0
            elif operator == 'between':
                if not (value[0] <= feature_value <= value[1]):
                    return False, 0.0
        
        # Toutes conditions satisfaites
        return True, self.confidence
    
    def to_dict(self) -> Dict:
        """Sérialisation"""
        return {
            'name': self.name,
            'conditions': self.conditions,
            'conclusion': {
                'event_type': self.conclusion['event_type'].name,
                'severity': self.conclusion['severity'].name,
                'alert_level': self.conclusion['alert_level'].name
            },
            'confidence': self.confidence,
            'description': self.description
        }

# ============================================================================
# BASE DE REGLES
# ============================================================================

CLIMATE_RULES = [
    # ===== CANICULES =====
    ClimateRule(
        name="CANICULE_EXTREME",
        conditions=[
            ('temp_rolling_48h', '>=', 42),
        ],
        conclusion={
            'event_type': EventType.CANICULE,
            'severity': SeverityLevel.EXTREME,
            'alert_level': AlertLevel.ROUGE
        },
        confidence=1.0,
        description="Canicule extrême: température >= 42°C sur 48h"
    ),
    
    ClimateRule(
        name="CANICULE_SEVERE",
        conditions=[
            ('temp_rolling_48h', '>=', 37),
            ('temp_rolling_48h', '<', 42),
        ],
        conclusion={
            'event_type': EventType.CANICULE,
            'severity': SeverityLevel.SEVERE,
            'alert_level': AlertLevel.ORANGE
        },
        confidence=0.95,
        description="Canicule sévère: température 37-42°C sur 48h"
    ),
    
    ClimateRule(
        name="CANICULE_MODERATE",
        conditions=[
            ('temp_rolling_48h', '>=', 33),
            ('temp_rolling_48h', '<', 37),
        ],
        conclusion={
            'event_type': EventType.CANICULE,
            'severity': SeverityLevel.MODERE,
            'alert_level': AlertLevel.JAUNE
        },
        confidence=0.90,
        description="Canicule modérée: température 33-37°C sur 48h"
    ),
    
    ClimateRule(
        name="CANICULE_FORTE_CHALEUR",
        conditions=[
            ('temp_rolling_48h', '>=', 28),
            ('temp_rolling_48h', '<', 33),
        ],
        conclusion={
            'event_type': EventType.CANICULE,
            'severity': SeverityLevel.FAIBLE,
            'alert_level': AlertLevel.VERT
        },
        confidence=0.80,
        description="Forte chaleur: température 28-33°C sur 48h"
    ),
    
    # ===== VAGUES DE FROID =====
    ClimateRule(
        name="FROID_EXTREME",
        conditions=[
            ('temp_rolling_48h', '<=', -20),
        ],
        conclusion={
            'event_type': EventType.VAGUE_FROID,
            'severity': SeverityLevel.EXTREME,
            'alert_level': AlertLevel.ROUGE
        },
        confidence=1.0,
        description="Froid extrême: température <= -20°C sur 48h"
    ),
    
    ClimateRule(
        name="FROID_SEVERE",
        conditions=[
            ('temp_rolling_48h', '<=', -10),
            ('temp_rolling_48h', '>', -20),
        ],
        conclusion={
            'event_type': EventType.VAGUE_FROID,
            'severity': SeverityLevel.SEVERE,
            'alert_level': AlertLevel.ORANGE
        },
        confidence=0.95,
        description="Froid sévère: température -10 à -20°C sur 48h"
    ),
    
    ClimateRule(
        name="FROID_MODERATE",
        conditions=[
            ('temp_rolling_48h', '<=', -5),
            ('temp_rolling_48h', '>', -10),
        ],
        conclusion={
            'event_type': EventType.VAGUE_FROID,
            'severity': SeverityLevel.MODERE,
            'alert_level': AlertLevel.JAUNE
        },
        confidence=0.90,
        description="Froid modéré: température -5 à -10°C sur 48h"
    ),
    
    ClimateRule(
        name="FROID_GEL_PROLONGE",
        conditions=[
            ('temp_rolling_48h', '<=', 2),
            ('temp_rolling_48h', '>', -5),
        ],
        conclusion={
            'event_type': EventType.VAGUE_FROID,
            'severity': SeverityLevel.FAIBLE,
            'alert_level': AlertLevel.VERT
        },
        confidence=0.80,
        description="Gel prolongé: température 0-2°C sur 48h"
    ),
]

# ============================================================================
# MOTEUR D'INFERENCE
# ============================================================================

class InferenceEngine:
    """
    Moteur d'inférence pour alertes climatiques
    
    Applique règles ontologie pour détecter événements et générer alertes
    """
    
    def __init__(self, rules: List[ClimateRule], ontology: Dict):
        self.rules = rules
        self.ontology = ontology
    
    def infer(self, data: Dict) -> List[Dict]:
        """
        Applique règles et retourne alertes déclenchées
        
        Args:
            data: Dict avec features (temperature, temp_rolling_48h, etc.)
        
        Returns:
            Liste alertes [{rule, conclusion, confidence}, ...]
        """
        alerts = []
        
        for rule in self.rules:
            is_applicable, confidence = rule.evaluate(data)
            
            if is_applicable:
                alert = {
                    'rule_name': rule.name,
                    'description': rule.description,
                    'event_type': rule.conclusion['event_type'].name,
                    'severity': rule.conclusion['severity'].name,
                    'alert_level': rule.conclusion['alert_level'].name,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Trier par sévérité (plus sévère en premier)
        severity_order = {
            'EXTREME': 4,
            'SEVERE': 3,
            'MODERE': 2,
            'FAIBLE': 1,
            'NORMAL': 0
        }
        
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
        
        return alerts
    
    def infer_batch(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Applique inférence sur DataFrame entier
        
        Returns:
            DataFrame avec colonnes alertes ajoutées
        """
        results = []
        
        for idx, row in dataframe.iterrows():
            data = row.to_dict()
            alerts = self.infer(data)
            
            # Prendre alerte la plus sévère
            if alerts:
                top_alert = alerts[0]
                results.append({
                    'event_inferred': top_alert['event_type'],
                    'severity_inferred': top_alert['severity'],
                    'alert_level_inferred': top_alert['alert_level'],
                    'confidence_inferred': top_alert['confidence'],
                    'rule_triggered': top_alert['rule_name']
                })
            else:
                results.append({
                    'event_inferred': 'NORMAL',
                    'severity_inferred': 'NORMAL',
                    'alert_level_inferred': 'VERT',
                    'confidence_inferred': 1.0,
                    'rule_triggered': 'NONE'
                })
        
        results_df = pd.DataFrame(results)
        return pd.concat([dataframe, results_df], axis=1)
    
    def get_recommendations(self, alert: Dict) -> List[str]:
        """
        Génère recommandations basées sur alerte
        
        Args:
            alert: Dict alerte
        
        Returns:
            Liste recommandations
        """
        event_type = alert['event_type']
        severity = alert['severity']
        
        # Récupérer concept ontologie
        concept = None
        for concept_name, concept_data in self.ontology['concepts'].items():
            if concept_name.upper() == event_type:
                concept = concept_data
                break
        
        if not concept:
            return ["Pas de recommandations disponibles"]
        
        # Recommandations générales
        recs = []
        
        if event_type == 'CANICULE':
            recs = [
                "Restez hydraté: buvez régulièrement de l'eau",
                "Évitez exposition soleil aux heures chaudes (11h-16h)",
                "Restez dans lieux climatisés ou frais",
                "Prenez nouvelles personnes vulnérables (âgées, enfants)",
                "Ne laissez personne dans véhicule fermé",
            ]
            
            if severity in ['SEVERE', 'EXTREME']:
                recs.extend([
                    "ALERTE: Risque vital pour personnes vulnérables",
                    "Évitez activités physiques intenses",
                    "Consultez médecin si symptômes (malaise, crampes)",
                ])
        
        elif event_type == 'VAGUE_FROID':
            recs = [
                "Couvrez-vous bien, portez plusieurs couches vêtements",
                "Limitez exposition au froid",
                "Chauffez logement correctement (19°C recommandé)",
                "Attention aux sans-abri et personnes isolées",
                "Vérifiez état chauffage et isolations",
            ]
            
            if severity in ['SEVERE', 'EXTREME']:
                recs.extend([
                    "ALERTE: Risque hypothermie",
                    "Évitez déplacements non essentiels",
                    "Anticipez panne électrique (chauffage d'appoint)",
                ])
        
        return recs

# ============================================================================
# SAUVEGARDE ONTOLOGIE
# ============================================================================

def save_ontology_and_rules():
    """Sauvegarde ontologie et règles en JSON"""
    
    output_dir = Path('knowledge_base')
    output_dir.mkdir(exist_ok=True)
    
    # Ontologie
    with open(output_dir / 'climate_ontology.json', 'w', encoding='utf-8') as f:
        json.dump(CLIMATE_ONTOLOGY, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Ontologie sauvegardée: {output_dir / 'climate_ontology.json'}")
    
    # Règles
    rules_dict = {
        'meta': {
            'version': '1.0',
            'num_rules': len(CLIMATE_RULES),
            'created': datetime.now().isoformat()
        },
        'rules': [rule.to_dict() for rule in CLIMATE_RULES]
    }
    
    with open(output_dir / 'climate_rules.json', 'w', encoding='utf-8') as f:
        json.dump(rules_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Règles sauvegardées: {output_dir / 'climate_rules.json'}")
    
    return output_dir

# ============================================================================
# TEST MOTEUR D'INFERENCE
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("ONTOLOGIE CLIMATIQUE + MOTEUR D'INFERENCE")
    print("="*80)
    
    # Sauvegarder
    print("\n1. Sauvegarde ontologie et règles...")
    kb_dir = save_ontology_and_rules()
    
    print(f"\n📚 Knowledge Base créée:")
    print(f"   - Ontologie: {len(CLIMATE_ONTOLOGY['concepts'])} concepts")
    print(f"   - Règles: {len(CLIMATE_RULES)} règles d'inférence")
    
    # Créer moteur
    print("\n2. Initialisation moteur d'inférence...")
    engine = InferenceEngine(CLIMATE_RULES, CLIMATE_ONTOLOGY)
    print("✅ Moteur initialisé")
    
    # Tests
    print("\n" + "="*80)
    print("TESTS MOTEUR D'INFERENCE")
    print("="*80)
    
    test_cases = [
        {
            'name': 'Canicule extrême',
            'data': {'temperature': 45, 'temp_rolling_48h': 43}
        },
        {
            'name': 'Canicule modérée',
            'data': {'temperature': 35, 'temp_rolling_48h': 34}
        },
        {
            'name': 'Froid extrême',
            'data': {'temperature': -25, 'temp_rolling_48h': -22}
        },
        {
            'name': 'Temps normal',
            'data': {'temperature': 20, 'temp_rolling_48h': 19}
        },
    ]
    
    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {test['name']}")
        print(f"Données: {test['data']}")
        print(f"{'='*80}")
        
        alerts = engine.infer(test['data'])
        
        if alerts:
            print(f"\n🚨 {len(alerts)} alerte(s) déclenchée(s):")
            for i, alert in enumerate(alerts, 1):
                print(f"\n   Alerte {i}:")
                print(f"      Règle: {alert['rule_name']}")
                print(f"      Description: {alert['description']}")
                print(f"      Événement: {alert['event_type']}")
                print(f"      Sévérité: {alert['severity']}")
                print(f"      Niveau alerte: {alert['alert_level']}")
                print(f"      Confiance: {alert['confidence']:.2f}")
                
                # Recommandations
                recs = engine.get_recommendations(alert)
                print(f"\n      📋 Recommandations:")
                for rec in recs[:3]:  # Top 3
                    print(f"         - {rec}")
        else:
            print("\n✅ Aucune alerte (conditions normales)")
    
    # Test sur données réelles
    print("\n" + "="*80)
    print("APPLICATION SUR DONNEES REELLES")
    print("="*80)
    
    classified_path = Path('data/processed/splits_classified/train_classified.parquet')
    if classified_path.exists():
        print("\nChargement données classifiées...")
        df = pd.read_parquet(classified_path)
        
        # Échantillon
        sample = df.head(1000)
        print(f"Traitement échantillon: {len(sample)} lignes")
        
        # Inférence
        print("Application moteur d'inférence...")
        df_inferred = engine.infer_batch(sample)
        
        # Statistiques
        print("\n📊 Résultats inférence:")
        print(f"\nÉvénements inférés:")
        print(df_inferred['event_inferred'].value_counts())
        
        print(f"\nNiveaux alerte:")
        print(df_inferred['alert_level_inferred'].value_counts())
        
        # Sauvegarder échantillon
        output_sample = kb_dir / 'inference_sample.parquet'
        df_inferred.to_parquet(output_sample, index=False)
        print(f"\n✅ Échantillon inféré sauvegardé: {output_sample}")
    else:
        print("\n⚠️  Données classifiées non disponibles")
        print("   Exécutez d'abord: python scripts/01_create_extreme_events_classification_v2.py")
    
    print("\n" + "="*80)
    print("✅ ONTOLOGIE + MOTEUR D'INFERENCE CREES!")
    print("="*80)
    print("\nProchaines étapes:")
    print("   1. Entraîner LSTM classification")
    print("   2. Intégrer inférence avec prédictions DL")
    print("   3. Créer API pour alertes temps réel")
    print("   4. Interface Web avec visualisation alertes")
    
    print("\n" + "="*80)
