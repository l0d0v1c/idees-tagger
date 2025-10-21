#!/usr/bin/env python3
"""
Script de test pour le modèle ONNX mDeBERTa
Permet de valider que le modèle fonctionne avant l'intégration iOS
"""

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

def test_onnx_model():
    """Teste le modèle ONNX avec ONNX Runtime"""

    print("=" * 60)
    print("Test du modèle ONNX mDeBERTa")
    print("=" * 60)

    # Chemins des fichiers
    # Utiliser le modèle COMPLET (non-quantifié) pour de meilleurs résultats
    model_path = "./model.onnx"
    # model_path = "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7/onnx/model_quantized.onnx"  # Version quantifiée (moins précise)
    tokenizer_path = "./"

    print("\n1. Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✓ Tokenizer chargé: {tokenizer.__class__.__name__}")
    print(f"  Vocabulaire: {tokenizer.vocab_size} tokens")

    print("\n2. Chargement du modèle ONNX...")
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    print(f"✓ Modèle ONNX chargé")

    # Afficher les inputs/outputs
    print("\n3. Informations du modèle:")
    print("   Inputs:")
    for input_meta in session.get_inputs():
        print(f"     - {input_meta.name}: {input_meta.shape} ({input_meta.type})")

    print("   Outputs:")
    for output_meta in session.get_outputs():
        print(f"     - {output_meta.name}: {output_meta.shape} ({output_meta.type})")

    print("\n4. Test de classification zero-shot...")

    # Texte à classifier
    text = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
    candidate_labels = ["politics", "economy", "entertainment", "environment"]

    print(f"\n   Texte: \"{text}\"")
    print(f"   Labels candidats: {candidate_labels}")

    results = []

    for label in candidate_labels:
        # Créer l'hypothèse au format NLI
        # Le template par défaut du pipeline zero-shot est "This example is {}."
        hypothesis = f"This example is {label}."

        # Tokeniser
        inputs = tokenizer(
            text,
            hypothesis,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='np'
        )

        # Préparer les inputs pour ONNX Runtime
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }

        # Inférence
        outputs = session.run(None, ort_inputs)
        logits = outputs[0][0]

        # Extraire le logit d'entailment (on appliquera softmax plus tard sur tous les labels)
        # Pour CE modèle spécifique: [entailment, neutral, contradiction] (voir config.json)
        entailment_logit = logits[0]  # Index 0 = entailment pour ce modèle

        # Calculer aussi les probabilités pour affichage
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        results.append({
            'label': label,
            'entailment_logit': entailment_logit,  # <- Important pour le softmax final
            'logits': logits,
            'probs': probs
        })

    # Appliquer softmax sur les LOGITS d'entailment (comme le pipeline avec multi_label=False)
    # C'est ce que fait le pipeline zero-shot classification par défaut
    entailment_logits = np.array([r['entailment_logit'] for r in results])

    # Softmax sur les logits: exp(logit) / sum(exp(logits))
    exp_logits = np.exp(entailment_logits - entailment_logits.max())  # Soustraire max pour stabilité numérique
    normalized_scores = exp_logits / exp_logits.sum()

    # Mettre à jour les scores
    for i, result in enumerate(results):
        result['normalized_score'] = normalized_scores[i]

    # Trier par score normalisé décroissant
    results.sort(key=lambda x: x['normalized_score'], reverse=True)

    print("\n5. Résultats de classification (scores normalisés):")
    print("   " + "-" * 50)
    for i, result in enumerate(results, 1):
        bar_length = int(result['normalized_score'] * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {i}. {result['label']:15} {result['normalized_score']:.2%}  {bar}")

    print("\n6. Détails des probabilités NLI:")
    print("   " + "-" * 50)
    for result in results:
        print(f"\n   {result['label']}:")
        print(f"     Entailment:    {result['probs'][0]:.4f}  <- utilisé pour le score")
        print(f"     Neutral:       {result['probs'][1]:.4f}")
        print(f"     Contradiction: {result['probs'][2]:.4f}")

    print("\n" + "=" * 60)
    print("✓ Test terminé avec succès!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    try:
        test_onnx_model()
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
