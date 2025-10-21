#!/usr/bin/env python3
"""
Script de test pour le NLI (Natural Language Inference)
Teste directement les capacités d'inférence du modèle mDeBERTa avec ONNX Runtime
"""

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

def test_nli():
    """Teste le modèle NLI avec des paires premise-hypothesis"""

    print("=" * 60)
    print("Test NLI - Natural Language Inference (ONNX)")
    print("=" * 60)

    # Chemins des fichiers locaux
    model_path = "./model.onnx"
    tokenizer_path = "./"

    print(f"\n1. Chargement du tokenizer depuis {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✓ Tokenizer chargé: {tokenizer.__class__.__name__}")

    print(f"\n2. Chargement du modèle ONNX depuis {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    print("✓ Modèle ONNX chargé")

    # Fonction helper pour faire l'inférence
    def predict_nli(premise, hypothesis):
        """Prédit la relation NLI entre premise et hypothesis"""
        # Tokeniser
        inputs = tokenizer(
            premise,
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

        # Appliquer softmax pour obtenir les probabilités
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Labels: entailment (0), neutral (1), contradiction (2)
        label_names = ["entailment", "neutral", "contradiction"]
        prediction_dict = {name: round(float(prob) * 100, 1) for prob, name in zip(probs, label_names)}

        return prediction_dict

    # Test 1: Premise en allemand, Hypothesis en anglais
    print("\n" + "=" * 60)
    print("Test 1: Multilingue (DE → EN)")
    print("=" * 60)

    premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
    hypothesis = "Emmanuel Macron is the President of France"

    print(f"\nPremise:    {premise}")
    print(f"Hypothesis: {hypothesis}")

    prediction_dict = predict_nli(premise, hypothesis)

    print("\nRésultats:")
    for label, score in prediction_dict.items():
        bar_length = int(score / 2.5)  # Scale to 40 chars max
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {label:15} {score:5.1f}%  {bar}")

    # Test 2: Entailment (implication vraie)
    print("\n" + "=" * 60)
    print("Test 2: Entailment - Implication vraie")
    print("=" * 60)

    premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
    hypothesis = "Angela Merkel is a politician"

    print(f"\nPremise:    {premise}")
    print(f"Hypothesis: {hypothesis}")

    prediction_dict = predict_nli(premise, hypothesis)

    print("\nRésultats:")
    for label, score in prediction_dict.items():
        bar_length = int(score / 2.5)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {label:15} {score:5.1f}%  {bar}")

    # Test 3: Contradiction
    print("\n" + "=" * 60)
    print("Test 3: Contradiction - Contradiction évidente")
    print("=" * 60)

    premise = "Le soleil brille et il fait beau"
    hypothesis = "Il pleut et il fait sombre"

    print(f"\nPremise:    {premise}")
    print(f"Hypothesis: {hypothesis}")

    prediction_dict = predict_nli(premise, hypothesis)

    print("\nRésultats:")
    for label, score in prediction_dict.items():
        bar_length = int(score / 2.5)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {label:15} {score:5.1f}%  {bar}")

    # Test 4: Neutral
    print("\n" + "=" * 60)
    print("Test 4: Neutral - Pas de relation claire")
    print("=" * 60)

    premise = "J'aime manger des pommes"
    hypothesis = "La tour Eiffel est à Paris"

    print(f"\nPremise:    {premise}")
    print(f"Hypothesis: {hypothesis}")

    prediction_dict = predict_nli(premise, hypothesis)

    print("\nRésultats:")
    for label, score in prediction_dict.items():
        bar_length = int(score / 2.5)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {label:15} {score:5.1f}%  {bar}")

    print("\n" + "=" * 60)
    print("✓ Tests NLI terminés avec succès!")
    print("=" * 60)
    print("\nNote: Ce modèle NLI est utilisé pour la classification zero-shot")
    print("en testant l'entailment entre le texte et chaque label candidat.")


if __name__ == "__main__":
    try:
        test_nli()
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
