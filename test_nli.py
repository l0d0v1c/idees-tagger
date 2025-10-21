#!/usr/bin/env python3
"""
Script de test pour le NLI (Natural Language Inference)
Teste directement les capacités d'inférence du modèle mDeBERTa
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def test_nli():
    """Teste le modèle NLI avec des paires premise-hypothesis"""

    print("=" * 60)
    print("Test NLI - Natural Language Inference")
    print("=" * 60)

    # Configuration
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\n1. Device: {device}")

    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    print(f"\n2. Chargement du modèle: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    print("✓ Modèle chargé")

    # Test 1: Premise en allemand, Hypothesis en anglais
    print("\n" + "=" * 60)
    print("Test 1: Multilingue (DE → EN)")
    print("=" * 60)

    premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
    hypothesis = "Emmanuel Macron is the President of France"

    print(f"\nPremise:    {premise}")
    print(f"Hypothesis: {hypothesis}")

    input_data = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input_data["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction_dict = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}

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

    input_data = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input_data["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    prediction_dict = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}

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

    input_data = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input_data["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    prediction_dict = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}

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

    input_data = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input_data["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    prediction_dict = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}

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
