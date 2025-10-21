# Classification Zero-Shot mDeBERTa pour iOS

Ce répertoire contient les fichiers et scripts pour utiliser le modèle de classification zero-shot `mDeBERTa-v3-base-mnli-xnli` sur iOS.

## 📦 Contenu du répertoire

**Modèle et tokenizer** (dans le répertoire racine):
- `model.onnx` - Modèle ONNX complet (1.1 GB) ⭐ RECOMMANDÉ
- `config.json` - Configuration du modèle
- `spm.model` - Tokenizer SentencePiece
- `tokenizer_config.json`, `special_tokens_map.json`, etc.

**Code et documentation**:
- `TextClassifier.swift` - Classe Swift pour classification zero-shot iOS
- `NLITester.swift` - Classe Swift pour tests NLI directs
- `test_model.py` - Script Python pour tester la classification zero-shot
- `test_nli.py` - Script Python pour tester le NLI directement
- `README.md` - Ce fichier
- `SETUP_IOS.md` - Guide complet d'intégration iOS
- `SOLUTION_FINALE.md` - Explication détaillée de la solution

## 🚀 Démarrage rapide

### 1. Tester le modèle en Python

**Classification zero-shot:**
```bash
python3 test_model.py
```

**Résultat attendu**:
```
1. politics        99.00%  ✓
2. economy          0.55%
3. entertainment    0.42%
4. environment      0.04%
```

**Test NLI direct (Natural Language Inference):**
```bash
python3 test_nli.py
```

**Résultats attendus:**
- Test 1 (Multilingue): 82.4% entailment
- Test 2 (Entailment): 99.9% entailment ✓
- Test 3 (Contradiction): 99.5% contradiction ✓
- Test 4 (Neutral): 99.8% neutral ✓

### 2. Intégrer dans une app iOS

#### Étape 1: Ajouter les dépendances

**Avec CocoaPods:**
```ruby
pod 'onnxruntime-objc', '~> 1.19.2'
pod 'SentencePiece'
```

**Avec Swift Package Manager:**
```swift
dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
             from: "1.19.2")
]
```

#### Étape 2: Ajouter les fichiers au projet

1. Copiez `model.onnx` dans votre projet Xcode
2. Copiez `spm.model` dans votre projet Xcode
3. Assurez-vous qu'ils sont ajoutés au Target > Build Phases > Copy Bundle Resources

#### Étape 3: Utiliser TextClassifier ou NLITester

**Classification zero-shot (TextClassifier.swift):**
```swift
let classifier = TextClassifier()

// Charger le modèle
guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx"),
      let tokenizerPath = Bundle.main.path(forResource: "spm", ofType: "model") else {
    fatalError("Fichiers du modèle introuvables")
}

try classifier.loadModel(modelPath: modelPath, tokenizerPath: tokenizerPath)

// Classifier un texte
let text = "Apple annonce un nouveau iPhone avec des fonctionnalités révolutionnaires"
let labels = ["technologie", "politique", "sport", "économie"]

let results = try classifier.classify(text: text, candidateLabels: labels)

for result in results {
    print("\(result.label): \(result.score * 100)%")
}
```

**Test NLI direct (NLITester.swift):**
```swift
let tester = NLITester()

// Charger le modèle
try tester.loadModel(modelPath: modelPath, tokenizerPath: tokenizerPath)

// Tester une paire premise-hypothesis
let result = try tester.predict(
    premise: "Angela Merkel is a politician in Germany",
    hypothesis: "Angela Merkel is a politician"
)

print("Entailment: \(result.entailment * 100)%")      // ~99.9%
print("Neutral: \(result.neutral * 100)%")            // ~0.1%
print("Contradiction: \(result.contradiction * 100)%") // ~0.0%

// Ou exécuter tous les tests
try tester.runAllTests()
```

## ⚠️ Important: Pourquoi ONNX Runtime au lieu de CoreML?

La conversion de mDeBERTa vers CoreML a échoué en raison de:
- Incompatibilités d'opérateurs Transformer
- Problèmes de types de données
- Complexité du modèle NLI

**ONNX Runtime offre:**
- ✅ Support complet des opérateurs Transformer
- ✅ Performance optimisée pour mobile
- ✅ Compatible avec tous les modèles HuggingFace
- ✅ Résultats identiques au pipeline Transformers (99.00% de précision)

## 📊 Performance attendue

Sur un iPhone 12 Pro:
- Temps d'inférence: ~200-250ms par classification
- Taille du modèle: 1.1 GB (complet) - meilleure précision
- RAM utilisée: ~500 MB

## 🔧 Points clés de l'implémentation

### Index d'entailment
Pour CE modèle, l'entailment est à l'index **0** (pas 2):
```swift
let entailmentLogit = logits[0]  // Index 0 = entailment
```

### Normalisation avec softmax
Appliquer softmax sur les **logits** d'entailment (pas les probabilités):
```swift
// Collecter les logits d'entailment de tous les labels
let maxLogit = results.map { $0.score }.max() ?? 0.0
let expLogits = results.map { exp($0.score - maxLogit) }
let sumExpLogits = expLogits.reduce(0.0, +)
// Calculer les scores finaux
let normalizedScore = expLogit / sumExpLogits
```

Voir `SOLUTION_FINALE.md` pour plus de détails.

## 📝 Exemple de test

```bash
# Tester avec Python (identique au résultat iOS)
python3 test_model.py
```

**Output**:
```
1. politics        99.00%  ███████████████████████████████████████
2. economy          0.55%
3. entertainment    0.42%
4. environment      0.04%
```

Le code Swift produit **exactement** les mêmes résultats!

## 🐛 Dépannage

### Erreur "Model file not found"

Vérifiez que les fichiers sont bien dans le Bundle:
1. Xcode > Target > Build Phases
2. Copy Bundle Resources
3. Vérifiez que `model.onnx` et `spm.model` sont présents

### Crash au chargement du modèle

Le modèle nécessite ~500 MB de RAM:
1. Sur simulateur, augmentez la RAM allouée
2. Sur device, fermez les autres apps
3. Chargez le modèle une seule fois au démarrage

### Résultats incorrects

Vérifiez:
1. Vous utilisez l'index **0** pour l'entailment (pas 2)
2. Vous appliquez softmax sur les **logits** (pas les probabilités)
3. Le template est "This example is {}." (pas "This text is about {}.")

## 📚 Ressources

- [ONNX Runtime iOS Guide](https://onnxruntime.ai/docs/tutorials/mobile/ios.html)
- [HuggingFace Model Card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
- [SentencePiece iOS](https://github.com/google/sentencepiece)

## 📄 License

Le modèle mDeBERTa est sous license MIT. Voir `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7/LICENSE`.
