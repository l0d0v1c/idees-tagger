# Guide d'intégration iOS - Classification Zero-Shot mDeBERTa

## 📱 Setup complet pour iOS

### Étape 1: Créer un nouveau projet Xcode

1. Ouvrez Xcode
2. File > New > Project
3. Choisissez "App" (iOS)
4. Nommez votre projet (ex: "TextClassifierApp")

### Étape 2: Ajouter les dépendances

#### Option A: CocoaPods (Recommandé)

1. Créez un `Podfile` à la racine du projet:

```ruby
platform :ios, '15.0'
use_frameworks!

target 'TextClassifierApp' do
  pod 'onnxruntime-objc', '~> 1.19.2'
end
```

2. Installez les pods:

```bash
pod install
```

3. Ouvrez le workspace:

```bash
open TextClassifierApp.xcworkspace
```

#### Option B: Swift Package Manager

1. Xcode > File > Add Package Dependencies
2. Ajoutez: `https://github.com/microsoft/onnxruntime-swift-package-manager`
3. Version: 1.19.2 ou supérieure

### Étape 3: Ajouter les fichiers du modèle

1. **Télécharger les fichiers nécessaires:**
   - `model_quantized.onnx` (338 MB)
   - `spm.model` (tokenizer SentencePiece)

2. **Ajouter au projet:**
   - Glissez-déposez les fichiers dans Xcode
   - Cochez "Copy items if needed"
   - Cochez votre Target
   - Vérifiez dans Build Phases > Copy Bundle Resources

### Étape 4: Implémenter le tokenizer SentencePiece

Le modèle utilise SentencePiece pour la tokenisation. Vous avez deux options:

#### Option A: Wrapper Python (développement/test)

Pour le développement rapide, utilisez PythonKit:

```swift
import PythonKit

class SentencePieceTokenizer {
    let tokenizer: PythonObject

    init(modelPath: String) throws {
        let transformers = Python.import("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(modelPath)
    }

    func encode(text: String, textPair: String?, maxLength: Int) -> ([Int32], [Int32]) {
        let inputs = tokenizer(
            text,
            textPair,
            truncation: true,
            max_length: maxLength,
            padding: "max_length",
            return_tensors: "np"
        )

        let inputIds = Array<Int32>(numpy: inputs["input_ids"])!
        let attentionMask = Array<Int32>(numpy: inputs["attention_mask"])!

        return (inputIds, attentionMask)
    }
}
```

#### Option B: Bibliothèque native (production)

Pour la production, utilisez une implémentation native:

```ruby
# Podfile
pod 'SentencePiece'
```

```swift
import SentencePiece

class SentencePieceTokenizer {
    private let processor: SPMProcessor

    init(modelPath: String) throws {
        processor = try SPMProcessor(modelPath: modelPath)
    }

    func encode(text: String, textPair: String?, maxLength: Int) -> ([Int32], [Int32]) {
        // Combiner text et textPair avec le séparateur [SEP]
        let fullText = textPair != nil ? "\\(text) [SEP] \\(textPair!)" : text

        // Tokeniser
        var tokens = processor.encode(fullText)

        // Tronquer ou padder à maxLength
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength))
        }

        let inputIds = tokens.map { Int32($0) }
        let attentionMask = Array(repeating: Int32(1), count: tokens.count)

        // Padding si nécessaire
        let paddingLength = maxLength - tokens.count
        if paddingLength > 0 {
            let paddedInputIds = inputIds + Array(repeating: Int32(0), count: paddingLength)
            let paddedMask = attentionMask + Array(repeating: Int32(0), count: paddingLength)
            return (paddedInputIds, paddedMask)
        }

        return (inputIds, attentionMask)
    }
}
```

### Étape 5: Intégrer TextClassifier

Copiez `TextClassifier.swift` dans votre projet et utilisez-le:

```swift
import SwiftUI

struct ContentView: View {
    @State private var text = ""
    @State private var results: [TextClassifier.ClassificationResult] = []
    @State private var isLoading = false

    private let classifier = TextClassifier()
    private let labels = ["politics", "economy", "technology", "sports", "entertainment"]

    var body: some View {
        VStack(spacing: 20) {
            Text("Classification Zero-Shot")
                .font(.title)
                .bold()

            TextEditor(text: $text)
                .frame(height: 150)
                .border(Color.gray, width: 1)
                .padding()

            Button("Classifier") {
                classifyText()
            }
            .disabled(text.isEmpty || isLoading)
            .buttonStyle(.borderedProminent)

            if isLoading {
                ProgressView()
            }

            List(results, id: \\.label) { result in
                HStack {
                    Text(result.label)
                        .font(.headline)
                    Spacer()
                    Text(String(format: "%.1f%%", result.score * 100))
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    // Barre de progression
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .frame(width: geometry.size.width, height: 8)
                                .opacity(0.3)
                                .foregroundColor(.gray)

                            Rectangle()
                                .frame(width: geometry.size.width * CGFloat(result.score), height: 8)
                                .foregroundColor(.blue)
                        }
                    }
                    .frame(height: 8)
                }
            }
        }
        .padding()
        .onAppear {
            loadModel()
        }
    }

    private func loadModel() {
        guard let modelPath = Bundle.main.path(forResource: "model_quantized", ofType: "onnx"),
              let tokenizerPath = Bundle.main.path(forResource: "spm", ofType: "model") else {
            print("❌ Fichiers du modèle introuvables")
            return
        }

        do {
            try classifier.loadModel(modelPath: modelPath, tokenizerPath: tokenizerPath)
        } catch {
            print("❌ Erreur lors du chargement: \\(error)")
        }
    }

    private func classifyText() {
        isLoading = true

        Task {
            do {
                let classificationResults = try classifier.classify(
                    text: text,
                    candidateLabels: labels
                )

                await MainActor.run {
                    results = classificationResults
                    isLoading = false
                }
            } catch {
                print("❌ Erreur: \\(error)")
                await MainActor.run {
                    isLoading = false
                }
            }
        }
    }
}
```

### Étape 6: Configuration de la mémoire

Le modèle est volumineux. Dans `Info.plist`, ajoutez:

```xml
<key>UIApplicationExitsOnSuspend</key>
<false/>
<key>UIBackgroundModes</key>
<array>
    <string>processing</string>
</array>
```

### Étape 7: Optimisations pour la performance

#### Mise en cache du modèle

```swift
class ModelManager {
    static let shared = ModelManager()
    private(set) var classifier: TextClassifier?

    private init() {}

    func loadModel() throws {
        guard classifier == nil else { return }

        let classifier = TextClassifier()
        try classifier.loadModel(/* ... */)
        self.classifier = classifier
    }
}
```

#### Inférence en arrière-plan

```swift
func classify(text: String) async throws -> [ClassificationResult] {
    try await Task.detached(priority: .userInitiated) {
        try classifier.classify(text: text, candidateLabels: labels)
    }.value
}
```

### Étape 8: Tester sur device

1. Connectez un iPhone/iPad
2. Sélectionnez votre device dans Xcode
3. Cmd + R pour compiler et lancer
4. Testez avec différents textes

## 🐛 Problèmes courants

### "Model file too large"

Le modèle quantifié (338 MB) peut dépasser les limites de certaines configurations. Solutions:

1. Téléchargez le modèle au premier lancement:

```swift
func downloadModelIfNeeded() async throws {
    let modelURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("model_quantized.onnx")

    if !FileManager.default.fileExists(atPath: modelURL.path) {
        // Télécharger depuis votre serveur
        let remoteURL = URL(string: "https://your-server.com/model_quantized.onnx")!
        let (localURL, _) = try await URLSession.shared.download(from: remoteURL)
        try FileManager.default.moveItem(at: localURL, to: modelURL)
    }
}
```

2. Hébergez le modèle sur un serveur et faites des appels API

### "Out of memory"

1. Fermez les autres apps
2. Réduisez `maxSequenceLength` de 128 à 64
3. Utilisez le modèle sur iPhone récents uniquement (≥ iPhone 12)

### "Slow inference"

1. Utilisez `model_quantized.onnx` (pas `model.onnx`)
2. Activez CoreML dans ONNX Runtime:

```swift
let options = try ORTSessionOptions()
options.appendExecutionProvider(.coreML)
```

3. Réduisez la longueur de séquence

## 📊 Performance attendue

| Device        | Temps d'inférence | Mémoire |
|---------------|-------------------|---------|
| iPhone 15 Pro | ~80ms            | 450 MB  |
| iPhone 14     | ~120ms           | 480 MB  |
| iPhone 12     | ~180ms           | 520 MB  |
| iPhone 11     | ~250ms           | 580 MB  |

## 🎯 Prochaines étapes

- [ ] Implémenter le cache des résultats
- [ ] Ajouter la détection de langue
- [ ] Créer une UI plus élégante
- [ ] Tests unitaires
- [ ] Tests d'intégration

## 📚 Ressources

- [ONNX Runtime iOS](https://onnxruntime.ai/docs/tutorials/mobile/ios.html)
- [SentencePiece iOS](https://github.com/google/sentencepiece)
- [Modèle HuggingFace](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
