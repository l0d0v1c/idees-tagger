import Foundation
import onnxruntime_objc

/// Classe pour la classification de texte zero-shot utilisant le modèle mDeBERTa
///
/// Exemple d'utilisation:
/// ```swift
/// let classifier = TextClassifier()
/// try classifier.loadModel()
/// let result = try classifier.classify(
///     text: "Angela Merkel ist eine Politikerin in Deutschland",
///     candidateLabels: ["politics", "economy", "entertainment", "environment"]
/// )
/// print(result)
/// ```
class TextClassifier {

    // MARK: - Properties

    private var session: ORTSession?
    private var tokenizer: SentencePieceTokenizer?
    private let maxSequenceLength = 128

    // MARK: - Initialization

    init() {}

    // MARK: - Model Loading

    /// Charge le modèle ONNX et le tokenizer
    /// - Parameter modelPath: Chemin vers le fichier model_quantized.onnx
    /// - Parameter tokenizerPath: Chemin vers le fichier spm.model (SentencePiece)
    func loadModel(modelPath: String = "model_quantized.onnx",
                   tokenizerPath: String = "spm.model") throws {

        // Charger le modèle ONNX
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()

        // Optimiser pour mobile
        options.setGraphOptimizationLevel(.all)

        // Créer la session
        session = try ORTSession(env: env,
                                 modelPath: modelPath,
                                 sessionOptions: options)

        // Charger le tokenizer SentencePiece
        tokenizer = try SentencePieceTokenizer(modelPath: tokenizerPath)

        print("✓ Modèle chargé avec succès")
    }

    // MARK: - Classification

    /// Structure pour les résultats de classification
    struct ClassificationResult {
        let label: String
        let score: Float
    }

    /// Classifie un texte selon plusieurs labels candidats (zero-shot classification)
    /// - Parameters:
    ///   - text: Le texte à classifier
    ///   - candidateLabels: Liste des labels possibles
    ///   - hypothesisTemplate: Template pour le format NLI (par défaut: "This text is about {}.")
    /// - Returns: Tableau de résultats triés par score décroissant
    func classify(text: String,
                  candidateLabels: [String],
                  hypothesisTemplate: String = "This example is {}.") throws -> [ClassificationResult] {

        guard let session = session, let tokenizer = tokenizer else {
            throw NSError(domain: "TextClassifier",
                         code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Model not loaded. Call loadModel() first."])
        }

        var results: [ClassificationResult] = []

        // Pour chaque label candidat, créer une paire (premise, hypothesis)
        for label in candidateLabels {
            let hypothesis = hypothesisTemplate.replacingOccurrences(of: "{}", with: label)

            // Tokeniser la paire texte-hypothèse
            let (inputIds, attentionMask) = try tokenizer.encode(
                text: text,
                textPair: hypothesis,
                maxLength: maxSequenceLength
            )

            // Créer les tenseurs d'entrée
            let inputIdsData = NSMutableData(bytes: inputIds,
                                            length: inputIds.count * MemoryLayout<Int32>.size)
            let attentionMaskData = NSMutableData(bytes: attentionMask,
                                                 length: attentionMask.count * MemoryLayout<Int32>.size)

            let inputIdsTensor = try ORTValue(
                tensorData: inputIdsData,
                elementType: .int32,
                shape: [1, NSNumber(value: inputIds.count)]
            )

            let attentionMaskTensor = try ORTValue(
                tensorData: attentionMaskData,
                elementType: .int32,
                shape: [1, NSNumber(value: attentionMask.count)]
            )

            // Exécuter l'inférence
            let inputs: [String: ORTValue] = [
                "input_ids": inputIdsTensor,
                "attention_mask": attentionMaskTensor
            ]

            let outputs = try session.run(
                withInputs: inputs,
                outputNames: ["logits"],
                runOptions: nil
            )

            // Extraire les logits
            guard let logitsTensor = outputs["logits"],
                  let logitsData = try logitsTensor.tensorData() as Data? else {
                throw NSError(domain: "TextClassifier",
                            code: 2,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get model output"])
            }

            // Extraire les logits
            let logits = logitsData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
                let floatPtr = ptr.bindMemory(to: Float.self)
                return Array(floatPtr)
            }

            // Extraire le LOGIT d'entailment (on appliquera softmax plus tard)
            // IMPORTANT: Pour le modèle mDeBERTa-v3-base-xnli-multilingual-nli-2mil7:
            // Index 0 = entailment, Index 1 = neutral, Index 2 = contradiction
            // (vérifiez config.json de votre modèle si vous utilisez un autre modèle)
            let entailmentLogit = logits[0]  // Index 0 = entailment

            results.append(ClassificationResult(label: label, score: entailmentLogit))
        }

        // Appliquer softmax sur les LOGITS d'entailment (comme le pipeline avec multi_label=False)
        // C'est ce que fait le pipeline zero-shot classification de HuggingFace par défaut
        let maxLogit = results.map { $0.score }.max() ?? 0.0
        let expLogits = results.map { exp($0.score - maxLogit) }  // Soustraire max pour stabilité numérique
        let sumExpLogits = expLogits.reduce(0.0, +)

        let normalizedResults = zip(results, expLogits).map { (result, expLogit) in
            ClassificationResult(
                label: result.label,
                score: expLogit / sumExpLogits  // Softmax sur les logits
            )
        }

        // Trier par score décroissant
        return normalizedResults.sorted { $0.score > $1.score }
    }

    // MARK: - Helper Functions

    /// Applique la fonction softmax
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0
        let expScores = logits.map { exp($0 - maxLogit) }
        let sumExpScores = expScores.reduce(0, +)
        return expScores.map { $0 / sumExpScores }
    }
}

// MARK: - SentencePiece Tokenizer

/// Tokenizer SentencePiece pour le modèle mDeBERTa
/// Note: Vous devrez implémenter ou utiliser une bibliothèque SentencePiece pour iOS
/// Voir: https://github.com/google/sentencepiece
class SentencePieceTokenizer {

    // TODO: Implémenter avec la bibliothèque SentencePiece
    // Pour l'instant, ceci est un placeholder

    init(modelPath: String) throws {
        // Charger le modèle SentencePiece
        print("⚠️  SentencePieceTokenizer non encore implémenté")
        print("   Vous devez intégrer la bibliothèque SentencePiece pour iOS")
        print("   Voir: https://github.com/google/sentencepiece")
    }

    func encode(text: String, textPair: String?, maxLength: Int) throws -> ([Int32], [Int32]) {
        // TODO: Implémenter l'encodage
        // Retourner (input_ids, attention_mask)

        // Placeholder pour la démonstration
        return ([], [])
    }
}

// MARK: - Usage Example

/// Exemple d'utilisation dans une app iOS
class ExampleUsage {

    func runExample() {
        let classifier = TextClassifier()

        do {
            // Charger le modèle depuis le bundle de l'app
            guard let modelPath = Bundle.main.path(forResource: "model_quantized", ofType: "onnx"),
                  let tokenizerPath = Bundle.main.path(forResource: "spm", ofType: "model") else {
                print("❌ Fichiers du modèle introuvables dans le bundle")
                return
            }

            try classifier.loadModel(modelPath: modelPath, tokenizerPath: tokenizerPath)

            // Classifier un texte
            let text = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
            let labels = ["politics", "economy", "entertainment", "environment"]

            let results = try classifier.classify(text: text, candidateLabels: labels)

            // Afficher les résultats
            print("\nRésultats de classification pour:")
            print("  \"\(text)\"\n")
            for result in results {
                print(String(format: "  %@ : %.2f%%", result.label, result.score * 100))
            }

        } catch {
            print("❌ Erreur: \(error)")
        }
    }
}

// MARK: - Package.swift Dependencies

/*
 Pour utiliser ONNX Runtime dans votre projet iOS, ajoutez à votre Podfile:

 ```ruby
 pod 'onnxruntime-objc', '~> 1.19.2'
 ```

 Ou avec Swift Package Manager, ajoutez:

 ```swift
 dependencies: [
     .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.19.2")
 ]
 ```

 Pour SentencePiece, vous pouvez utiliser:
 ```ruby
 pod 'SentencePiece'
 ```
*/
