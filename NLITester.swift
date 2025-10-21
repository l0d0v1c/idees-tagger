import Foundation
import onnxruntime_objc

/// Résultat d'une prédiction NLI
struct NLIPrediction {
    let entailment: Double
    let neutral: Double
    let contradiction: Double

    /// Label avec le score le plus élevé
    var predictedLabel: String {
        let scores = [
            ("entailment", entailment),
            ("neutral", neutral),
            ("contradiction", contradiction)
        ]
        return scores.max(by: { $0.1 < $1.1 })?.0 ?? "unknown"
    }

    /// Affichage formaté
    func format() -> String {
        let formatter: (String, Double) -> String = { label, score in
            let percentage = score * 100
            let barLength = Int(percentage / 2.5)  // Scale to 40 chars max
            let bar = String(repeating: "█", count: barLength) + String(repeating: "░", count: 40 - barLength)
            return String(format: "  %-15s %5.1f%%  %@", label, percentage, bar)
        }

        return """
        \(formatter("entailment", entailment))
        \(formatter("neutral", neutral))
        \(formatter("contradiction", contradiction))
        """
    }
}

/// Testeur NLI avec ONNX Runtime
class NLITester {
    private var session: ORTSession?
    private var tokenizer: SentencePieceTokenizer?

    // MARK: - Configuration

    struct Config {
        static let maxLength = 128
        static let padTokenId = 0
        static let sepTokenId = 2
        static let clsTokenId = 1
    }

    // MARK: - Loading

    /// Charge le modèle ONNX et le tokenizer
    func loadModel(modelPath: String, tokenizerPath: String) throws {
        print("=" + String(repeating: "=", count: 60))
        print("Test NLI - Natural Language Inference (ONNX)")
        print("=" + String(repeating: "=", count: 60))

        print("\n1. Chargement du tokenizer depuis \(tokenizerPath)")
        // TODO: Implémenter SentencePiece tokenizer
        // tokenizer = try SentencePieceTokenizer(modelPath: tokenizerPath)
        print("⚠️  Tokenizer SentencePiece à implémenter")

        print("\n2. Chargement du modèle ONNX depuis \(modelPath)")
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        print("✓ Modèle ONNX chargé")
    }

    // MARK: - Inference

    /// Prédit la relation NLI entre premise et hypothesis
    func predict(premise: String, hypothesis: String) throws -> NLIPrediction {
        guard let session = session else {
            throw NSError(domain: "NLITester", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        // Tokeniser (placeholder - à implémenter avec SentencePiece)
        let inputIds = tokenizePlaceholder(premise: premise, hypothesis: hypothesis)
        let attentionMask = Array(repeating: Int64(1), count: inputIds.count)

        // Préparer les inputs pour ONNX Runtime
        let inputIdsData = Data(bytes: inputIds, count: inputIds.count * MemoryLayout<Int64>.size)
        let attentionMaskData = Data(bytes: attentionMask, count: attentionMask.count * MemoryLayout<Int64>.size)

        let inputIdsShape: [NSNumber] = [1, NSNumber(value: inputIds.count)]
        let attentionMaskShape: [NSNumber] = [1, NSNumber(value: attentionMask.count)]

        let inputIdsTensor = try ORTValue(
            tensorData: NSMutableData(data: inputIdsData),
            elementType: .int64,
            shape: inputIdsShape
        )

        let attentionMaskTensor = try ORTValue(
            tensorData: NSMutableData(data: attentionMaskData),
            elementType: .int64,
            shape: attentionMaskShape
        )

        // Inférence
        let outputs = try session.run(
            withInputs: [
                "input_ids": inputIdsTensor,
                "attention_mask": attentionMaskTensor
            ],
            outputNames: ["logits"],
            runOptions: nil
        )

        guard let logitsTensor = outputs["logits"],
              let logitsData = try? logitsTensor.tensorData() as Data else {
            throw NSError(domain: "NLITester", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to get logits"])
        }

        // Extraire les logits
        let logits = logitsData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
            let floatPtr = ptr.bindMemory(to: Float.self)
            return Array(floatPtr)
        }

        // Appliquer softmax
        let maxLogit = logits.max() ?? 0.0
        let expLogits = logits.map { exp($0 - maxLogit) }
        let sumExpLogits = expLogits.reduce(0.0, +)
        let probs = expLogits.map { $0 / sumExpLogits }

        // Créer le résultat
        // Labels: entailment (0), neutral (1), contradiction (2)
        return NLIPrediction(
            entailment: Double(probs[0]),
            neutral: Double(probs[1]),
            contradiction: Double(probs[2])
        )
    }

    // MARK: - Tokenization (Placeholder)

    /// Placeholder pour la tokenization - à remplacer par SentencePiece
    private func tokenizePlaceholder(premise: String, hypothesis: String) -> [Int64] {
        // TODO: Implémenter avec SentencePiece
        // Format: [CLS] premise [SEP] hypothesis [SEP] + padding

        // Pour l'instant, retourne un placeholder de la bonne taille
        var tokens: [Int64] = [Int64(Config.clsTokenId)]

        // Ajouter des tokens factices pour premise
        tokens += Array(repeating: Int64(100), count: 30)
        tokens.append(Int64(Config.sepTokenId))

        // Ajouter des tokens factices pour hypothesis
        tokens += Array(repeating: Int64(100), count: 20)
        tokens.append(Int64(Config.sepTokenId))

        // Padding
        while tokens.count < Config.maxLength {
            tokens.append(Int64(Config.padTokenId))
        }

        // Truncate si nécessaire
        if tokens.count > Config.maxLength {
            tokens = Array(tokens.prefix(Config.maxLength))
        }

        return tokens
    }

    // MARK: - Test Suite

    /// Exécute tous les tests NLI
    func runAllTests() throws {
        print("\n" + String(repeating: "=", count: 60))
        print("⚠️  ATTENTION: Tests utilisent un tokenizer placeholder")
        print("Implémentez SentencePiece pour des résultats réels")
        print(String(repeating: "=", count: 60))

        // Test 1: Multilingue
        try runTest(
            name: "Test 1: Multilingue (DE → EN)",
            premise: "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU",
            hypothesis: "Emmanuel Macron is the President of France"
        )

        // Test 2: Entailment
        try runTest(
            name: "Test 2: Entailment - Implication vraie",
            premise: "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU",
            hypothesis: "Angela Merkel is a politician"
        )

        // Test 3: Contradiction
        try runTest(
            name: "Test 3: Contradiction - Contradiction évidente",
            premise: "Le soleil brille et il fait beau",
            hypothesis: "Il pleut et il fait sombre"
        )

        // Test 4: Neutral
        try runTest(
            name: "Test 4: Neutral - Pas de relation claire",
            premise: "J'aime manger des pommes",
            hypothesis: "La tour Eiffel est à Paris"
        )

        print("\n" + String(repeating: "=", count: 60))
        print("✓ Tests NLI terminés")
        print(String(repeating: "=", count: 60))
        print("\nNote: Implémentez SentencePiece tokenizer pour des résultats réels")
    }

    private func runTest(name: String, premise: String, hypothesis: String) throws {
        print("\n" + String(repeating: "=", count: 60))
        print(name)
        print(String(repeating: "=", count: 60))
        print("\nPremise:    \(premise)")
        print("Hypothesis: \(hypothesis)")

        let prediction = try predict(premise: premise, hypothesis: hypothesis)

        print("\nRésultats:")
        print(prediction.format())
        print("\nPrédiction: \(prediction.predictedLabel)")
    }
}

// MARK: - SentencePiece Tokenizer (à implémenter)

/// Placeholder pour le tokenizer SentencePiece
class SentencePieceTokenizer {
    init(modelPath: String) throws {
        // TODO: Implémenter avec la librairie SentencePiece
        // https://github.com/google/sentencepiece
        throw NSError(domain: "SentencePieceTokenizer", code: 1,
                     userInfo: [NSLocalizedDescriptionKey: "Not implemented yet"])
    }

    func encode(text: String) -> [Int64] {
        // TODO: Implémenter
        return []
    }
}

// MARK: - Exemple d'utilisation

#if DEBUG
func exampleUsage() {
    let tester = NLITester()

    do {
        // Chemins des fichiers
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx"),
              let tokenizerPath = Bundle.main.path(forResource: "spm", ofType: "model") else {
            print("❌ Fichiers du modèle introuvables")
            return
        }

        // Charger le modèle
        try tester.loadModel(modelPath: modelPath, tokenizerPath: tokenizerPath)

        // Exécuter les tests
        try tester.runAllTests()

        // Ou tester une paire spécifique
        let result = try tester.predict(
            premise: "Angela Merkel is a politician in Germany",
            hypothesis: "Angela Merkel is a politician"
        )

        print("\nRésultat:")
        print(result.format())

    } catch {
        print("❌ Erreur: \(error)")
    }
}
#endif
