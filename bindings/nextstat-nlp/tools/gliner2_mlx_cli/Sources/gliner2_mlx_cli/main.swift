import Foundation
import GLiNER2Swift
import Darwin

struct Request: Codable {
    let text: String
    let labels: [String]
}

struct Entity: Codable {
    let label: String
    let text: String
    let start: Int
    let end: Int
    let score: Double?
}

struct Response: Codable {
    let entities: [Entity]
    let error: String?
}

func printUsage() {
    FileHandle.standardError.write(Data(
        "Usage: gliner2_mlx_cli [--model-id <hf_id>] [--threshold <f>] [--jsonl]\n".utf8
    ))
}

@main
struct CLI {
    static func main() async {
        var modelId = "fastino/gliner2-base-v1"
        var jsonl = false
        var threshold: Float = 0.5

        let args = Array(CommandLine.arguments.dropFirst())
        var i = 0
        while i < args.count {
            let a = args[i]
            if a == "--model-id" {
                guard i + 1 < args.count else { printUsage(); exit(2) }
                modelId = args[i + 1]
                i += 2
                continue
            }
            if a == "--threshold" {
                guard i + 1 < args.count else { printUsage(); exit(2) }
                threshold = Float(args[i + 1]) ?? threshold
                i += 2
                continue
            }
            if a == "--jsonl" {
                jsonl = true
                i += 1
                continue
            }
            if a == "-h" || a == "--help" {
                printUsage()
                exit(0)
            }
            FileHandle.standardError.write(Data(("Unknown arg: \(a)\n").utf8))
            printUsage()
            exit(2)
        }

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Redirect all library logging to stderr, but keep our JSON protocol on the original stdout.
        // MLX / swift-transformers may print during model load/inference.
        let savedStdout = dup(STDOUT_FILENO)
        if savedStdout != -1 {
            _ = dup2(STDERR_FILENO, STDOUT_FILENO)
        }
        let jsonOut = FileHandle(fileDescriptor: savedStdout == -1 ? STDOUT_FILENO : savedStdout, closeOnDealloc: savedStdout != -1)

        let gliner: GLiNER2
        do {
            gliner = try await GLiNER2.fromPretrained(modelId)
        } catch {
            let resp = Response(entities: [], error: "Failed to load model_id=\(modelId): \(error)")
            let data = try! encoder.encode(resp)
            jsonOut.write(data)
            jsonOut.write(Data("\n".utf8))
            exit(1)
        }

        func handleLine(_ line: String) {
            guard let data = line.data(using: .utf8) else { return }
            do {
                let req = try decoder.decode(Request.self, from: data)
                let out = gliner.extractEntities(
                    text: req.text,
                    entityTypes: req.labels,
                    threshold: threshold,
                    includeConfidence: true,
                    includeSpans: true
                )

                var entities: [Entity] = []
                if let ents = out["entities"] as? [String: Any] {
                    for (label, itemsAny) in ents {
                        if let items = itemsAny as? [[String: Any]] {
                            for it in items {
                                let t = it["text"] as? String ?? ""
                                let start = it["start"] as? Int ?? -1
                                let end = it["end"] as? Int ?? -1
                                let conf = it["confidence"] as? Double
                                if !t.isEmpty && start >= 0 && end >= 0 {
                                    entities.append(Entity(label: label, text: t, start: start, end: end, score: conf))
                                }
                            }
                        }
                    }
                }

                let resp = Response(entities: entities, error: nil)
                let respData = try encoder.encode(resp)
                jsonOut.write(respData)
                jsonOut.write(Data("\n".utf8))
            } catch {
                let resp = Response(entities: [], error: "\(error)")
                let respData = try! encoder.encode(resp)
                jsonOut.write(respData)
                jsonOut.write(Data("\n".utf8))
            }
        }

        if jsonl {
            while let line = readLine() {
                if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    continue
                }
                handleLine(line)
            }
        } else {
            let stdin = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) ?? ""
            if stdin.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let resp = Response(entities: [], error: "Empty stdin")
                let data = try! encoder.encode(resp)
                jsonOut.write(data)
                jsonOut.write(Data("\n".utf8))
                exit(2)
            }
            let firstLine = stdin.split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: true).first
            handleLine(String(firstLine ?? Substring(stdin)))
        }
    }
}
