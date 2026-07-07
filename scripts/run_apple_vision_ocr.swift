#!/usr/bin/env swift
import AppKit
import Foundation
import Vision

func value(after flag: String, in args: [String]) -> String? {
    guard let index = args.firstIndex(of: flag), index + 1 < args.count else { return nil }
    return args[index + 1]
}

let args = CommandLine.arguments
let inputPath = value(after: "--input", in: args)
let outputPath = value(after: "--output", in: args)
let recognitionLevel = value(after: "--recognition-level", in: args) ?? "accurate"
let language = value(after: "--language", in: args) ?? "en-US"

guard let inputPath, let outputPath else {
    FileHandle.standardError.write(Data("Usage: run_apple_vision_ocr.swift --input IMAGE --output MARKDOWN [--recognition-level accurate|fast] [--language en-US]\n".utf8))
    exit(2)
}

guard let image = NSImage(contentsOfFile: inputPath) else {
    FileHandle.standardError.write(Data("Cannot open image: \(inputPath)\n".utf8))
    exit(1)
}

var rect = CGRect(origin: .zero, size: image.size)
guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
    FileHandle.standardError.write(Data("Cannot convert image to CGImage: \(inputPath)\n".utf8))
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = recognitionLevel == "fast" ? .fast : .accurate
request.usesLanguageCorrection = true
if language != "unknown" {
    request.recognitionLanguages = [language]
}

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    FileHandle.standardError.write(Data("Apple Vision OCR failed: \(error)\n".utf8))
    exit(1)
}

let observations = (request.results ?? []).sorted { left, right in
    let dy = abs(left.boundingBox.midY - right.boundingBox.midY)
    if dy > 0.01 { return left.boundingBox.midY > right.boundingBox.midY }
    return left.boundingBox.minX < right.boundingBox.minX
}

let lines = observations.compactMap { observation in
    observation.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines)
}.filter { !$0.isEmpty }

let outputURL = URL(fileURLWithPath: outputPath)
do {
    try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try (lines.joined(separator: "\n") + "\n").write(to: outputURL, atomically: true, encoding: .utf8)
} catch {
    FileHandle.standardError.write(Data("Cannot write OCR output: \(error)\n".utf8))
    exit(1)
}
