import Foundation
import PDFKit
let args = CommandLine.arguments
guard args.count >= 3, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])) else { fatalError("Usage: extract.swift input.pdf output.txt") }
var out = "PAGES: \(doc.pageCount)\n"
for i in 0..<doc.pageCount {
    out += "\n\n===== PDF PAGE \(i+1) =====\n"
    out += doc.page(at: i)?.string ?? "[NO TEXT]"
}
try out.write(toFile: args[2], atomically: true, encoding: .utf8)
print("Extracted \(doc.pageCount) pages to \(args[2])")
