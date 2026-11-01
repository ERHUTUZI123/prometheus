import Foundation
import PDFKit
let doc=PDFDocument(url:URL(fileURLWithPath:CommandLine.arguments[1]))!
for i in 0..<doc.pageCount { print("\nPAGE \(i+1)\n"); print(doc.page(at:i)!.string ?? "[NO TEXT]") }
