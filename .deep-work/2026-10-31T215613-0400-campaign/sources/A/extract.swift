import Foundation
import PDFKit
let args = CommandLine.arguments
guard args.count >= 3 else { fatalError("usage extract PDF output [start end]") }
guard let doc = PDFDocument(url:URL(fileURLWithPath:args[1])) else { fatalError("no PDF") }
let start = args.count > 3 ? Int(args[3])! : 1
let end = args.count > 4 ? min(Int(args[4])!,doc.pageCount) : doc.pageCount
var result="PDF pages: \(doc.pageCount)\n"
for p in (start-1)..<end {
 result += "\n\n===== PDF PAGE \(p+1) =====\n"
 result += doc.page(at:p)?.string ?? "[NO EXTRACTED TEXT]"
}
try result.write(toFile:args[2], atomically:true, encoding:.utf8)
print("extracted \(start)-\(end) / \(doc.pageCount); \(result.utf8.count) bytes")
