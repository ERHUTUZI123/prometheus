import Foundation
import PDFKit
let args = CommandLine.arguments
guard args.count >= 4, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])) else { fatalError("path mode range-or-regex") }
print("FILE: \(args[1])\nPDF PAGES: \(doc.pageCount)")
if args[2] == "search" {
  let pattern = try NSRegularExpression(pattern: args[3], options: [.caseInsensitive])
  for i in 0..<doc.pageCount {
    let s = doc.page(at:i)?.string ?? ""
    let ns = s as NSString
    let matches = pattern.matches(in:s,range:NSRange(location:0,length:ns.length))
    if !matches.isEmpty {
      print("PDF \(i+1) MATCHES \(matches.count)")
      for m in matches.prefix(3) {
        let start = max(0,m.range.location-90)
        let end = min(ns.length,m.range.location+m.range.length+160)
        print(ns.substring(with:NSRange(location:start,length:end-start)).replacingOccurrences(of:"\n",with:" "))
      }
    }
  }
} else {
  for r in args[3].split(separator:",") {
    let v = r.split(separator:"-").compactMap { Int($0) }
    guard let first = v.first else { continue }
    let last = v.count>1 ? v[1] : first
    for n in first...last where n>0 && n<=doc.pageCount {
      print("\n=== PDF PAGE \(n) ===\n\(doc.page(at:n-1)?.string ?? "[NO TEXT]")")
    }
  }
}
