import PDFKit
import AppKit
import Vision
import Foundation
let args=CommandLine.arguments
let doc=PDFDocument(url:URL(fileURLWithPath:args[1]))!
for token in args.dropFirst(2) {
 let idx=Int(token)!-1
 let page=doc.page(at:idx)!
 print("\nPDF PAGE \(idx+1)\n")
 let text=page.string ?? ""
 if text.count > 80 { print(text); continue }
 let img=page.thumbnail(of:NSSize(width:1600,height:2200),for:.mediaBox)
 var rect=CGRect(origin:.zero,size:img.size)
 let cg=img.cgImage(forProposedRect:&rect,context:nil,hints:nil)!
 let req=VNRecognizeTextRequest()
 req.recognitionLevel = .accurate
 req.recognitionLanguages=["en-US"]
 req.usesLanguageCorrection=true
 try VNImageRequestHandler(cgImage:cg).perform([req])
 for line in req.results ?? [] { if let c=line.topCandidates(1).first { print(c.string) } }
}
