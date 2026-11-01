import Foundation
import PDFKit
import Vision
import AppKit
let args=CommandLine.arguments
guard args.count >= 5, let doc=PDFDocument(url:URL(fileURLWithPath:args[1])) else {fatalError("usage PDF output start end")}
var out="PDF pages: \(doc.pageCount); OCR Vision accurate, English; inspect equations against scans\n"
for p in (Int(args[3])!-1)..<min(Int(args[4])!, doc.pageCount) {
 guard let page=doc.page(at:p) else {continue}
 let bounds=page.bounds(for:.mediaBox)
 let scale:CGFloat=2.0
 let width=Int(bounds.width*scale), height=Int(bounds.height*scale)
 guard let ctx=CGContext(data:nil,width:width,height:height,bitsPerComponent:8,bytesPerRow:0,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else {fatalError("ctx")}
 ctx.setFillColor(NSColor.white.cgColor);ctx.fill(CGRect(x:0,y:0,width:width,height:height))
 ctx.scaleBy(x:scale,y:scale);page.draw(with:.mediaBox,to:ctx)
 guard let img=ctx.makeImage() else {fatalError("image")}
 let req=VNRecognizeTextRequest();req.recognitionLevel = .accurate;req.recognitionLanguages=["en-US"];req.usesLanguageCorrection=true
 try VNImageRequestHandler(cgImage:img).perform([req])
 out += "\n\n===== PDF PAGE \(p+1) =====\n"
 for observation in req.results ?? [] {out += (observation.topCandidates(1).first?.string ?? "")+"\n"}
 print("OCR page \(p+1)")
}
try out.write(toFile:args[2],atomically:true,encoding:.utf8)
