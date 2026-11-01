import Foundation
import AppKit
import PDFKit
let doc=PDFDocument(url:URL(fileURLWithPath:CommandLine.arguments[1]))!
let out=CommandLine.arguments[2]
for a in CommandLine.arguments.dropFirst(3) {let i=Int(a)!-1;let img=doc.page(at:i)!.thumbnail(of:NSSize(width:1500,height:2000),for:.mediaBox);let rep=NSBitmapImageRep(data:img.tiffRepresentation!)!;try rep.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:out+"/page-"+String(i+1)+".png"))}
