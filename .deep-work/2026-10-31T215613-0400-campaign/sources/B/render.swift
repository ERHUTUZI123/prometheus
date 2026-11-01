import Foundation
import AppKit
import PDFKit
let args=CommandLine.arguments
guard args.count == 4, let page=Int(args[2]), let doc=PDFDocument(url:URL(fileURLWithPath:args[1])), let p=doc.page(at:page-1) else {fatalError("render.swift input.pdf 1-based-page output.png")}
let im=p.thumbnail(of:NSSize(width:1400,height:1900),for:.mediaBox)
guard let tiff=im.tiffRepresentation, let bitmap=NSBitmapImageRep(data:tiff), let png=bitmap.representation(using:.png,properties:[:]) else {fatalError("render failed")}
try png.write(to:URL(fileURLWithPath:args[3]))
print(args[3])
