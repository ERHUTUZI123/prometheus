import Foundation
import AppKit
import PDFKit

// Render only the explicitly supplied manuscript with the installed macOS PDF
// renderer. No UI automation, external downloads, or network operations.
let arguments = CommandLine.arguments
guard arguments.count == 3 else { fatalError("PDF input and output directory required") }
let output = URL(fileURLWithPath: arguments[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
guard let document = PDFDocument(url: URL(fileURLWithPath: arguments[1])) else {
    fatalError("Cannot read manuscript PDF")
}
func canvas(_ width: Int, _ height: Int) -> (NSBitmapImageRep, NSGraphicsContext) {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width,
        pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
        isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: width * 4,
        bitsPerPixel: 32)!
    let context = NSGraphicsContext(bitmapImageRep: bitmap)!
    context.cgContext.setFillColor(NSColor.white.cgColor)
    context.cgContext.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return (bitmap, context)
}
var images: [NSImage] = []
for index in 0..<document.pageCount {
    let page = document.page(at: index)!
    let bounds = page.bounds(for: .mediaBox)
    let scale = 1000.0 / bounds.width
    let height = Int(ceil(bounds.height * scale))
    let (bitmap, graphics) = canvas(1000, height)
    graphics.cgContext.scaleBy(x: scale, y: scale)
    graphics.cgContext.translateBy(x: -bounds.minX, y: -bounds.minY)
    page.draw(with: .mediaBox, to: graphics.cgContext)
    let data = bitmap.representation(using: .png, properties: [:])!
    try data.write(to: output.appendingPathComponent(String(format: "page-%02d.png", index + 1)))
    images.append(NSImage(data: data)!)
}
let columns = 4, tileWidth = 240, tileHeight = 340
let rows = Int(ceil(Double(images.count) / Double(columns)))
let (contact, graphics) = canvas(columns * tileWidth, rows * tileHeight)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = graphics
for (index, image) in images.enumerated() {
    let column = index % columns, row = rows - 1 - index / columns
    let rectangle = NSRect(x: column * tileWidth + 8, y: row * tileHeight + 25,
                           width: tileWidth - 16, height: tileHeight - 35)
    image.draw(in: rectangle, from: .zero, operation: .sourceOver, fraction: 1)
    ("Page \(index + 1)" as NSString).draw(at: NSPoint(x: column * tileWidth + 8,
        y: row * tileHeight + 5), withAttributes: [.font: NSFont.systemFont(ofSize: 12),
                                               .foregroundColor: NSColor.black])
}
NSGraphicsContext.restoreGraphicsState()
try contact.representation(using: .png, properties: [:])!.write(
    to: output.appendingPathComponent("contact.png"))
let metadata: [String: Any] = ["page_count": document.pageCount,
                              "renderer": "installed macOS PDFKit", "page_width_pixels": 1000]
let encoded = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted])
try encoded.write(to: output.appendingPathComponent("render.json"))
print(String(data: encoded, encoding: .utf8)!)
