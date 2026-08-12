import Foundation
import RustApi
import AppKit

class CPGEDelegate: NSObject, NSApplicationDelegate {
    let bound: NSRect
    var layer: CAMetalLayer
    let window: NSWindow

    internal init(bound: NSRect) {
        self.bound = bound

        self.window = .init(contentRect: bound,
            styleMask: .init(arrayLiteral: .closable, .titled, .miniaturizable),
            backing: .buffered,
            defer: false)
        self.layer = .init()
        self.layer.bounds = bound

        var view: NSView = .init()
        view.layer = self.layer
        self.window.contentView = view
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menuBar: NSMenu = .init();

        let menuItem: NSMenuItem = .init();
        let menu: NSMenu = .init();
        menu.addItem(withTitle: "Quit", action: #selector(NSApp.terminate), keyEquivalent: "q")
        menuItem.submenu = menu;

        menuBar.addItem(menuItem)

        NSApp.mainMenu = menuBar
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        self.window.makeKeyAndOrderFront(nil)

        cpge_spawn_vulkan(Unmanaged.passUnretained(self.layer).toOpaque())
    }
}

@c func cpge_init_application(_ width: Int, height: Int) {
    let application = NSApplication.shared
    let delegate = CPGEDelegate.init(bound: .init(x: 0, y: 0, width: width, height: height))

    application.delegate = delegate
}

@c func cpge_mainloop() {
    NSApplication.shared.run()
}
