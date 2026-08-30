import Foundation
import RustApi
import AppKit

final class CPGEDelegate: NSObject, NSApplicationDelegate {
    let bound: NSRect
    var layer: CAMetalLayer
    let window: NSWindow
    var data: UnsafeMutableRawPointer

    internal init(bound: NSRect) {
        self.bound = bound

        self.window = .init(contentRect: bound,
                            styleMask: .init(arrayLiteral: .closable, .titled, .miniaturizable),
                            backing: .buffered,
                            defer: false)
        self.layer = .init()
        self.layer.bounds = bound

        let view: NSView = .init()
        view.wantsLayer = true
        view.layer = self.layer
        self.window.contentView = view

        self.data = .allocate(byteCount: 0, alignment: 0)
        cpge_make_vulkan_data(&self.data)
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
        self.window.center()
        self.window.makeKeyAndOrderFront(nil)

        cpge_spawn_vulkan(
            Unmanaged.passUnretained(self.layer).toOpaque(),
            self.data
        )
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        cpge_macos_should_terminate()
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        cpge_macos_will_terminate()
    }
}

@c func cpge_init_application(_ width: Int, height: Int) {
    let application = NSApplication.shared
    let delegate = CPGEDelegate.init(bound: .init(x: 0, y: 0, width: width, height: height))

    application.delegate = delegate
}
