/// Windows Native Menu via muda (§88–94).
///
/// Usage:
///   1. Call `NativeMenu::build()` once to construct menu items and store IDs.
///   2. Call `NativeMenu::attach(hwnd)` after the window is created (first update() frame).
///   3. Call `NativeMenu::poll()` each frame to drain MenuEvents into FileCommand.
use muda::{
    Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu,
    accelerator::{Accelerator, Code, Modifiers},
};
use crate::desktop::file_ops::FileCommand;

pub struct NativeMenu {
    menu: Menu,
    id_new:     muda::MenuId,
    id_open:    muda::MenuId,
    id_save:    muda::MenuId,
    id_save_as: muda::MenuId,
    attached:   bool,
}

impl NativeMenu {
    /// Build the menu structure (no HWND needed yet).
    pub fn build() -> Self {
        let item_new = MenuItem::with_id(
            "file_new", "&New", true,
            Some(Accelerator::new(Modifiers::CONTROL, Code::KeyN)),
        );
        let item_open = MenuItem::with_id(
            "file_open", "&Open…", true,
            Some(Accelerator::new(Modifiers::CONTROL, Code::KeyO)),
        );
        let item_save = MenuItem::with_id(
            "file_save", "&Save", true,
            Some(Accelerator::new(Modifiers::CONTROL, Code::KeyS)),
        );
        let item_save_as = MenuItem::with_id(
            "file_save_as", "Save &As…", true,
            Some(Accelerator::new(Modifiers::CONTROL | Modifiers::SHIFT, Code::KeyS)),
        );

        let id_new     = item_new.id().clone();
        let id_open    = item_open.id().clone();
        let id_save    = item_save.id().clone();
        let id_save_as = item_save_as.id().clone();

        let file_menu = Submenu::with_id_and_items(
            "file_menu", "&File", true,
            &[
                &item_new,
                &item_open,
                &item_save,
                &item_save_as,
                &PredefinedMenuItem::separator(),
                &PredefinedMenuItem::quit(Some("E&xit")),
            ],
        ).expect("submenu");

        let menu = Menu::with_id_and_items(
            "main_menu",
            &[&file_menu],
        ).expect("menu");

        Self { menu, id_new, id_open, id_save, id_save_as, attached: false }
    }

    /// Attach to window by HWND (call once when the window is ready).
    ///
    /// # Safety
    /// `hwnd` must be a valid Win32 window handle for the lifetime of this menu.
    pub unsafe fn attach(&mut self, hwnd: isize) -> Result<(), muda::Error> {
        if !self.attached {
            unsafe { self.menu.init_for_hwnd(hwnd)?; }
            self.attached = true;
        }
        Ok(())
    }

    /// Drain pending menu events and return any FileCommand they map to.
    pub fn poll(&self) -> Option<FileCommand> {
        let rx = MenuEvent::receiver();
        while let Ok(ev) = rx.try_recv() {
            if ev.id == self.id_new     { return Some(FileCommand::New); }
            if ev.id == self.id_open    { return Some(FileCommand::Open); }
            if ev.id == self.id_save    { return Some(FileCommand::Save); }
            if ev.id == self.id_save_as { return Some(FileCommand::SaveAs); }
        }
        None
    }
}
