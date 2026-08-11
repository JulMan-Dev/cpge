//! macOS window logic.

extern crate objc;

use objc::*;
use objc::runtime::Object;

unsafe extern "C" {
    fn cpge_init_application() -> i32;
}

pub struct MacOsApplication;

pub fn init_application() {
    unsafe { cpge_init_application(); }
}
