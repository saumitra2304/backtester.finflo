// src/lib.rs
pub mod backtester;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn run_backtest(config: *const c_char) -> *mut c_char {
    let c_str = unsafe {
        assert!(!config.is_null());
        CStr::from_ptr(config)
    };
    let r_str = c_str.to_str().unwrap();
    match backtester::backtesting(r_str) {
        Ok((equity_curve, trades)) => {
            let result = serde_json::to_string(&(equity_curve, trades)).unwrap();
            CString::new(result).unwrap().into_raw()
        }
        Err(e) => {
            let error_message = CString::new(format!("Error: {}", e)).unwrap();
            error_message.into_raw()
        }
    }
}

#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    unsafe {
        if s.is_null() { return }
        CString::from_raw(s)
    };
}
