//! Spawning children without a console window flashing on screen.
//!
//! Every child this shell starts — `nvidia-smi`, `pip`, `app.py` — is a
//! console subsystem program. On Windows, spawning one from a GUI process pops
//! a black console for as long as it runs; `pip install` would flash one for
//! several minutes. CREATE_NO_WINDOW suppresses it. The flag does not exist on
//! Unix, where nothing is shown in the first place.

use std::path::Path;
use std::process::Command;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub fn hidden_command(program: &Path) -> Command {
    // `mut` only on Windows: it is the sole platform that mutates the command
    // here, and binding it `mut` on Unix is an unused_mut warning.
    #[cfg(windows)]
    let mut command = Command::new(program);
    #[cfg(not(windows))]
    let command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

/// The same thing for a program looked up on PATH rather than by path.
pub fn hidden_program(program: &str) -> Command {
    hidden_command(Path::new(program))
}
