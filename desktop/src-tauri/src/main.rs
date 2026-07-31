// A GUI app must not also open a console window on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// events and proc are the contracts later tasks build against, so they are
// compiled — and therefore checked — before anything calls them. Task 10
// removes this line and fixes whatever it was hiding.
#![allow(dead_code)]

mod events;
mod plan;
mod proc;
mod resources;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            .title("local-img")
            .inner_size(1180.0, 800.0)
            .min_inner_size(900.0, 600.0)
            .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running local-img");
}
