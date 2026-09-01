use build_rs::{input, output};

const EXTENSIONS: &[&str] = &["c", "h", "swift", "modulemap"];

fn main() {
    input::cargo_cfg_target_os();

    if cfg!(feature = "gl") {
        let mut lib = cmake::Config::new(".")
            .generator("Ninja")
            .build_target("cpge-native")
            .build();

        lib.push("build");

        for entry in walkdir::WalkDir::new("src").into_iter().flatten() {
            if entry.file_type().is_file() &&
                entry.path().extension().is_some_and(|ext| EXTENSIONS.iter().any(|x| ext == *x)) {
                output::rerun_if_changed(entry.path());
            }
        }

        output::rustc_link_search_kind("native", lib);
        output::rerun_if_changed("CMakeLists.txt");
    }

    output::rerun_if_changed("build.rs");
}
