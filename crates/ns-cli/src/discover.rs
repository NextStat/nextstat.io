use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub(crate) fn find_combination_xmls(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk_dir_for_combination_xml(root, &mut out)?;
    out.sort();
    Ok(out)
}

pub(crate) fn discover_single_combination_xml(root: &Path) -> Result<PathBuf> {
    let found = find_combination_xmls(root)?;
    match found.len() {
        0 => bail!("No HistFactory `combination.xml` found under {}", root.display()),
        1 => Ok(found[0].clone()),
        _ => {
            let mut msg =
                format!("Found multiple HistFactory `combination.xml` under {}:\n", root.display());
            for p in &found {
                msg.push_str(&format!("  {}\n", p.display()));
            }
            bail!(msg.trim_end().to_string())
        }
    }
}

fn walk_dir_for_combination_xml(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let rd = fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))?;
    for entry in rd {
        let entry = entry.with_context(|| format!("iter dir {}", root.display()))?;
        let ft =
            entry.file_type().with_context(|| format!("file_type {}", entry.path().display()))?;

        // Avoid symlink loops when scanning arbitrary export trees.
        if ft.is_symlink() {
            continue;
        }

        let path = entry.path();
        if ft.is_dir() {
            walk_dir_for_combination_xml(&path, out)?;
            continue;
        }

        if ft.is_file()
            && let Some(name) = path.file_name().and_then(|s| s.to_str())
            && name == "combination.xml"
        {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        p.push(format!("ns-cli-{}-{}-{}", name, std::process::id(), nanos));
        p
    }

    fn rm_rf(path: &Path) {
        let _ = std::fs::remove_dir_all(path);
    }

    #[test]
    fn find_combination_xmls_finds_single() {
        let root = tmp_dir("find1");
        rm_rf(&root);
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("a/b/combination.xml"), "<xml/>").unwrap();

        let found = find_combination_xmls(&root).unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("a/b/combination.xml"));

        rm_rf(&root);
    }

    #[test]
    fn discover_single_combination_xml_errors_on_multiple() {
        let root = tmp_dir("find2");
        rm_rf(&root);
        std::fs::create_dir_all(root.join("x")).unwrap();
        std::fs::create_dir_all(root.join("y")).unwrap();
        std::fs::write(root.join("x/combination.xml"), "<xml/>").unwrap();
        std::fs::write(root.join("y/combination.xml"), "<xml/>").unwrap();

        let err = discover_single_combination_xml(&root).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("multiple"));
        assert!(msg.contains("x/combination.xml"));
        assert!(msg.contains("y/combination.xml"));

        rm_rf(&root);
    }
}
