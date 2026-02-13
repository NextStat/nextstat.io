use ns_root::RootFile;
use std::path::PathBuf;

fn hex_bytes(b: &[u8]) -> String {
    let mut s = String::new();
    for (i, x) in b.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{:02x}", x));
    }
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: dump_basket_tail <file.root> [tree] [branch]")?;
    let tree_name = args.next().unwrap_or_else(|| "events".to_string());
    let branch_name = args.next().unwrap_or_else(|| "jet_pt".to_string());

    let f = RootFile::open(&path)?;
    let tree = f.get_tree(&tree_name)?;
    let info = tree.find_branch(&branch_name).ok_or(format!("missing branch: {branch_name}"))?;

    eprintln!(
        "tree={} entries={} branch={} leaf_type={:?} entry_offset_len={} n_baskets={}",
        tree.name, tree.entries, info.name, info.leaf_type, info.entry_offset_len, info.n_baskets
    );
    eprintln!("basket_entry={:?}", info.basket_entry);
    eprintln!("basket_seek={:?}", info.basket_seek);

    if info.n_baskets == 0 {
        eprintln!("no baskets");
        return Ok(());
    }

    let file_bytes = std::fs::read(&path)?;
    let payload =
        ns_root::basket::read_basket_data(&file_bytes, info.basket_seek[0], f.is_large())?;
    let tail_len = 96usize.min(payload.len());
    let tail = &payload[payload.len() - tail_len..];
    eprintln!("payload_len={} tail_len={}", payload.len(), tail_len);
    eprintln!("tail_hex={}", hex_bytes(tail));

    Ok(())
}
