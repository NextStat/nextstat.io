use std::time::Instant;

fn main() {
    let path = "tests/fixtures/simple_tree.root";
    
    // 1. Открытие файла (mmap)
    let t0 = Instant::now();
    let file = ns_root::RootFile::open(path).unwrap();
    let t_open = t0.elapsed();
    
    // 2. Парсинг TTree metadata
    let t1 = Instant::now();
    let tree = file.get_tree("events").unwrap();
    let t_parse = t1.elapsed();
    
    println!("Branches: {:?}", tree.branch_names());
    println!("Entries:  {}", tree.entries);
    
    // 3. Чтение одной ветки (1000 f64)
    let t2 = Instant::now();
    let mbb = file.branch_data(&tree, "mbb").unwrap();
    let t_read = t2.elapsed();
    
    // 4. Чтение всех 7 веток
    let t3 = Instant::now();
    for name in tree.branch_names() {
        let _ = file.branch_data(&tree, &name).unwrap();
    }
    let t_all = t3.elapsed();
    
    // 5. Expression compile + eval
    let t4 = Instant::now();
    let expr = ns_root::CompiledExpr::compile("njet >= 4 && pt > 25.0").unwrap();
    let t_compile = t4.elapsed();
    
    let pt: Vec<f64> = file.branch_data(&tree, "pt").unwrap();
    let njet: Vec<f64> = file.branch_data(&tree, "njet").unwrap();
    let columns: Vec<&[f64]> = vec![&njet, &pt];
    
    let t5 = Instant::now();
    let mask = expr.eval_bulk(&columns);
    let t_eval = t5.elapsed();
    let selected = mask.iter().filter(|&&v| v > 0.0).count();
    
    // 6. Histogram fill
    let weight_mc: Vec<f64> = file.branch_data(&tree, "weight_mc").unwrap();
    let var_expr = ns_root::CompiledExpr::compile("mbb").unwrap();
    let wt_expr = ns_root::CompiledExpr::compile("weight_mc").unwrap();
    let sel_expr = ns_root::CompiledExpr::compile("njet >= 4 && pt > 25.0").unwrap();
    
    let spec = ns_root::HistogramSpec {
        name: "mbb".into(),
        variable: var_expr,
        weight: Some(wt_expr),
        selection: Some(sel_expr),
        bin_edges: vec![0., 50., 100., 150., 200., 300.],
    };
    
    let mut cols = std::collections::HashMap::new();
    cols.insert("mbb".into(), mbb);
    cols.insert("pt".into(), pt);
    cols.insert("njet".into(), njet);
    cols.insert("weight_mc".into(), weight_mc);
    
    let t6 = Instant::now();
    let histos = ns_root::fill_histograms(&[spec], &cols).unwrap();
    let t_fill = t6.elapsed();
    
    println!("\n--- Timing (1000 entries, 7 branches) ---");
    println!("File open (mmap):     {:>8.1}µs", t_open.as_nanos() as f64 / 1000.0);
    println!("TTree parse:          {:>8.1}µs", t_parse.as_nanos() as f64 / 1000.0);
    println!("Read 1 branch (f64):  {:>8.1}µs", t_read.as_nanos() as f64 / 1000.0);
    println!("Read all 7 branches:  {:>8.1}µs", t_all.as_nanos() as f64 / 1000.0);
    println!("Expr compile:         {:>8.1}µs", t_compile.as_nanos() as f64 / 1000.0);
    println!("Expr eval (1000 rows):{:>8.1}µs  ({} selected)", t_eval.as_nanos() as f64 / 1000.0, selected);
    println!("Histogram fill:       {:>8.1}µs", t_fill.as_nanos() as f64 / 1000.0);
    println!("\nHistogram: {:?}", histos[0].bin_content);
}
