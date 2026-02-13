//! Expression engine for evaluating selections and weights from string
//! expressions over TTree branches.
//!
//! Supports arithmetic (+, -, *, /), comparisons (==, !=, <, <=, >, >=),
//! boolean operators (&&, ||, !), and built-in functions (abs, sqrt, log,
//! exp, pow, min, max).
//!
//! v1 additions:
//! - Ternary operator: `cond ? a : b` (right-associative).
//! - Span-aware errors with line/col.
//! - Bracket indexing:
//!   - Static: `branch[0]` is rewritten into a required scalar virtual branch name `branch[0]`
//!     (materialized by the ROOT reader as needed).
//!   - Dynamic: `branch[expr]` compiles to a jagged-load instruction and requires jagged
//!     branch data at evaluation time.

use crate::branch_reader::JaggedCol;
use crate::error::{Result, RootError};

// ── AST ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Expr {
    Number(f64),
    Var(usize), // index into required_branches
    UnaryNeg(Box<Expr>),
    UnaryNot(Box<Expr>),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
    Index {
        base: Box<Expr>,
        index: usize,
        span: Span,
    },
    /// Dynamic indexing: `branch[expr]` where expr is evaluated at runtime.
    #[allow(dead_code)]
    DynamicIndex {
        base: String,
        index: Box<Expr>,
        span: Span,
    },
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy)]
enum Func {
    Abs,
    Sqrt,
    Log,
    Log10,
    Exp,
    Sin,
    Cos,
    Pow,
    Atan2,
    Min,
    Max,
}

fn func_from_ident(name: &str) -> Option<Func> {
    // Accept TREx/ROOT spellings by stripping namespaces and lowercasing.
    //
    // Examples:
    // - abs(x)
    // - fabs(x)
    // - TMath::Abs(x)
    // - TMath::Power(x, 2)
    let leaf = name.rsplit("::").next().unwrap_or(name);
    let leaf = leaf.trim().to_ascii_lowercase();
    match leaf.as_str() {
        "abs" | "fabs" => Some(Func::Abs),
        "sqrt" => Some(Func::Sqrt),
        "log" => Some(Func::Log),
        "log10" => Some(Func::Log10),
        "exp" => Some(Func::Exp),
        "sin" => Some(Func::Sin),
        "cos" => Some(Func::Cos),
        "pow" | "power" => Some(Func::Pow),
        "atan2" => Some(Func::Atan2),
        "min" => Some(Func::Min),
        "max" => Some(Func::Max),
        _ => None,
    }
}

// ── Bytecode (vectorized evaluator) ────────────────────────────

#[derive(Debug, Clone)]
enum Instr {
    Const(f64),
    Load(usize),
    Neg,
    Not,
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Abs,
    Sqrt,
    Log,
    Log10,
    Exp,
    Sin,
    Cos,
    Pow,
    Atan2,
    Min,
    Max,
    /// Short-circuit AND for the vectorized path.
    ///
    /// Stack: [..., lhs] -> [..., result]
    ///
    /// The RHS bytecode immediately follows this instruction and has length `rhs_len`.
    /// The evaluator may skip RHS evaluation for lanes where `lhs` is falsey (== 0.0).
    AndSc {
        rhs_len: usize,
    },
    /// Short-circuit OR for the vectorized path.
    ///
    /// Stack: [..., lhs] -> [..., result]
    ///
    /// The RHS bytecode immediately follows this instruction and has length `rhs_len`.
    /// The evaluator may skip RHS evaluation for lanes where `lhs` is truthy (!= 0.0).
    OrSc {
        rhs_len: usize,
    },
    /// Mask-select: pops else, then, mask. result[i] = mask[i] > 0 ? then[i] : else[i].
    Select,
    /// Pop index from stack, load element from jagged branch #id. OOR → 0.0.
    /// Forces row-wise evaluation since index varies per row.
    DynLoad(usize),
    /// Jump to absolute instruction index if condition (popped) is == 0.0.
    /// Used for short-circuit boolean operators (&&/||) and DynLoad row-wise fallback.
    Jz(usize),
    /// Unconditional jump to absolute instruction index.
    /// Used for short-circuit boolean operators (&&/||) and DynLoad row-wise fallback.
    Jmp(usize),
}

// ── Compiled expression ────────────────────────────────────────

/// A compiled expression ready for evaluation.
///
/// Variable identifiers in the expression are mapped to branch names.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    #[allow(dead_code)]
    ast: Expr,
    bytecode: Vec<Instr>,
    /// Branch names referenced by this expression (ordered by first occurrence).
    pub required_branches: Vec<String>,
    /// Jagged branch names required for dynamic indexing (`branch[expr]`).
    /// The index in this vec corresponds to the `DynLoad(id)` instruction operand.
    pub required_jagged_branches: Vec<String>,
}

impl CompiledExpr {
    /// Parse and compile an expression string.
    ///
    /// Variable names in the expression correspond to TTree branch names.
    pub fn compile(input: &str) -> Result<Self> {
        let tokens = tokenize(input)?;
        let mut parser = Parser::new(input, &tokens);
        let mut ast = parser.parse_ternary()?;
        if parser.pos < parser.tokens.len() {
            let t = &parser.tokens[parser.pos];
            return Err(expr_err(
                input,
                t.span,
                format!("unexpected token after expression: {:?}", t.kind),
            ));
        }
        // Indexing is supported only as syntactic sugar for selecting a scalar "view" of
        // a branch, i.e. `jet_pt[0]` becomes a required branch named `jet_pt[0]`.
        //
        // The ROOT reader currently provides only scalar columns; vector-branch extraction
        // is handled elsewhere by materializing such `name[idx]` columns.
        rewrite_indexing(input, &mut ast, &mut parser.branches)?;
        prune_unused_branches(&mut ast, &mut parser.branches);
        let mut jagged_branches = Vec::new();
        let mut bytecode = Vec::new();
        let needs_dynload = expr_contains_dynamic_index(&ast);
        compile_bytecode(input, &ast, &mut bytecode, &mut jagged_branches, needs_dynload)?;
        let branches = std::mem::take(&mut parser.branches);
        Ok(CompiledExpr {
            ast,
            bytecode,
            required_branches: branches,
            required_jagged_branches: jagged_branches,
        })
    }

    /// Evaluate the expression for a single row.
    ///
    /// `values` must have the same length and order as `required_branches`.
    pub fn eval_row(&self, values: &[f64]) -> f64 {
        eval_bytecode_row(&self.bytecode, values)
    }

    /// Evaluate the expression for all rows (column-wise).
    ///
    /// `columns` must have the same length and order as `required_branches`;
    /// each column must have the same number of entries.
    pub fn eval_bulk(&self, columns: &[&[f64]]) -> Vec<f64> {
        if columns.is_empty() {
            // Constant expression — evaluate once
            return vec![eval_bytecode_row(&self.bytecode, &[])];
        }
        eval_bytecode_bulk(&self.bytecode, columns)
    }

    /// Evaluate the expression with jagged (variable-length) branch data.
    ///
    /// `columns` maps to `required_branches`, `jagged` maps to `required_jagged_branches`.
    /// DynLoad instructions use the jagged arrays for per-row indexed access.
    pub fn eval_bulk_with_jagged(&self, columns: &[&[f64]], jagged: &[&JaggedCol]) -> Vec<f64> {
        if columns.is_empty() && jagged.is_empty() {
            return vec![eval_bytecode_row(&self.bytecode, &[])];
        }
        eval_bytecode_bulk_with_jagged(&self.bytecode, columns, jagged)
    }

    /// Evaluate the expression in chunks to limit peak memory usage.
    ///
    /// For large datasets (>100M entries), vectorized evaluation allocates
    /// full-length intermediate vectors. This method processes `chunk_size`
    /// rows at a time, keeping peak memory bounded to ~`chunk_size * n_intermediates`.
    ///
    /// Results are bitwise-identical to `eval_bulk`.
    pub fn eval_bulk_chunked(&self, columns: &[&[f64]], chunk_size: usize) -> Vec<f64> {
        let n = columns.first().map(|c| c.len()).unwrap_or(0);
        if n <= chunk_size {
            return self.eval_bulk(columns);
        }

        let mut out = Vec::with_capacity(n);
        let mut offset = 0;
        while offset < n {
            let end = (offset + chunk_size).min(n);
            let chunk_cols: Vec<&[f64]> = columns.iter().map(|c| &c[offset..end]).collect();
            let chunk_result = eval_bytecode_bulk(&self.bytecode, &chunk_cols);
            out.extend_from_slice(&chunk_result);
            offset = end;
        }
        out
    }
}

/// Default chunk size for automatic chunked evaluation (64K rows).
pub const DEFAULT_CHUNK_SIZE: usize = 65536;

// ── Evaluation ─────────────────────────────────────────────────

#[cfg(test)]
fn eval_expr(e: &Expr, vals: &[f64]) -> f64 {
    match e {
        Expr::Number(n) => *n,
        Expr::Var(i) => vals[*i],
        Expr::UnaryNeg(a) => -eval_expr(a, vals),
        Expr::UnaryNot(a) => {
            // ROOT/TTreeFormula truthiness: non-zero is true (including negative and NaN).
            if eval_expr(a, vals) != 0.0 { 0.0 } else { 1.0 }
        }
        Expr::BinOp(op, a, b) => {
            let lhs = eval_expr(a, vals);
            match op {
                BinOp::Add => lhs + eval_expr(b, vals),
                BinOp::Sub => lhs - eval_expr(b, vals),
                BinOp::Mul => lhs * eval_expr(b, vals),
                BinOp::Div => lhs / eval_expr(b, vals),
                BinOp::Eq => {
                    let rhs = eval_expr(b, vals);
                    if (lhs - rhs).abs() < f64::EPSILON { 1.0 } else { 0.0 }
                }
                BinOp::Ne => {
                    let rhs = eval_expr(b, vals);
                    if (lhs - rhs).abs() >= f64::EPSILON { 1.0 } else { 0.0 }
                }
                BinOp::Lt => {
                    let rhs = eval_expr(b, vals);
                    if lhs < rhs { 1.0 } else { 0.0 }
                }
                BinOp::Le => {
                    let rhs = eval_expr(b, vals);
                    if lhs <= rhs { 1.0 } else { 0.0 }
                }
                BinOp::Gt => {
                    let rhs = eval_expr(b, vals);
                    if lhs > rhs { 1.0 } else { 0.0 }
                }
                BinOp::Ge => {
                    let rhs = eval_expr(b, vals);
                    if lhs >= rhs { 1.0 } else { 0.0 }
                }
                BinOp::And => {
                    if lhs != 0.0 {
                        if eval_expr(b, vals) != 0.0 { 1.0 } else { 0.0 }
                    } else {
                        0.0
                    }
                }
                BinOp::Or => {
                    if lhs != 0.0 || eval_expr(b, vals) != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        }
        Expr::Ternary(c, t, f) => {
            if eval_expr(c, vals) != 0.0 {
                eval_expr(t, vals)
            } else {
                eval_expr(f, vals)
            }
        }
        Expr::Call(f, args) => {
            let a0 = || eval_expr(&args[0], vals);
            let a1 = || eval_expr(&args[1], vals);
            match f {
                Func::Abs => a0().abs(),
                Func::Sqrt => a0().sqrt(),
                Func::Log => a0().ln(),
                Func::Log10 => a0().log10(),
                Func::Exp => a0().exp(),
                Func::Sin => a0().sin(),
                Func::Cos => a0().cos(),
                Func::Pow => a0().powf(a1()),
                Func::Atan2 => a0().atan2(a1()),
                Func::Min => a0().min(a1()),
                Func::Max => a0().max(a1()),
            }
        }
        Expr::Index { .. } | Expr::DynamicIndex { .. } => {
            // Rejected/rewritten in compile(), but keep evaluation total.
            f64::NAN
        }
    }
}

fn expr_contains_dynamic_index(e: &Expr) -> bool {
    match e {
        Expr::DynamicIndex { .. } => true,
        Expr::Number(_) | Expr::Var(_) => false,
        Expr::UnaryNeg(a) | Expr::UnaryNot(a) => expr_contains_dynamic_index(a),
        Expr::BinOp(_, a, b) => expr_contains_dynamic_index(a) || expr_contains_dynamic_index(b),
        Expr::Ternary(c, t, f) => {
            expr_contains_dynamic_index(c)
                || expr_contains_dynamic_index(t)
                || expr_contains_dynamic_index(f)
        }
        Expr::Call(_, args) => args.iter().any(expr_contains_dynamic_index),
        Expr::Index { base, .. } => expr_contains_dynamic_index(base),
    }
}

fn compile_bytecode(
    input: &str,
    e: &Expr,
    out: &mut Vec<Instr>,
    jagged: &mut Vec<String>,
    needs_dynload: bool,
) -> Result<()> {
    match e {
        Expr::Number(n) => out.push(Instr::Const(*n)),
        Expr::Var(i) => out.push(Instr::Load(*i)),
        Expr::UnaryNeg(a) => {
            compile_bytecode(input, a, out, jagged, needs_dynload)?;
            out.push(Instr::Neg);
        }
        Expr::UnaryNot(a) => {
            compile_bytecode(input, a, out, jagged, needs_dynload)?;
            out.push(Instr::Not);
        }
        Expr::BinOp(op, a, b) => {
            match op {
                BinOp::And => {
                    if needs_dynload {
                        // Short-circuit `a && b` for the row-wise path (DynLoad already forces it):
                        //   if (!a) return 0; else return !!b
                        compile_bytecode(input, a, out, jagged, needs_dynload)?;
                        let jz_pos = out.len();
                        out.push(Instr::Jz(usize::MAX));

                        compile_bytecode(input, b, out, jagged, needs_dynload)?;
                        out.push(Instr::Not);
                        out.push(Instr::Not);
                        let jmp_pos = out.len();
                        out.push(Instr::Jmp(usize::MAX));

                        let false_pos = out.len();
                        out.push(Instr::Const(0.0));
                        let end_pos = out.len();
                        out[jz_pos] = Instr::Jz(false_pos);
                        out[jmp_pos] = Instr::Jmp(end_pos);
                    } else {
                        compile_bytecode(input, a, out, jagged, needs_dynload)?;
                        let sc_pos = out.len();
                        out.push(Instr::AndSc { rhs_len: usize::MAX });
                        let rhs_start = out.len();
                        compile_bytecode(input, b, out, jagged, needs_dynload)?;
                        let rhs_len = out.len() - rhs_start;
                        out[sc_pos] = Instr::AndSc { rhs_len };
                    }
                }
                BinOp::Or => {
                    if needs_dynload {
                        // Short-circuit `a || b` for the row-wise path (DynLoad already forces it):
                        //   if (a) return 1; else return !!b
                        compile_bytecode(input, a, out, jagged, needs_dynload)?;
                        let jz_pos = out.len();
                        out.push(Instr::Jz(usize::MAX));

                        out.push(Instr::Const(1.0));
                        let jmp_pos = out.len();
                        out.push(Instr::Jmp(usize::MAX));

                        let eval_b_pos = out.len();
                        compile_bytecode(input, b, out, jagged, needs_dynload)?;
                        out.push(Instr::Not);
                        out.push(Instr::Not);

                        let end_pos = out.len();
                        out[jz_pos] = Instr::Jz(eval_b_pos);
                        out[jmp_pos] = Instr::Jmp(end_pos);
                    } else {
                        compile_bytecode(input, a, out, jagged, needs_dynload)?;
                        let sc_pos = out.len();
                        out.push(Instr::OrSc { rhs_len: usize::MAX });
                        let rhs_start = out.len();
                        compile_bytecode(input, b, out, jagged, needs_dynload)?;
                        let rhs_len = out.len() - rhs_start;
                        out[sc_pos] = Instr::OrSc { rhs_len };
                    }
                }
                _ => {
                    compile_bytecode(input, a, out, jagged, needs_dynload)?;
                    compile_bytecode(input, b, out, jagged, needs_dynload)?;
                    out.push(match op {
                        BinOp::Add => Instr::Add,
                        BinOp::Sub => Instr::Sub,
                        BinOp::Mul => Instr::Mul,
                        BinOp::Div => Instr::Div,
                        BinOp::Eq => Instr::Eq,
                        BinOp::Ne => Instr::Ne,
                        BinOp::Lt => Instr::Lt,
                        BinOp::Le => Instr::Le,
                        BinOp::Gt => Instr::Gt,
                        BinOp::Ge => Instr::Ge,
                        BinOp::And | BinOp::Or => unreachable!(),
                    });
                }
            }
        }
        Expr::Ternary(c, t, f) => {
            if needs_dynload {
                // Short-circuit ternary for the row-wise path (DynLoad already forces it):
                //   if (c) { t } else { f }
                compile_bytecode(input, c, out, jagged, needs_dynload)?;
                let jz_pos = out.len();
                out.push(Instr::Jz(usize::MAX));

                compile_bytecode(input, t, out, jagged, needs_dynload)?;
                let jmp_pos = out.len();
                out.push(Instr::Jmp(usize::MAX));

                let else_pos = out.len();
                compile_bytecode(input, f, out, jagged, needs_dynload)?;

                let end_pos = out.len();
                out[jz_pos] = Instr::Jz(else_pos);
                out[jmp_pos] = Instr::Jmp(end_pos);
            } else {
                compile_bytecode(input, c, out, jagged, needs_dynload)?; // mask
                compile_bytecode(input, t, out, jagged, needs_dynload)?; // then_val
                compile_bytecode(input, f, out, jagged, needs_dynload)?; // else_val
                out.push(Instr::Select);
            }
        }
        Expr::Call(func, args) => {
            for a in args {
                compile_bytecode(input, a, out, jagged, needs_dynload)?;
            }
            out.push(match func {
                Func::Abs => Instr::Abs,
                Func::Sqrt => Instr::Sqrt,
                Func::Log => Instr::Log,
                Func::Log10 => Instr::Log10,
                Func::Exp => Instr::Exp,
                Func::Sin => Instr::Sin,
                Func::Cos => Instr::Cos,
                Func::Pow => Instr::Pow,
                Func::Atan2 => Instr::Atan2,
                Func::Min => Instr::Min,
                Func::Max => Instr::Max,
            });
        }
        Expr::Index { span, .. } => {
            // Should be rejected by `rewrite_indexing`, but keep a clear error if it ever gets here.
            return Err(expr_err(input, *span, "vector branches are not supported".to_string()));
        }
        Expr::DynamicIndex { base, index, .. } => {
            compile_bytecode(input, index, out, jagged, needs_dynload)?; // push index value
            let id = jagged.iter().position(|s| s == base).unwrap_or_else(|| {
                jagged.push(base.clone());
                jagged.len() - 1
            });
            out.push(Instr::DynLoad(id));
        }
    }
    Ok(())
}

fn eval_bytecode_row(code: &[Instr], vars: &[f64]) -> f64 {
    let mut stack: Vec<f64> = Vec::with_capacity(16);
    let mut ip = 0usize;
    while ip < code.len() {
        match code[ip] {
            Instr::Const(v) => stack.push(v),
            Instr::Load(i) => {
                stack.push(vars[i]);
            }
            Instr::Neg => {
                let a = stack.pop().unwrap();
                stack.push(-a);
            }
            Instr::Not => {
                let a = stack.pop().unwrap();
                // ROOT/TTreeFormula truthiness: non-zero is true (including negative and NaN).
                stack.push(if a != 0.0 { 0.0 } else { 1.0 });
            }
            Instr::Add => bin2(&mut stack, |a, b| a + b),
            Instr::Sub => bin2(&mut stack, |a, b| a - b),
            Instr::Mul => bin2(&mut stack, |a, b| a * b),
            Instr::Div => bin2(&mut stack, |a, b| a / b),
            Instr::Eq => {
                bin2(&mut stack, |a, b| if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 })
            }
            Instr::Ne => {
                bin2(&mut stack, |a, b| if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 })
            }
            Instr::Lt => bin2(&mut stack, |a, b| if a < b { 1.0 } else { 0.0 }),
            Instr::Le => bin2(&mut stack, |a, b| if a <= b { 1.0 } else { 0.0 }),
            Instr::Gt => bin2(&mut stack, |a, b| if a > b { 1.0 } else { 0.0 }),
            Instr::Ge => bin2(&mut stack, |a, b| if a >= b { 1.0 } else { 0.0 }),
            Instr::Abs => {
                let a = stack.pop().unwrap();
                stack.push(a.abs());
            }
            Instr::Sqrt => {
                let a = stack.pop().unwrap();
                stack.push(a.sqrt());
            }
            Instr::Log => {
                let a = stack.pop().unwrap();
                stack.push(a.ln());
            }
            Instr::Log10 => {
                let a = stack.pop().unwrap();
                stack.push(a.log10());
            }
            Instr::Exp => {
                let a = stack.pop().unwrap();
                stack.push(a.exp());
            }
            Instr::Sin => {
                let a = stack.pop().unwrap();
                stack.push(a.sin());
            }
            Instr::Cos => {
                let a = stack.pop().unwrap();
                stack.push(a.cos());
            }
            Instr::Pow => bin2(&mut stack, |a, b| a.powf(b)),
            Instr::Atan2 => bin2(&mut stack, |a, b| a.atan2(b)),
            Instr::Min => bin2(&mut stack, |a, b| a.min(b)),
            Instr::Max => bin2(&mut stack, |a, b| a.max(b)),
            Instr::AndSc { rhs_len } => {
                let lhs = stack.pop().unwrap();
                // ROOT/TTreeFormula truthiness: non-zero is true (including NaN).
                if lhs == 0.0 {
                    stack.push(0.0);
                } else {
                    let rhs_start = ip + 1;
                    let rhs_end = rhs_start + rhs_len;
                    let rhs = eval_bytecode_row(&code[rhs_start..rhs_end], vars);
                    stack.push(if rhs != 0.0 { 1.0 } else { 0.0 });
                }
                ip += 1 + rhs_len;
                continue;
            }
            Instr::OrSc { rhs_len } => {
                let lhs = stack.pop().unwrap();
                if lhs != 0.0 {
                    stack.push(1.0);
                } else {
                    let rhs_start = ip + 1;
                    let rhs_end = rhs_start + rhs_len;
                    let rhs = eval_bytecode_row(&code[rhs_start..rhs_end], vars);
                    stack.push(if rhs != 0.0 { 1.0 } else { 0.0 });
                }
                ip += 1 + rhs_len;
                continue;
            }
            Instr::Select => {
                let else_val = stack.pop().unwrap();
                let then_val = stack.pop().unwrap();
                let mask = stack.pop().unwrap();
                stack.push(if mask != 0.0 { then_val } else { else_val });
            }
            Instr::DynLoad(_) => {
                // DynLoad requires jagged data context — not available in plain row eval.
                // Pop the index and push NaN as a sentinel.
                let _idx = stack.pop().unwrap();
                stack.push(f64::NAN);
            }
            Instr::Jz(target) => {
                let c = stack.pop().unwrap();
                // Jump if condition is falsey (zero). NaN is truthy in ROOT/TTreeFormula.
                if c == 0.0 {
                    ip = target;
                    continue;
                }
            }
            Instr::Jmp(target) => {
                ip = target;
                continue;
            }
        }
        ip += 1;
    }
    stack.pop().unwrap_or(f64::NAN)
}

fn bin2(stack: &mut Vec<f64>, f: impl FnOnce(f64, f64) -> f64) {
    let b = stack.pop().unwrap();
    let a = stack.pop().unwrap();
    stack.push(f(a, b));
}

fn eval_bytecode_bulk(code: &[Instr], cols: &[&[f64]]) -> Vec<f64> {
    // Vectorized path operates on whole columns (Scalar/Col/Owned slots).
    // Falls back to row-wise for expressions requiring DynLoad or explicit jumps.
    if bytecode_requires_rowwise(code) {
        eval_bytecode_bulk_rowwise(code, cols, &[])
    } else {
        eval_bytecode_bulk_vectorized(code, cols)
    }
}

fn eval_bytecode_bulk_with_jagged(
    code: &[Instr],
    cols: &[&[f64]],
    jagged: &[&JaggedCol],
) -> Vec<f64> {
    if bytecode_requires_rowwise(code) {
        eval_bytecode_bulk_rowwise(code, cols, jagged)
    } else {
        eval_bytecode_bulk_vectorized(code, cols)
    }
}

fn eval_bytecode_bulk_rowwise(code: &[Instr], cols: &[&[f64]], jagged: &[&JaggedCol]) -> Vec<f64> {
    let n = cols[0].len();
    let mut out = Vec::with_capacity(n);
    let mut stack: Vec<f64> = Vec::with_capacity(16);

    #[allow(clippy::needless_range_loop)]
    for row in 0..n {
        stack.clear();
        let mut ip = 0usize;
        while ip < code.len() {
            match code[ip] {
                Instr::Const(v) => stack.push(v),
                Instr::Load(i) => stack.push(cols[i][row]),
                Instr::Neg => {
                    let a = stack.pop().unwrap();
                    stack.push(-a);
                }
                Instr::Not => {
                    let a = stack.pop().unwrap();
                    stack.push(if a != 0.0 { 0.0 } else { 1.0 });
                }
                Instr::Add => bin2(&mut stack, |a, b| a + b),
                Instr::Sub => bin2(&mut stack, |a, b| a - b),
                Instr::Mul => bin2(&mut stack, |a, b| a * b),
                Instr::Div => bin2(&mut stack, |a, b| a / b),
                Instr::Eq => {
                    bin2(&mut stack, |a, b| if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 })
                }
                Instr::Ne => {
                    bin2(&mut stack, |a, b| if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 })
                }
                Instr::Lt => bin2(&mut stack, |a, b| if a < b { 1.0 } else { 0.0 }),
                Instr::Le => bin2(&mut stack, |a, b| if a <= b { 1.0 } else { 0.0 }),
                Instr::Gt => bin2(&mut stack, |a, b| if a > b { 1.0 } else { 0.0 }),
                Instr::Ge => bin2(&mut stack, |a, b| if a >= b { 1.0 } else { 0.0 }),
                Instr::Abs => {
                    let a = stack.pop().unwrap();
                    stack.push(a.abs());
                }
                Instr::Sqrt => {
                    let a = stack.pop().unwrap();
                    stack.push(a.sqrt());
                }
                Instr::Log => {
                    let a = stack.pop().unwrap();
                    stack.push(a.ln());
                }
                Instr::Log10 => {
                    let a = stack.pop().unwrap();
                    stack.push(a.log10());
                }
                Instr::Exp => {
                    let a = stack.pop().unwrap();
                    stack.push(a.exp());
                }
                Instr::Sin => {
                    let a = stack.pop().unwrap();
                    stack.push(a.sin());
                }
                Instr::Cos => {
                    let a = stack.pop().unwrap();
                    stack.push(a.cos());
                }
                Instr::Pow => bin2(&mut stack, |a, b| a.powf(b)),
                Instr::Atan2 => bin2(&mut stack, |a, b| a.atan2(b)),
                Instr::Min => bin2(&mut stack, |a, b| a.min(b)),
                Instr::Max => bin2(&mut stack, |a, b| a.max(b)),
                Instr::AndSc { rhs_len } => {
                    let lhs = stack.pop().unwrap();
                    if lhs == 0.0 {
                        stack.push(0.0);
                    } else {
                        let rhs_start = ip + 1;
                        let rhs_end = rhs_start + rhs_len;
                        // AndSc/OrSc are emitted only when DynLoad is absent (vectorizable program).
                        // Evaluate RHS for this single row using the row evaluator.
                        debug_assert!(!code[rhs_start..rhs_end].iter().any(|i| matches!(
                            i,
                            Instr::DynLoad(_) | Instr::Jz(_) | Instr::Jmp(_)
                        )));
                        let mut vars: Vec<f64> = Vec::with_capacity(cols.len());
                        for c in cols {
                            vars.push(c[row]);
                        }
                        let rhs = eval_bytecode_row(&code[rhs_start..rhs_end], &vars);
                        stack.push(if rhs != 0.0 { 1.0 } else { 0.0 });
                    }
                    ip += 1 + rhs_len;
                    continue;
                }
                Instr::OrSc { rhs_len } => {
                    let lhs = stack.pop().unwrap();
                    if lhs != 0.0 {
                        stack.push(1.0);
                    } else {
                        let rhs_start = ip + 1;
                        let rhs_end = rhs_start + rhs_len;
                        debug_assert!(!code[rhs_start..rhs_end].iter().any(|i| matches!(
                            i,
                            Instr::DynLoad(_) | Instr::Jz(_) | Instr::Jmp(_)
                        )));
                        let mut vars: Vec<f64> = Vec::with_capacity(cols.len());
                        for c in cols {
                            vars.push(c[row]);
                        }
                        let rhs = eval_bytecode_row(&code[rhs_start..rhs_end], &vars);
                        stack.push(if rhs != 0.0 { 1.0 } else { 0.0 });
                    }
                    ip += 1 + rhs_len;
                    continue;
                }
                Instr::Select => {
                    let else_val = stack.pop().unwrap();
                    let then_val = stack.pop().unwrap();
                    let mask = stack.pop().unwrap();
                    stack.push(if mask != 0.0 { then_val } else { else_val });
                }
                Instr::Jz(target) => {
                    let c = stack.pop().unwrap();
                    if c == 0.0 {
                        ip = target;
                        continue;
                    }
                }
                Instr::Jmp(target) => {
                    ip = target;
                    continue;
                }
                Instr::DynLoad(id) => {
                    let idx_f = stack.pop().unwrap();
                    let Some(j) = jagged.get(id) else {
                        // Missing jagged context (caller bug): keep NaN as a sentinel.
                        stack.push(f64::NAN);
                        continue;
                    };

                    // ROOT/TTreeFormula convention: out-of-range numeric access yields 0.0.
                    //
                    // For dynamic indices, treat negative and non-finite as out-of-range.
                    if !idx_f.is_finite() || idx_f < 0.0 {
                        stack.push(0.0);
                    } else {
                        let idx = idx_f as usize; // truncates toward zero for finite values
                        stack.push(j.get(row, idx, 0.0));
                    }
                }
            }
            ip += 1;
        }
        out.push(stack.pop().unwrap_or(f64::NAN));
    }

    out
}

fn bytecode_requires_rowwise(code: &[Instr]) -> bool {
    code.iter().any(|i| matches!(i, Instr::Jz(_) | Instr::Jmp(_) | Instr::DynLoad(_)))
}

/// Slot-based value for the vectorized bulk evaluator.
/// `Scalar` = broadcast constant, `Col` = borrowed column, `Owned` = arena index.
#[derive(Debug, Clone, Copy)]
enum Slot<'a> {
    Scalar(f64),
    Col(&'a [f64]),
    Owned(usize),
}

/// Evaluation state for the vectorized bytecode interpreter.
/// All mutable state lives in one struct so methods can borrow `&mut self` cleanly.
struct BulkEvalState<'a> {
    arena: Vec<Vec<f64>>,
    slots: Vec<Slot<'a>>,
    stack: Vec<usize>,
    n: usize,
}

// ── SIMD helpers (optional) ─────────────────────────────────────

#[cfg(target_arch = "aarch64")]
/// NEON SIMD helpers for vectorised element-wise arithmetic on `f64` slices.
///
/// # Safety
///
/// All public `unsafe fn` in this module require:
/// - `lhs` and `rhs` (when present) must be valid, non-overlapping slices.
/// - For two-operand functions, `lhs.len() == rhs.len()`.
/// - The target must support NEON (guaranteed on aarch64).
mod simd_arm {
    use std::arch::aarch64::*;

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn select_f64(
        mask: uint64x2_t,
        on_true: float64x2_t,
        on_false: float64x2_t,
    ) -> float64x2_t {
        vreinterpretq_f64_u64(vbslq_u64(
            mask,
            vreinterpretq_u64_f64(on_true),
            vreinterpretq_u64_f64(on_false),
        ))
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn is_nan_mask_f64(v: float64x2_t) -> uint64x2_t {
        // v != v iff NaN; vceqq is false for NaN, true otherwise.
        let not_nan = vceqq_f64(v, v);
        veorq_u64(not_nan, vdupq_n_u64(u64::MAX))
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn add_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vaddq_f64(a, b));
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x += y;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn sub_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vsubq_f64(a, b));
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x -= y;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn mul_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vmulq_f64(a, b));
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x *= y;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn div_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vdivq_f64(a, b));
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x /= y;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn add_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vaddq_f64(a, s));
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x += scalar;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn sub_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vsubq_f64(a, s));
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x -= scalar;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn mul_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vmulq_f64(a, s));
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x *= scalar;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn div_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vdivq_f64(a, s));
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x /= scalar;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn min_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            let min0 = vminq_f64(a, b);
            let a_nan = is_nan_mask_f64(a);
            let b_nan = is_nan_mask_f64(b);
            let r = select_f64(b_nan, a, select_f64(a_nan, b, min0));
            vst1q_f64(lhs.as_mut_ptr().add(i), r);
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x = (*x).min(y);
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn max_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            let max0 = vmaxq_f64(a, b);
            let a_nan = is_nan_mask_f64(a);
            let b_nan = is_nan_mask_f64(b);
            let r = select_f64(b_nan, a, select_f64(a_nan, b, max0));
            vst1q_f64(lhs.as_mut_ptr().add(i), r);
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x = (*x).max(y);
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn min_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        let s_nan = is_nan_mask_f64(s);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let min0 = vminq_f64(a, s);
            let a_nan = is_nan_mask_f64(a);
            let r = select_f64(s_nan, a, select_f64(a_nan, s, min0));
            vst1q_f64(lhs.as_mut_ptr().add(i), r);
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x = (*x).min(scalar);
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn max_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        let s_nan = is_nan_mask_f64(s);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let max0 = vmaxq_f64(a, s);
            let a_nan = is_nan_mask_f64(a);
            let r = select_f64(s_nan, a, select_f64(a_nan, s, max0));
            vst1q_f64(lhs.as_mut_ptr().add(i), r);
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x = (*x).max(scalar);
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn rsub_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vsubq_f64(s, a));
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x = scalar - *x;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn rdiv_scalar_inplace_neon(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = vdupq_n_f64(scalar);
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vdivq_f64(s, a));
            i += 2;
        }
        for x in &mut lhs[i..] {
            *x = scalar / *x;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn rsub_col_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vsubq_f64(b, a));
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x = y - *x;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn rdiv_col_inplace_neon(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 2 <= n {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            vst1q_f64(lhs.as_mut_ptr().add(i), vdivq_f64(b, a));
            i += 2;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x = y / *x;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use std::arch::x86_64::*;

    macro_rules! impl_avx_binop_inplace {
        ($name:ident, $op:ident, $scalar_fallback:expr) => {
            #[allow(unsafe_op_in_unsafe_fn)]
            #[target_feature(enable = "avx")]
            pub unsafe fn $name(lhs: &mut [f64], rhs: &[f64]) {
                let mut i = 0usize;
                let n = lhs.len();
                while i + 4 <= n {
                    let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
                    let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
                    _mm256_storeu_pd(lhs.as_mut_ptr().add(i), $op(a, b));
                    i += 4;
                }
                for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
                    *x = $scalar_fallback(*x, y);
                }
            }
        };
    }

    macro_rules! impl_avx_scalar_inplace {
        ($name:ident, $op:ident, $scalar_fallback:expr) => {
            #[allow(unsafe_op_in_unsafe_fn)]
            #[target_feature(enable = "avx")]
            pub unsafe fn $name(lhs: &mut [f64], scalar: f64) {
                let mut i = 0usize;
                let n = lhs.len();
                let s = _mm256_set1_pd(scalar);
                while i + 4 <= n {
                    let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
                    _mm256_storeu_pd(lhs.as_mut_ptr().add(i), $op(a, s));
                    i += 4;
                }
                for x in &mut lhs[i..] {
                    *x = $scalar_fallback(*x, scalar);
                }
            }
        };
    }

    macro_rules! impl_avx_rscalar_inplace {
        ($name:ident, $op:ident, $scalar_fallback:expr) => {
            #[allow(unsafe_op_in_unsafe_fn)]
            #[target_feature(enable = "avx")]
            pub unsafe fn $name(lhs: &mut [f64], scalar: f64) {
                let mut i = 0usize;
                let n = lhs.len();
                let s = _mm256_set1_pd(scalar);
                while i + 4 <= n {
                    let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
                    _mm256_storeu_pd(lhs.as_mut_ptr().add(i), $op(s, a));
                    i += 4;
                }
                for x in &mut lhs[i..] {
                    *x = $scalar_fallback(*x, scalar);
                }
            }
        };
    }

    macro_rules! impl_avx_rcol_inplace {
        ($name:ident, $op:ident, $scalar_fallback:expr) => {
            #[allow(unsafe_op_in_unsafe_fn)]
            #[target_feature(enable = "avx")]
            pub unsafe fn $name(lhs: &mut [f64], rhs: &[f64]) {
                let mut i = 0usize;
                let n = lhs.len();
                while i + 4 <= n {
                    let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
                    let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
                    _mm256_storeu_pd(lhs.as_mut_ptr().add(i), $op(b, a));
                    i += 4;
                }
                for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
                    *x = $scalar_fallback(*x, y);
                }
            }
        };
    }

    impl_avx_binop_inplace!(add_inplace_avx, _mm256_add_pd, |a, b| a + b);
    impl_avx_binop_inplace!(sub_inplace_avx, _mm256_sub_pd, |a, b| a - b);
    impl_avx_binop_inplace!(mul_inplace_avx, _mm256_mul_pd, |a, b| a * b);
    impl_avx_binop_inplace!(div_inplace_avx, _mm256_div_pd, |a, b| a / b);

    // f64::min/max semantics: if one operand is NaN, return the other operand.
    // AVX min/max intrinsics are asymmetric for NaNs, so we repair lanes via blends.
    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx")]
    pub unsafe fn min_inplace_avx(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 4 <= n {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
            let min0 = _mm256_min_pd(a, b);
            let a_nan = _mm256_cmp_pd(a, a, _CMP_UNORD_Q);
            let b_nan = _mm256_cmp_pd(b, b, _CMP_UNORD_Q);
            let r = _mm256_blendv_pd(_mm256_blendv_pd(min0, b, a_nan), a, b_nan);
            _mm256_storeu_pd(lhs.as_mut_ptr().add(i), r);
            i += 4;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x = (*x).min(y);
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx")]
    pub unsafe fn max_inplace_avx(lhs: &mut [f64], rhs: &[f64]) {
        let mut i = 0usize;
        let n = lhs.len();
        while i + 4 <= n {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
            let max0 = _mm256_max_pd(a, b);
            let a_nan = _mm256_cmp_pd(a, a, _CMP_UNORD_Q);
            let b_nan = _mm256_cmp_pd(b, b, _CMP_UNORD_Q);
            let r = _mm256_blendv_pd(_mm256_blendv_pd(max0, b, a_nan), a, b_nan);
            _mm256_storeu_pd(lhs.as_mut_ptr().add(i), r);
            i += 4;
        }
        for (x, &y) in lhs[i..].iter_mut().zip(&rhs[i..]) {
            *x = (*x).max(y);
        }
    }

    impl_avx_scalar_inplace!(add_scalar_inplace_avx, _mm256_add_pd, |a, s| a + s);
    impl_avx_scalar_inplace!(sub_scalar_inplace_avx, _mm256_sub_pd, |a, s| a - s);
    impl_avx_scalar_inplace!(mul_scalar_inplace_avx, _mm256_mul_pd, |a, s| a * s);
    impl_avx_scalar_inplace!(div_scalar_inplace_avx, _mm256_div_pd, |a, s| a / s);

    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx")]
    pub unsafe fn min_scalar_inplace_avx(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = _mm256_set1_pd(scalar);
        let s_nan = _mm256_cmp_pd(s, s, _CMP_UNORD_Q);
        while i + 4 <= n {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let min0 = _mm256_min_pd(a, s);
            let a_nan = _mm256_cmp_pd(a, a, _CMP_UNORD_Q);
            let r = _mm256_blendv_pd(_mm256_blendv_pd(min0, s, a_nan), a, s_nan);
            _mm256_storeu_pd(lhs.as_mut_ptr().add(i), r);
            i += 4;
        }
        for x in &mut lhs[i..] {
            *x = (*x).min(scalar);
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx")]
    pub unsafe fn max_scalar_inplace_avx(lhs: &mut [f64], scalar: f64) {
        let mut i = 0usize;
        let n = lhs.len();
        let s = _mm256_set1_pd(scalar);
        let s_nan = _mm256_cmp_pd(s, s, _CMP_UNORD_Q);
        while i + 4 <= n {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let max0 = _mm256_max_pd(a, s);
            let a_nan = _mm256_cmp_pd(a, a, _CMP_UNORD_Q);
            let r = _mm256_blendv_pd(_mm256_blendv_pd(max0, s, a_nan), a, s_nan);
            _mm256_storeu_pd(lhs.as_mut_ptr().add(i), r);
            i += 4;
        }
        for x in &mut lhs[i..] {
            *x = (*x).max(scalar);
        }
    }

    impl_avx_rscalar_inplace!(rsub_scalar_inplace_avx, _mm256_sub_pd, |a, s| s - a);
    impl_avx_rscalar_inplace!(rdiv_scalar_inplace_avx, _mm256_div_pd, |a, s| s / a);

    impl_avx_rcol_inplace!(rsub_col_inplace_avx, _mm256_sub_pd, |a, b| b - a);
    impl_avx_rcol_inplace!(rdiv_col_inplace_avx, _mm256_div_pd, |a, b| b / a);
}

#[cfg(target_arch = "aarch64")]
fn add_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    // SAFETY: aarch64 guarantees NEON.
    unsafe { simd_arm::add_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn add_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe { simd_x86::add_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x += y;
    }
}

#[cfg(target_arch = "aarch64")]
fn sub_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    unsafe { simd_arm::sub_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn sub_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::sub_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x -= y;
    }
}

#[cfg(target_arch = "aarch64")]
fn mul_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    unsafe { simd_arm::mul_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn mul_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::mul_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x *= y;
    }
}

#[cfg(target_arch = "aarch64")]
fn div_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    unsafe { simd_arm::div_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn div_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::div_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x /= y;
    }
}

#[cfg(target_arch = "aarch64")]
fn min_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    unsafe { simd_arm::min_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn min_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::min_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x = (*x).min(y);
    }
}

#[cfg(target_arch = "aarch64")]
fn max_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    unsafe { simd_arm::max_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn max_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::max_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x = (*x).max(y);
    }
}

#[cfg(target_arch = "aarch64")]
fn add_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::add_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn add_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::add_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x += scalar;
    }
}

#[cfg(target_arch = "aarch64")]
fn sub_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::sub_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn sub_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::sub_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x -= scalar;
    }
}

#[cfg(target_arch = "aarch64")]
fn mul_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::mul_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn mul_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::mul_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x *= scalar;
    }
}

#[cfg(target_arch = "aarch64")]
fn div_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::div_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn div_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::div_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x /= scalar;
    }
}

#[cfg(target_arch = "aarch64")]
fn min_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::min_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn min_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::min_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x = (*x).min(scalar);
    }
}

#[cfg(target_arch = "aarch64")]
fn max_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::max_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn max_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::max_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x = (*x).max(scalar);
    }
}

#[cfg(target_arch = "aarch64")]
fn rsub_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::rsub_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn rsub_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::rsub_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x = scalar - *x;
    }
}

#[cfg(target_arch = "aarch64")]
fn rdiv_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    unsafe { simd_arm::rdiv_scalar_inplace_neon(lhs, scalar) };
}
#[cfg(not(target_arch = "aarch64"))]
fn rdiv_scalar_inplace(lhs: &mut [f64], scalar: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::rdiv_scalar_inplace_avx(lhs, scalar) };
            return;
        }
    }
    for x in lhs.iter_mut() {
        *x = scalar / *x;
    }
}

#[cfg(target_arch = "aarch64")]
fn rsub_col_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    unsafe { simd_arm::rsub_col_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn rsub_col_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::rsub_col_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x = y - *x;
    }
}

#[cfg(target_arch = "aarch64")]
fn rdiv_col_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    unsafe { simd_arm::rdiv_col_inplace_neon(lhs, rhs) };
}
#[cfg(not(target_arch = "aarch64"))]
fn rdiv_col_inplace(lhs: &mut [f64], rhs: &[f64]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { simd_x86::rdiv_col_inplace_avx(lhs, rhs) };
            return;
        }
    }
    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
        *x = y / *x;
    }
}

impl<'a> BulkEvalState<'a> {
    fn new(n: usize) -> Self {
        Self { arena: Vec::new(), slots: Vec::with_capacity(16), stack: Vec::with_capacity(16), n }
    }

    fn push(&mut self, s: Slot<'a>) {
        self.slots.push(s);
        self.stack.push(self.slots.len() - 1);
    }

    fn pop(&mut self) -> Slot<'a> {
        let idx = self.stack.pop().unwrap();
        self.slots[idx]
    }

    fn neg(&mut self, v: Slot<'a>) -> Slot<'a> {
        match v {
            Slot::Scalar(a) => Slot::Scalar(-a),
            Slot::Col(c) => {
                let mut out = c.to_vec();
                mul_scalar_inplace(&mut out, -1.0);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            Slot::Owned(i) => {
                mul_scalar_inplace(&mut self.arena[i], -1.0);
                Slot::Owned(i)
            }
        }
    }

    fn add(&mut self, a: Slot<'a>, b: Slot<'a>) -> Slot<'a> {
        match (a, b) {
            (Slot::Scalar(x), Slot::Scalar(y)) => Slot::Scalar(x + y),
            (Slot::Owned(i), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Owned(i)) => {
                add_scalar_inplace(&mut self.arena[i], y);
                Slot::Owned(i)
            }
            (Slot::Owned(i), Slot::Col(c)) | (Slot::Col(c), Slot::Owned(i)) => {
                add_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(ca), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Col(ca)) => {
                let mut out = ca.to_vec();
                add_scalar_inplace(&mut out, y);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Col(ca), Slot::Col(cb)) => {
                let mut out = ca.to_vec();
                add_inplace(&mut out, cb);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Owned(i), Slot::Owned(j)) => {
                if i == j {
                    mul_scalar_inplace(&mut self.arena[i], 2.0);
                    return Slot::Owned(i);
                }
                if i < j {
                    let (left, right) = self.arena.split_at_mut(j);
                    let lhs = &mut left[i];
                    let rhs = &right[0];
                    add_inplace(lhs, rhs);
                    Slot::Owned(i)
                } else {
                    let (left, right) = self.arena.split_at_mut(i);
                    let rhs = &left[j];
                    let lhs = &mut right[0];
                    add_inplace(lhs, rhs);
                    Slot::Owned(i)
                }
            }
        }
    }

    fn sub(&mut self, a: Slot<'a>, b: Slot<'a>) -> Slot<'a> {
        match (a, b) {
            (Slot::Scalar(x), Slot::Scalar(y)) => Slot::Scalar(x - y),
            (Slot::Owned(i), Slot::Scalar(y)) => {
                sub_scalar_inplace(&mut self.arena[i], y);
                Slot::Owned(i)
            }
            (Slot::Scalar(x), Slot::Owned(i)) => {
                rsub_scalar_inplace(&mut self.arena[i], x);
                Slot::Owned(i)
            }
            (Slot::Owned(i), Slot::Col(c)) => {
                sub_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(c), Slot::Owned(i)) => {
                rsub_col_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(ca), Slot::Scalar(y)) => {
                let mut out = ca.to_vec();
                sub_scalar_inplace(&mut out, y);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Scalar(x), Slot::Col(cb)) => {
                let mut out = cb.to_vec();
                rsub_scalar_inplace(&mut out, x);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Col(ca), Slot::Col(cb)) => {
                let mut out = ca.to_vec();
                sub_inplace(&mut out, cb);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Owned(i), Slot::Owned(j)) => {
                if i == j {
                    self.arena[i].fill(0.0);
                    return Slot::Owned(i);
                }
                if i < j {
                    let (left, right) = self.arena.split_at_mut(j);
                    let lhs = &mut left[i];
                    let rhs = &right[0];
                    sub_inplace(lhs, rhs);
                    Slot::Owned(i)
                } else {
                    let (left, right) = self.arena.split_at_mut(i);
                    let rhs = &left[j];
                    let lhs = &mut right[0];
                    sub_inplace(lhs, rhs);
                    Slot::Owned(i)
                }
            }
        }
    }

    fn mul(&mut self, a: Slot<'a>, b: Slot<'a>) -> Slot<'a> {
        match (a, b) {
            (Slot::Scalar(x), Slot::Scalar(y)) => Slot::Scalar(x * y),
            (Slot::Owned(i), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Owned(i)) => {
                mul_scalar_inplace(&mut self.arena[i], y);
                Slot::Owned(i)
            }
            (Slot::Owned(i), Slot::Col(c)) | (Slot::Col(c), Slot::Owned(i)) => {
                mul_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(ca), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Col(ca)) => {
                let mut out = ca.to_vec();
                mul_scalar_inplace(&mut out, y);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Col(ca), Slot::Col(cb)) => {
                let mut out = ca.to_vec();
                mul_inplace(&mut out, cb);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Owned(i), Slot::Owned(j)) => {
                if i == j {
                    for x in self.arena[i].iter_mut() {
                        *x *= *x;
                    }
                    return Slot::Owned(i);
                }
                if i < j {
                    let (left, right) = self.arena.split_at_mut(j);
                    let lhs = &mut left[i];
                    let rhs = &right[0];
                    mul_inplace(lhs, rhs);
                    Slot::Owned(i)
                } else {
                    let (left, right) = self.arena.split_at_mut(i);
                    let rhs = &left[j];
                    let lhs = &mut right[0];
                    mul_inplace(lhs, rhs);
                    Slot::Owned(i)
                }
            }
        }
    }

    fn div(&mut self, a: Slot<'a>, b: Slot<'a>) -> Slot<'a> {
        match (a, b) {
            (Slot::Scalar(x), Slot::Scalar(y)) => Slot::Scalar(x / y),
            (Slot::Owned(i), Slot::Scalar(y)) => {
                div_scalar_inplace(&mut self.arena[i], y);
                Slot::Owned(i)
            }
            (Slot::Scalar(x), Slot::Owned(i)) => {
                rdiv_scalar_inplace(&mut self.arena[i], x);
                Slot::Owned(i)
            }
            (Slot::Owned(i), Slot::Col(c)) => {
                div_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(c), Slot::Owned(i)) => {
                rdiv_col_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(ca), Slot::Scalar(y)) => {
                let mut out = ca.to_vec();
                div_scalar_inplace(&mut out, y);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Scalar(x), Slot::Col(cb)) => {
                let mut out = cb.to_vec();
                rdiv_scalar_inplace(&mut out, x);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Col(ca), Slot::Col(cb)) => {
                let mut out = ca.to_vec();
                div_inplace(&mut out, cb);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Owned(i), Slot::Owned(j)) => {
                if i == j {
                    self.arena[i].fill(1.0);
                    return Slot::Owned(i);
                }
                if i < j {
                    let (left, right) = self.arena.split_at_mut(j);
                    let lhs = &mut left[i];
                    let rhs = &right[0];
                    div_inplace(lhs, rhs);
                    Slot::Owned(i)
                } else {
                    let (left, right) = self.arena.split_at_mut(i);
                    let rhs = &left[j];
                    let lhs = &mut right[0];
                    div_inplace(lhs, rhs);
                    Slot::Owned(i)
                }
            }
        }
    }

    fn min(&mut self, a: Slot<'a>, b: Slot<'a>) -> Slot<'a> {
        match (a, b) {
            (Slot::Scalar(x), Slot::Scalar(y)) => Slot::Scalar(x.min(y)),
            (Slot::Owned(i), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Owned(i)) => {
                min_scalar_inplace(&mut self.arena[i], y);
                Slot::Owned(i)
            }
            (Slot::Owned(i), Slot::Col(c)) | (Slot::Col(c), Slot::Owned(i)) => {
                min_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(ca), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Col(ca)) => {
                let mut out = ca.to_vec();
                min_scalar_inplace(&mut out, y);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Col(ca), Slot::Col(cb)) => {
                let mut out = ca.to_vec();
                min_inplace(&mut out, cb);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Owned(i), Slot::Owned(j)) => {
                if i == j {
                    return Slot::Owned(i);
                }
                if i < j {
                    let (left, right) = self.arena.split_at_mut(j);
                    let lhs = &mut left[i];
                    let rhs = &right[0];
                    min_inplace(lhs, rhs);
                    Slot::Owned(i)
                } else {
                    let (left, right) = self.arena.split_at_mut(i);
                    let rhs = &left[j];
                    let lhs = &mut right[0];
                    min_inplace(lhs, rhs);
                    Slot::Owned(i)
                }
            }
        }
    }

    fn max(&mut self, a: Slot<'a>, b: Slot<'a>) -> Slot<'a> {
        match (a, b) {
            (Slot::Scalar(x), Slot::Scalar(y)) => Slot::Scalar(x.max(y)),
            (Slot::Owned(i), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Owned(i)) => {
                max_scalar_inplace(&mut self.arena[i], y);
                Slot::Owned(i)
            }
            (Slot::Owned(i), Slot::Col(c)) | (Slot::Col(c), Slot::Owned(i)) => {
                max_inplace(&mut self.arena[i], c);
                Slot::Owned(i)
            }
            (Slot::Col(ca), Slot::Scalar(y)) | (Slot::Scalar(y), Slot::Col(ca)) => {
                let mut out = ca.to_vec();
                max_scalar_inplace(&mut out, y);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Col(ca), Slot::Col(cb)) => {
                let mut out = ca.to_vec();
                max_inplace(&mut out, cb);
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Owned(i), Slot::Owned(j)) => {
                if i == j {
                    return Slot::Owned(i);
                }
                if i < j {
                    let (left, right) = self.arena.split_at_mut(j);
                    let lhs = &mut left[i];
                    let rhs = &right[0];
                    max_inplace(lhs, rhs);
                    Slot::Owned(i)
                } else {
                    let (left, right) = self.arena.split_at_mut(i);
                    let rhs = &left[j];
                    let lhs = &mut right[0];
                    max_inplace(lhs, rhs);
                    Slot::Owned(i)
                }
            }
        }
    }

    fn unary<F>(&mut self, v: Slot<'a>, f: F) -> Slot<'a>
    where
        F: Fn(f64) -> f64 + Copy,
    {
        match v {
            Slot::Scalar(a) => Slot::Scalar(f(a)),
            Slot::Col(c) => {
                let mut out = Vec::with_capacity(self.n);
                out.extend(c.iter().map(|&x| f(x)));
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            Slot::Owned(i) => {
                for x in self.arena[i].iter_mut() {
                    *x = f(*x);
                }
                Slot::Owned(i)
            }
        }
    }

    fn binary<F>(&mut self, a: Slot<'a>, b: Slot<'a>, f: F) -> Slot<'a>
    where
        F: Fn(f64, f64) -> f64 + Copy,
    {
        match (a, b) {
            (Slot::Scalar(x), Slot::Scalar(y)) => Slot::Scalar(f(x, y)),
            (Slot::Owned(i), Slot::Scalar(y)) => {
                for x in self.arena[i].iter_mut() {
                    *x = f(*x, y);
                }
                Slot::Owned(i)
            }
            (Slot::Scalar(x), Slot::Owned(i)) => {
                for y in self.arena[i].iter_mut() {
                    *y = f(x, *y);
                }
                Slot::Owned(i)
            }
            (Slot::Owned(i), Slot::Col(c)) => {
                for (x, &y) in self.arena[i].iter_mut().zip(c.iter()) {
                    *x = f(*x, y);
                }
                Slot::Owned(i)
            }
            (Slot::Col(c), Slot::Owned(i)) => {
                for (&x, y) in c.iter().zip(self.arena[i].iter_mut()) {
                    *y = f(x, *y);
                }
                Slot::Owned(i)
            }
            (Slot::Col(ca), Slot::Scalar(y)) => {
                let mut out = Vec::with_capacity(self.n);
                out.extend(ca.iter().map(|&x| f(x, y)));
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Scalar(x), Slot::Col(cb)) => {
                let mut out = Vec::with_capacity(self.n);
                out.extend(cb.iter().map(|&y| f(x, y)));
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Col(ca), Slot::Col(cb)) => {
                let mut out = Vec::with_capacity(self.n);
                out.extend(ca.iter().zip(cb.iter()).map(|(&x, &y)| f(x, y)));
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
            (Slot::Owned(i), Slot::Owned(j)) => {
                if i == j {
                    for x in self.arena[i].iter_mut() {
                        *x = f(*x, *x);
                    }
                    return Slot::Owned(i);
                }
                if i < j {
                    let (left, right) = self.arena.split_at_mut(j);
                    let lhs = &mut left[i];
                    let rhs = &right[0];
                    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
                        *x = f(*x, y);
                    }
                    Slot::Owned(i)
                } else {
                    let (left, right) = self.arena.split_at_mut(i);
                    let rhs = &left[j];
                    let lhs = &mut right[0];
                    for (x, &y) in lhs.iter_mut().zip(rhs.iter()) {
                        *x = f(*x, y);
                    }
                    Slot::Owned(i)
                }
            }
        }
    }

    fn read_at(&self, slot: &Slot<'a>, i: usize) -> f64 {
        match *slot {
            Slot::Scalar(v) => v,
            Slot::Col(c) => c[i],
            Slot::Owned(idx) => self.arena[idx][i],
        }
    }

    fn select(&mut self, mask: Slot<'a>, then_val: Slot<'a>, else_val: Slot<'a>) -> Slot<'a> {
        match (mask, then_val, else_val) {
            (Slot::Scalar(m), Slot::Scalar(t), Slot::Scalar(e)) => {
                Slot::Scalar(if m != 0.0 { t } else { e })
            }
            _ => {
                let mut out = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let m = self.read_at(&mask, i);
                    let t = self.read_at(&then_val, i);
                    let e = self.read_at(&else_val, i);
                    out.push(if m != 0.0 { t } else { e });
                }
                self.arena.push(out);
                Slot::Owned(self.arena.len() - 1)
            }
        }
    }

    fn finish(mut self) -> Vec<f64> {
        let v = self.pop();
        match v {
            Slot::Scalar(x) => vec![x; self.n],
            Slot::Col(c) => c.to_vec(),
            Slot::Owned(i) => self.arena.swap_remove(i),
        }
    }
}

fn eval_bytecode_bulk_vectorized(code: &[Instr], cols: &[&[f64]]) -> Vec<f64> {
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    if n == 0 {
        return Vec::new();
    }

    // DynLoad / explicit jumps are not supported in the vectorized path.
    debug_assert!(!bytecode_requires_rowwise(code));

    let mut st = BulkEvalState::new(n);

    #[inline]
    fn boolize(x: f64) -> f64 {
        if x != 0.0 { 1.0 } else { 0.0 }
    }

    // Evaluate a sub-program on a subset of rows.
    fn eval_on_rows(code: &[Instr], cols: &[&[f64]], rows: &[usize]) -> Vec<f64> {
        debug_assert!(!bytecode_requires_rowwise(code));
        let k = rows.len();
        if k == 0 {
            return Vec::new();
        }
        if k == cols[0].len() {
            return eval_bytecode_bulk_vectorized(code, cols);
        }

        let mut subset_storage: Vec<Vec<f64>> = Vec::with_capacity(cols.len());
        for c in cols {
            let mut v = Vec::with_capacity(k);
            for &r in rows {
                v.push(c[r]);
            }
            subset_storage.push(v);
        }
        let subset_cols: Vec<&[f64]> = subset_storage.iter().map(|v| v.as_slice()).collect();
        eval_bytecode_bulk_vectorized(code, &subset_cols)
    }

    let mut ip = 0usize;
    while ip < code.len() {
        match code[ip] {
            Instr::Const(v) => st.push(Slot::Scalar(v)),
            Instr::Load(i) => st.push(Slot::Col(cols[i])),
            Instr::Neg => {
                let a = st.pop();
                let r = st.neg(a);
                st.push(r);
            }
            Instr::Not => {
                let a = st.pop();
                let r = st.unary(a, |x| if x != 0.0 { 0.0 } else { 1.0 });
                st.push(r);
            }
            Instr::Abs => {
                let a = st.pop();
                let r = st.unary(a, f64::abs);
                st.push(r);
            }
            Instr::Sqrt => {
                let a = st.pop();
                let r = st.unary(a, f64::sqrt);
                st.push(r);
            }
            Instr::Log => {
                let a = st.pop();
                let r = st.unary(a, f64::ln);
                st.push(r);
            }
            Instr::Log10 => {
                let a = st.pop();
                let r = st.unary(a, f64::log10);
                st.push(r);
            }
            Instr::Exp => {
                let a = st.pop();
                let r = st.unary(a, f64::exp);
                st.push(r);
            }
            Instr::Sin => {
                let a = st.pop();
                let r = st.unary(a, f64::sin);
                st.push(r);
            }
            Instr::Cos => {
                let a = st.pop();
                let r = st.unary(a, f64::cos);
                st.push(r);
            }
            Instr::Add => {
                let b = st.pop();
                let a = st.pop();
                let r = st.add(a, b);
                st.push(r);
            }
            Instr::Sub => {
                let b = st.pop();
                let a = st.pop();
                let r = st.sub(a, b);
                st.push(r);
            }
            Instr::Mul => {
                let b = st.pop();
                let a = st.pop();
                let r = st.mul(a, b);
                st.push(r);
            }
            Instr::Div => {
                let b = st.pop();
                let a = st.pop();
                let r = st.div(a, b);
                st.push(r);
            }
            Instr::Eq => {
                let b = st.pop();
                let a = st.pop();
                let r =
                    st.binary(a, b, |x, y| if (x - y).abs() < f64::EPSILON { 1.0 } else { 0.0 });
                st.push(r);
            }
            Instr::Ne => {
                let b = st.pop();
                let a = st.pop();
                let r =
                    st.binary(a, b, |x, y| if (x - y).abs() >= f64::EPSILON { 1.0 } else { 0.0 });
                st.push(r);
            }
            Instr::Lt => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| if x < y { 1.0 } else { 0.0 });
                st.push(r);
            }
            Instr::Le => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| if x <= y { 1.0 } else { 0.0 });
                st.push(r);
            }
            Instr::Gt => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| if x > y { 1.0 } else { 0.0 });
                st.push(r);
            }
            Instr::Ge => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| if x >= y { 1.0 } else { 0.0 });
                st.push(r);
            }
            Instr::Pow => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x.powf(y));
                st.push(r);
            }
            Instr::Atan2 => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x.atan2(y));
                st.push(r);
            }
            Instr::Min => {
                let b = st.pop();
                let a = st.pop();
                let r = st.min(a, b);
                st.push(r);
            }
            Instr::Max => {
                let b = st.pop();
                let a = st.pop();
                let r = st.max(a, b);
                st.push(r);
            }
            Instr::AndSc { rhs_len } => {
                let lhs = st.pop();

                let rhs_start = ip + 1;
                let rhs_end = rhs_start + rhs_len;
                let rhs_code = &code[rhs_start..rhs_end];

                match lhs {
                    Slot::Scalar(x) => {
                        if x == 0.0 {
                            st.push(Slot::Scalar(0.0));
                        } else {
                            let rhs = eval_bytecode_bulk_vectorized(rhs_code, cols);
                            let mut out = rhs;
                            for v in out.iter_mut() {
                                *v = boolize(*v);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        }
                    }
                    Slot::Col(c) => {
                        let mut rows: Vec<usize> = Vec::with_capacity(n / 2);
                        for (i, &v) in c.iter().enumerate() {
                            if v != 0.0 {
                                rows.push(i);
                            }
                        }
                        if rows.is_empty() {
                            st.push(Slot::Scalar(0.0));
                        } else if rows.len() == n {
                            let rhs = eval_bytecode_bulk_vectorized(rhs_code, cols);
                            let mut out = rhs;
                            for v in out.iter_mut() {
                                *v = boolize(*v);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        } else {
                            let rhs = eval_on_rows(rhs_code, cols, &rows);
                            let mut out = vec![0.0f64; n];
                            for (k, &row) in rows.iter().enumerate() {
                                out[row] = boolize(rhs[k]);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        }
                    }
                    Slot::Owned(idx) => {
                        let c = &st.arena[idx];
                        let mut rows: Vec<usize> = Vec::with_capacity(n / 2);
                        for (i, &v) in c.iter().enumerate() {
                            if v != 0.0 {
                                rows.push(i);
                            }
                        }
                        if rows.is_empty() {
                            st.push(Slot::Scalar(0.0));
                        } else if rows.len() == n {
                            let rhs = eval_bytecode_bulk_vectorized(rhs_code, cols);
                            let mut out = rhs;
                            for v in out.iter_mut() {
                                *v = boolize(*v);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        } else {
                            let rhs = eval_on_rows(rhs_code, cols, &rows);
                            let mut out = vec![0.0f64; n];
                            for (k, &row) in rows.iter().enumerate() {
                                out[row] = boolize(rhs[k]);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        }
                    }
                }

                ip += 1 + rhs_len;
                continue;
            }
            Instr::OrSc { rhs_len } => {
                let lhs = st.pop();

                let rhs_start = ip + 1;
                let rhs_end = rhs_start + rhs_len;
                let rhs_code = &code[rhs_start..rhs_end];

                match lhs {
                    Slot::Scalar(x) => {
                        if x != 0.0 {
                            st.push(Slot::Scalar(1.0));
                        } else {
                            let rhs = eval_bytecode_bulk_vectorized(rhs_code, cols);
                            let mut out = rhs;
                            for v in out.iter_mut() {
                                *v = boolize(*v);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        }
                    }
                    Slot::Col(c) => {
                        let mut rows: Vec<usize> = Vec::with_capacity(n / 2);
                        for (i, &v) in c.iter().enumerate() {
                            if v == 0.0 {
                                rows.push(i);
                            }
                        }
                        if rows.is_empty() {
                            st.push(Slot::Scalar(1.0));
                        } else if rows.len() == n {
                            let rhs = eval_bytecode_bulk_vectorized(rhs_code, cols);
                            let mut out = rhs;
                            for v in out.iter_mut() {
                                *v = boolize(*v);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        } else {
                            let rhs = eval_on_rows(rhs_code, cols, &rows);
                            let mut out = vec![1.0f64; n];
                            for (k, &row) in rows.iter().enumerate() {
                                out[row] = boolize(rhs[k]);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        }
                    }
                    Slot::Owned(idx) => {
                        let c = &st.arena[idx];
                        let mut rows: Vec<usize> = Vec::with_capacity(n / 2);
                        for (i, &v) in c.iter().enumerate() {
                            if v == 0.0 {
                                rows.push(i);
                            }
                        }
                        if rows.is_empty() {
                            st.push(Slot::Scalar(1.0));
                        } else if rows.len() == n {
                            let rhs = eval_bytecode_bulk_vectorized(rhs_code, cols);
                            let mut out = rhs;
                            for v in out.iter_mut() {
                                *v = boolize(*v);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        } else {
                            let rhs = eval_on_rows(rhs_code, cols, &rows);
                            let mut out = vec![1.0f64; n];
                            for (k, &row) in rows.iter().enumerate() {
                                out[row] = boolize(rhs[k]);
                            }
                            st.arena.push(out);
                            st.push(Slot::Owned(st.arena.len() - 1));
                        }
                    }
                }

                ip += 1 + rhs_len;
                continue;
            }
            Instr::Select => {
                let e = st.pop();
                let t = st.pop();
                let m = st.pop();
                let r = st.select(m, t, e);
                st.push(r);
            }
            Instr::Jz(_) | Instr::Jmp(_) | Instr::DynLoad(_) => {
                unreachable!("control flow / DynLoad handled by row-wise path")
            }
        }
        ip += 1;
    }

    st.finish()
}

// ── Tokenizer ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    Question,
    Colon,
    LBracket,
    RBracket,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Token {
    kind: TokenKind,
    span: Span,
}

fn line_col_1based(input: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(input.len());
    let bytes = input.as_bytes();
    let mut line: usize = 1;
    let mut col: usize = 1;
    for &b in &bytes[..offset] {
        if b == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn expr_err(input: &str, span: Span, msg: String) -> RootError {
    let (line, col) = line_col_1based(input, span.start);
    RootError::Expression(format!("line {line}, col {col}: {msg}"))
}

fn rewrite_indexing(input: &str, ast: &mut Expr, branches: &mut Vec<String>) -> Result<()> {
    fn walk(input: &str, e: &mut Expr, branches: &mut Vec<String>) -> Result<()> {
        match e {
            Expr::Index { base, index, span } => {
                // Rewrite base first so chained indexing (`x[0][1]`) becomes `Var` then `Var`.
                walk(input, base, branches)?;
                match &**base {
                    Expr::Var(i) => {
                        let name = branches.get(*i).ok_or_else(|| {
                            RootError::Expression(
                                "internal error: branch index out of bounds".to_string(),
                            )
                        })?;
                        let indexed = format!("{name}[{index}]");
                        let new_i =
                            branches.iter().position(|s| s == &indexed).unwrap_or_else(|| {
                                branches.push(indexed);
                                branches.len() - 1
                            });
                        *e = Expr::Var(new_i);
                        Ok(())
                    }
                    _ => Err(expr_err(
                        input,
                        *span,
                        "indexing is only supported on branch names (e.g. jet_pt[0])".to_string(),
                    )),
                }
            }
            Expr::Number(_) | Expr::Var(_) => Ok(()),
            Expr::UnaryNeg(a) | Expr::UnaryNot(a) => walk(input, a, branches),
            Expr::BinOp(_, a, b) => {
                walk(input, a, branches)?;
                walk(input, b, branches)
            }
            Expr::Ternary(c, t, f) => {
                walk(input, c, branches)?;
                walk(input, t, branches)?;
                walk(input, f, branches)
            }
            Expr::Call(_, args) => {
                for a in args {
                    walk(input, a, branches)?;
                }
                Ok(())
            }
            Expr::DynamicIndex { index, .. } => {
                // DynamicIndex bypasses rewrite_indexing — only recurse into the index expr.
                walk(input, index, branches)
            }
        }
    }

    walk(input, ast, branches)
}

fn prune_unused_branches(ast: &mut Expr, branches: &mut Vec<String>) {
    fn mark_used(e: &Expr, used: &mut Vec<bool>) {
        match e {
            Expr::Var(i) => {
                if let Some(v) = used.get_mut(*i) {
                    *v = true;
                }
            }
            Expr::Number(_) => {}
            Expr::UnaryNeg(a) | Expr::UnaryNot(a) => mark_used(a, used),
            Expr::BinOp(_, a, b) => {
                mark_used(a, used);
                mark_used(b, used);
            }
            Expr::Ternary(c, t, f) => {
                mark_used(c, used);
                mark_used(t, used);
                mark_used(f, used);
            }
            Expr::Call(_, args) => {
                for a in args {
                    mark_used(a, used);
                }
            }
            Expr::Index { .. } => {
                // `rewrite_indexing` should remove all Index nodes.
            }
            Expr::DynamicIndex { index, .. } => {
                // Recurse into the index expression (it may reference Var nodes).
                mark_used(index, used);
            }
        }
    }

    fn remap(e: &mut Expr, map: &[Option<usize>]) {
        match e {
            Expr::Var(i) => {
                if let Some(Some(new_i)) = map.get(*i) {
                    *i = *new_i;
                }
            }
            Expr::Number(_) => {}
            Expr::UnaryNeg(a) | Expr::UnaryNot(a) => remap(a, map),
            Expr::BinOp(_, a, b) => {
                remap(a, map);
                remap(b, map);
            }
            Expr::Ternary(c, t, f) => {
                remap(c, map);
                remap(t, map);
                remap(f, map);
            }
            Expr::Call(_, args) => {
                for a in args {
                    remap(a, map);
                }
            }
            Expr::Index { .. } => {}
            Expr::DynamicIndex { index, .. } => {
                // Remap Var references inside the index expression.
                remap(index, map);
            }
        }
    }

    let mut used = vec![false; branches.len()];
    mark_used(ast, &mut used);

    let mut map: Vec<Option<usize>> = vec![None; branches.len()];
    let mut new_branches: Vec<String> = Vec::new();
    for (i, (name, keep)) in branches.iter().cloned().zip(used.iter().copied()).enumerate() {
        if keep {
            map[i] = Some(new_branches.len());
            new_branches.push(name);
        }
    }

    remap(ast, &map);
    *branches = new_branches;
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    if !input.is_ascii() {
        if let Some((start, ch)) = input.char_indices().find(|(_, ch)| !ch.is_ascii()) {
            return Err(expr_err(
                input,
                Span { start, end: start + ch.len_utf8() },
                "expression must be ASCII (non-ASCII input is not supported yet)".to_string(),
            ));
        }
        return Err(RootError::Expression(
            "expression must be ASCII (non-ASCII input is not supported yet)".to_string(),
        ));
    }

    let bytes = input.as_bytes();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i: usize = 0;

    while i < bytes.len() {
        let b = bytes[i];

        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Two-character operators
        if i + 1 < bytes.len() {
            let two = &input[i..i + 2];
            let kind = match two {
                "&&" => Some(TokenKind::And),
                "||" => Some(TokenKind::Or),
                "==" => Some(TokenKind::Eq),
                "!=" => Some(TokenKind::Ne),
                "<=" => Some(TokenKind::Le),
                ">=" => Some(TokenKind::Ge),
                _ => None,
            };
            if let Some(kind) = kind {
                tokens.push(Token { kind, span: Span { start: i, end: i + 2 } });
                i += 2;
                continue;
            }
        }

        let start = i;
        let kind = match b {
            b'+' => {
                i += 1;
                TokenKind::Plus
            }
            b'-' => {
                i += 1;
                TokenKind::Minus
            }
            b'*' => {
                i += 1;
                TokenKind::Star
            }
            b'/' => {
                i += 1;
                TokenKind::Slash
            }
            b'(' => {
                i += 1;
                TokenKind::LParen
            }
            b')' => {
                i += 1;
                TokenKind::RParen
            }
            b',' => {
                i += 1;
                TokenKind::Comma
            }
            b'?' => {
                i += 1;
                TokenKind::Question
            }
            b':' => {
                i += 1;
                TokenKind::Colon
            }
            b'[' => {
                i += 1;
                TokenKind::LBracket
            }
            b']' => {
                i += 1;
                TokenKind::RBracket
            }
            b'<' => {
                i += 1;
                TokenKind::Lt
            }
            b'>' => {
                i += 1;
                TokenKind::Gt
            }
            b'!' => {
                i += 1;
                TokenKind::Not
            }
            _ if (b as char).is_ascii_digit() || b == b'.' => {
                i += 1;
                while i < bytes.len() {
                    let c = bytes[i] as char;
                    if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' {
                        i += 1;
                        continue;
                    }
                    if (c == '+' || c == '-')
                        && i > start
                        && (bytes[i - 1] == b'e' || bytes[i - 1] == b'E')
                    {
                        i += 1;
                        continue;
                    }
                    break;
                }
                let s = &input[start..i];
                let n: f64 = s.parse().map_err(|_| {
                    expr_err(input, Span { start, end: i }, format!("invalid number: '{s}'"))
                })?;
                TokenKind::Num(n)
            }
            _ if (b as char).is_ascii_alphabetic() || b == b'_' => {
                i += 1;
                while i < bytes.len() {
                    let c = bytes[i] as char;
                    if c.is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.' {
                        i += 1;
                        continue;
                    }
                    // Allow C++-style namespace qualifier inside identifiers (e.g. `TMath::Abs`).
                    if bytes[i] == b':' && i + 1 < bytes.len() && bytes[i + 1] == b':' {
                        i += 2;
                        continue;
                    }
                    break;
                }
                TokenKind::Ident(input[start..i].to_string())
            }
            _ => {
                return Err(expr_err(
                    input,
                    Span { start: i, end: i + 1 },
                    format!("unexpected character: '{}'", bytes[i] as char),
                ));
            }
        };

        tokens.push(Token { kind, span: Span { start, end: i } });
    }

    Ok(tokens)
}

// ── Parser (recursive descent) ─────────────────────────────────

struct Parser<'a> {
    input: &'a str,
    tokens: &'a [Token],
    pos: usize,
    branches: Vec<String>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str, tokens: &'a [Token]) -> Self {
        Self { input, tokens, pos: 0, branches: Vec::new() }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: TokenKind) -> Result<()> {
        match self.advance() {
            Some(t) if t.kind == expected => Ok(()),
            Some(t) => Err(expr_err(
                self.input,
                t.span,
                format!("expected {:?}, got {:?}", expected, t.kind),
            )),
            None => Err(expr_err(
                self.input,
                Span { start: self.input.len(), end: self.input.len() },
                format!("expected {:?}, got end of input", expected),
            )),
        }
    }

    fn resolve_var(&mut self, name: &str) -> usize {
        if let Some(i) = self.branches.iter().position(|b| b == name) {
            i
        } else {
            self.branches.push(name.to_string());
            self.branches.len() - 1
        }
    }

    // ── Grammar rules ──────────────────────────────────────────

    fn parse_ternary(&mut self) -> Result<Expr> {
        let cond = self.parse_or()?;
        if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Question)) {
            self.advance(); // '?'
            let then_expr = self.parse_ternary()?;
            self.expect(TokenKind::Colon)?;
            let else_expr = self.parse_ternary()?;
            Ok(Expr::Ternary(Box::new(cond), Box::new(then_expr), Box::new(else_expr)))
        } else {
            Ok(cond)
        }
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Or)) {
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::BinOp(BinOp::Or, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_cmp()?;
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::And)) {
            self.advance();
            let rhs = self.parse_cmp()?;
            lhs = Expr::BinOp(BinOp::And, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Expr> {
        let lhs = self.parse_add()?;
        let op = match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Eq) => BinOp::Eq,
            Some(TokenKind::Ne) => BinOp::Ne,
            Some(TokenKind::Lt) => BinOp::Lt,
            Some(TokenKind::Le) => BinOp::Le,
            Some(TokenKind::Gt) => BinOp::Gt,
            Some(TokenKind::Ge) => BinOp::Ge,
            _ => return Ok(lhs),
        };
        self.advance();
        let rhs = self.parse_add()?;
        Ok(Expr::BinOp(op, Box::new(lhs), Box::new(rhs)))
    }

    fn parse_add(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_mul()?;
        loop {
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::Plus) => {
                    self.advance();
                    let rhs = self.parse_mul()?;
                    lhs = Expr::BinOp(BinOp::Add, Box::new(lhs), Box::new(rhs));
                }
                Some(TokenKind::Minus) => {
                    self.advance();
                    let rhs = self.parse_mul()?;
                    lhs = Expr::BinOp(BinOp::Sub, Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::Star) => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    lhs = Expr::BinOp(BinOp::Mul, Box::new(lhs), Box::new(rhs));
                }
                Some(TokenKind::Slash) => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    lhs = Expr::BinOp(BinOp::Div, Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Minus) => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::UnaryNeg(Box::new(e)))
            }
            Some(TokenKind::Not) => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::UnaryNot(Box::new(e)))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut e = self.parse_atom()?;

        while let Some(TokenKind::LBracket) = self.peek().map(|t| &t.kind) {
            let lb = self.advance().unwrap();

            // Try static integer literal: peek at next two tokens for `Num(int) ]`
            let is_static =
                self.peek().is_some_and(
                    |t| matches!(&t.kind, TokenKind::Num(n) if n.fract() == 0.0 && *n >= 0.0),
                ) && self.tokens.get(self.pos + 1).is_some_and(|t| t.kind == TokenKind::RBracket);

            if is_static {
                let idx_tok = self.advance().unwrap();
                let index = match idx_tok.kind {
                    TokenKind::Num(n) => n as usize,
                    _ => unreachable!(),
                };
                let rb = self.advance().unwrap();
                let span = Span { start: lb.span.start, end: rb.span.end };
                e = Expr::Index { base: Box::new(e), index, span };
            } else {
                // Dynamic index: parse a full subexpression.
                // The base must be a variable name for DynamicIndex.
                let base_name = match &e {
                    Expr::Var(i) => self.branches.get(*i).cloned().ok_or_else(|| {
                        expr_err(
                            self.input,
                            lb.span,
                            "internal error: branch index out of bounds".to_string(),
                        )
                    })?,
                    _ => {
                        return Err(expr_err(
                            self.input,
                            lb.span,
                            "dynamic indexing is only supported on branch names (e.g. jet_pt[njet-1])".to_string(),
                        ));
                    }
                };

                let index_expr = self.parse_ternary()?;

                let rb = self
                    .advance()
                    .ok_or_else(|| expr_err(self.input, lb.span, "expected ']'".to_string()))?;
                if rb.kind != TokenKind::RBracket {
                    return Err(expr_err(
                        self.input,
                        rb.span,
                        format!("expected ']', got {:?}", rb.kind),
                    ));
                }
                let span = Span { start: lb.span.start, end: rb.span.end };
                e = Expr::DynamicIndex { base: base_name, index: Box::new(index_expr), span };
            }
        }

        Ok(e)
    }

    fn parse_atom(&mut self) -> Result<Expr> {
        match self.advance() {
            Some(Token { kind: TokenKind::Num(n), .. }) => Ok(Expr::Number(n)),
            Some(Token { kind: TokenKind::LParen, span, .. }) => {
                let e = self.parse_ternary()?;
                if let Some(t) = self.peek() {
                    if t.kind == TokenKind::RParen {
                        self.advance();
                        Ok(e)
                    } else {
                        Err(expr_err(self.input, t.span, format!("expected ')', got {:?}", t.kind)))
                    }
                } else {
                    Err(expr_err(self.input, span, "expected ')'".to_string()))
                }
            }
            Some(Token { kind: TokenKind::Ident(name), span, .. }) => {
                // Check for function call
                if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::LParen)) {
                    self.advance(); // consume '('
                    let Some(func) = func_from_ident(&name) else {
                        return Err(expr_err(
                            self.input,
                            span,
                            format!("unknown function: '{name}'"),
                        ));
                    };

                    let mut args = Vec::new();
                    if !matches!(self.peek().map(|t| &t.kind), Some(TokenKind::RParen)) {
                        args.push(self.parse_ternary()?);
                        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Comma)) {
                            self.advance();
                            args.push(self.parse_ternary()?);
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr::Call(func, args))
                } else {
                    // Variable reference
                    let idx = self.resolve_var(&name);
                    Ok(Expr::Var(idx))
                }
            }
            Some(Token { span, kind, .. }) => Err(expr_err(
                self.input,
                span,
                format!("expected number, identifier, or '(', got {:?}", kind),
            )),
            None => Err(expr_err(
                self.input,
                Span { start: self.input.len(), end: self.input.len() },
                "expected expression, got end of input".to_string(),
            )),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_next(state: &mut u64) -> u64 {
        // Numerical Recipes LCG (deterministic, good enough for tests)
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    fn rand_f64(state: &mut u64) -> f64 {
        let x = lcg_next(state) >> 11; // 53-ish bits
        let u = (x as f64) / ((1u64 << 53) as f64);
        // Map to [-5, 5]
        (u * 10.0) - 5.0
    }

    #[test]
    fn simple_arithmetic() {
        let e = CompiledExpr::compile("2 + 3 * 4").unwrap();
        assert!(e.required_branches.is_empty());
        assert!((e.eval_row(&[]) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn ternary_operator_basic() {
        let e = CompiledExpr::compile("x > 0 ? 10 : 20").unwrap();
        assert_eq!(e.required_branches, vec!["x"]);
        assert_eq!(e.eval_row(&[1.0]), 10.0);
        assert_eq!(e.eval_row(&[-1.0]), 20.0);
    }

    #[test]
    fn ternary_is_right_associative() {
        // x>0 ? 1 : (y>0 ? 2 : 3)
        let e = CompiledExpr::compile("x > 0 ? 1 : y > 0 ? 2 : 3").unwrap();
        assert_eq!(e.required_branches, vec!["x", "y"]);
        assert_eq!(e.eval_row(&[1.0, -1.0]), 1.0);
        assert_eq!(e.eval_row(&[-1.0, 1.0]), 2.0);
        assert_eq!(e.eval_row(&[-1.0, -1.0]), 3.0);
    }

    #[test]
    fn ternary_precedence_is_lower_than_or() {
        // Must parse as: (0 || 1) ? 3 : 4  => 3
        // If parsed as: 0 || (1 ? 3 : 4)  => 1
        let e = CompiledExpr::compile("0 || 1 ? 3 : 4").unwrap();
        assert_eq!(e.eval_row(&[]), 3.0);
    }

    #[test]
    fn variables() {
        let e = CompiledExpr::compile("pt * weight_mc").unwrap();
        assert_eq!(e.required_branches, vec!["pt", "weight_mc"]);
        assert!((e.eval_row(&[100.0, 0.5]) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn variable_names_can_contain_dots() {
        let e = CompiledExpr::compile("jet.pt + 1").unwrap();
        assert_eq!(e.required_branches, vec!["jet.pt"]);
        assert!((e.eval_row(&[2.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn comparison_and_boolean() {
        let e = CompiledExpr::compile("njet >= 4 && pt_lead > 25.0").unwrap();
        assert_eq!(e.required_branches, vec!["njet", "pt_lead"]);
        assert!((e.eval_row(&[4.0, 30.0]) - 1.0).abs() < 1e-10);
        assert!((e.eval_row(&[3.0, 30.0]) - 0.0).abs() < 1e-10);
        assert!((e.eval_row(&[4.0, 20.0]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn truthiness_matches_ttreeformula_nonzero_is_true() {
        // In ROOT/TTreeFormula, any non-zero numeric value is truthy (including negatives and NaN).
        // Ensure `!`, `&&`, `||`, and ternary conditions follow that rule.
        let e = CompiledExpr::compile("!!x").unwrap();
        assert_eq!(e.required_branches, vec!["x"]);
        assert_eq!(e.eval_row(&[0.0]), 0.0);
        assert_eq!(e.eval_row(&[1.0]), 1.0);
        assert_eq!(e.eval_row(&[-1.0]), 1.0);
        assert_eq!(e.eval_row(&[f64::NAN]), 1.0);

        let e = CompiledExpr::compile("!x").unwrap();
        assert_eq!(e.eval_row(&[0.0]), 1.0);
        assert_eq!(e.eval_row(&[2.0]), 0.0);
        assert_eq!(e.eval_row(&[-2.0]), 0.0);
        assert_eq!(e.eval_row(&[f64::NAN]), 0.0);

        let e = CompiledExpr::compile("x && 1").unwrap();
        assert_eq!(e.eval_row(&[0.0]), 0.0);
        assert_eq!(e.eval_row(&[2.0]), 1.0);
        assert_eq!(e.eval_row(&[-2.0]), 1.0);
        assert_eq!(e.eval_row(&[f64::NAN]), 1.0);

        let e = CompiledExpr::compile("x || 0").unwrap();
        assert_eq!(e.eval_row(&[0.0]), 0.0);
        assert_eq!(e.eval_row(&[2.0]), 1.0);
        assert_eq!(e.eval_row(&[-2.0]), 1.0);
        assert_eq!(e.eval_row(&[f64::NAN]), 1.0);

        let e = CompiledExpr::compile("x ? 10 : 20").unwrap();
        assert_eq!(e.eval_row(&[0.0]), 20.0);
        assert_eq!(e.eval_row(&[2.0]), 10.0);
        assert_eq!(e.eval_row(&[-2.0]), 10.0);
        assert_eq!(e.eval_row(&[f64::NAN]), 10.0);
    }

    #[test]
    fn functions() {
        let e = CompiledExpr::compile("sqrt(x)").unwrap();
        assert!((e.eval_row(&[9.0]) - 3.0).abs() < 1e-10);

        let e = CompiledExpr::compile("pow(x, 2)").unwrap();
        assert!((e.eval_row(&[3.0]) - 9.0).abs() < 1e-10);

        let e = CompiledExpr::compile("max(a, b)").unwrap();
        assert!((e.eval_row(&[3.0, 7.0]) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn root_function_aliases() {
        let e = CompiledExpr::compile("fabs(x)").unwrap();
        assert!((e.eval_row(&[-2.0]) - 2.0).abs() < 1e-10);
        let x = [-2.0, -1.0, 0.5];
        let got = e.eval_bulk(&[&x]);
        assert_eq!(got, vec![2.0, 1.0, 0.5]);

        let e = CompiledExpr::compile("TMath::Abs(x)").unwrap();
        assert!((e.eval_row(&[-3.0]) - 3.0).abs() < 1e-10);

        let e = CompiledExpr::compile("TMath::Power(x, 2)").unwrap();
        assert!((e.eval_row(&[4.0]) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn additional_math_functions() {
        let e = CompiledExpr::compile("log10(x)").unwrap();
        assert!((e.eval_row(&[100.0]) - 2.0).abs() < 1e-12);

        let e = CompiledExpr::compile("TMath::Log10(x)").unwrap();
        assert!((e.eval_row(&[1e3]) - 3.0).abs() < 1e-12);

        let e = CompiledExpr::compile("sin(x) + cos(x)").unwrap();
        let got = e.eval_row(&[0.0]);
        assert!((got - 1.0).abs() < 1e-12);

        let e = CompiledExpr::compile("atan2(y, x)").unwrap();
        assert!((e.eval_row(&[1.0, 0.0]) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        // Vectorized bulk path (no control flow).
        let e = CompiledExpr::compile("atan2(y, x) + log10(x)").unwrap();
        let x = [10.0, 100.0, 1000.0];
        let y = [0.0, 1.0, 1.0];
        let got = e.eval_bulk(&[&y, &x]);
        assert_eq!(got.len(), 3);
        assert!((got[0] - (0.0 + 1.0)).abs() < 1e-12);
        assert!((got[1] - (y[1].atan2(x[1]) + 2.0)).abs() < 1e-12);
        assert!((got[2] - (y[2].atan2(x[2]) + 3.0)).abs() < 1e-12);
    }

    #[test]
    fn negation() {
        let e = CompiledExpr::compile("-x + 1").unwrap();
        assert!((e.eval_row(&[5.0]) - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn logical_not() {
        let e = CompiledExpr::compile("!(x > 3)").unwrap();
        assert!((e.eval_row(&[2.0]) - 1.0).abs() < 1e-10);
        assert!((e.eval_row(&[5.0]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn bulk_eval() {
        let e = CompiledExpr::compile("a + b").unwrap();
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let result = e.eval_bulk(&[&a, &b]);
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn bulk_eval_matches_recursive_eval_for_random_inputs() {
        let exprs = [
            "a + b * 2",
            "sqrt(abs(a)) + max(a, b)",
            "a > 0 ? a : -a",
            "a > b ? (a - b) : (b - a)",
            "!(a > 0) || b < 0",
            "(a > 0 && b > 0) ? pow(a, 2) : min(a, b)",
        ];

        let n = 257usize;
        let mut state = 123456789u64;
        let mut a = vec![0.0f64; n];
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            a[i] = rand_f64(&mut state);
            b[i] = rand_f64(&mut state);
        }

        for src in exprs {
            let e = CompiledExpr::compile(src).unwrap();
            let mut eval_cols: Vec<&[f64]> = Vec::new();
            for name in &e.required_branches {
                match name.as_str() {
                    "a" => eval_cols.push(&a),
                    "b" => eval_cols.push(&b),
                    other => panic!("unexpected branch in test expr={src}: {other}"),
                }
            }
            let got = e.eval_bulk(&eval_cols);
            assert_eq!(got.len(), n, "len mismatch for expr={src}");

            for i in 0..n {
                let mut vars: Vec<f64> = Vec::with_capacity(e.required_branches.len());
                for name in &e.required_branches {
                    vars.push(match name.as_str() {
                        "a" => a[i],
                        "b" => b[i],
                        other => panic!("unexpected branch in test expr={src}: {other}"),
                    });
                }
                let want = eval_expr(&e.ast, &vars);
                let diff = (got[i] - want).abs();
                assert!(
                    diff < 1e-12 || (got[i].is_nan() && want.is_nan()),
                    "mismatch expr={src} i={i} got={} want={} diff={diff}",
                    got[i],
                    want
                );
            }
        }
    }

    #[test]
    fn or_expression() {
        let e = CompiledExpr::compile("x > 5 || y < 2").unwrap();
        assert!((e.eval_row(&[6.0, 3.0]) - 1.0).abs() < 1e-10);
        assert!((e.eval_row(&[3.0, 1.0]) - 1.0).abs() < 1e-10);
        assert!((e.eval_row(&[3.0, 3.0]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn nested_parens() {
        let e = CompiledExpr::compile("(1 + 2) * (3 + 4)").unwrap();
        assert!((e.eval_row(&[]) - 21.0).abs() < 1e-10);
    }

    #[test]
    fn scientific_notation() {
        let e = CompiledExpr::compile("1.5e2 + 3.0E-1").unwrap();
        assert!((e.eval_row(&[]) - 150.3).abs() < 1e-10);
    }

    #[test]
    fn indexing_is_rewritten_into_a_scalar_branch_name() {
        let e = CompiledExpr::compile("jet_pt[0] > 0").unwrap();
        assert_eq!(e.required_branches, vec!["jet_pt[0]".to_string()]);
    }

    #[test]
    fn ternary_uses_vectorized_path() {
        let e = CompiledExpr::compile("a > 0 ? a : -a").unwrap();
        // Ternary now compiles to Select, not Jz/Jmp → no control flow
        assert!(!bytecode_requires_rowwise(&e.bytecode));

        // Verify vectorized eval produces correct results
        let a = [-3.0, -1.0, 0.0, 1.0, 5.0];
        let got = e.eval_bulk(&[&a]);
        assert_eq!(got, vec![3.0, 1.0, 0.0, 1.0, 5.0]);
    }

    #[test]
    fn boolean_ops_use_vectorized_path_when_no_dynload() {
        let e = CompiledExpr::compile("(a > 0 && b > 0) || (a < 0 && b < 0)").unwrap();
        assert!(
            !bytecode_requires_rowwise(&e.bytecode),
            "&&/|| should not require row-wise path when DynLoad is absent, bytecode={:?}",
            e.bytecode
        );
        assert!(
            e.bytecode.iter().any(|i| matches!(i, Instr::AndSc { .. } | Instr::OrSc { .. })),
            "expected AndSc/OrSc in bytecode, got={:?}",
            e.bytecode
        );

        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let b = [-3.0, 2.0, 0.0, 3.0, -4.0];
        let got = e.eval_bulk(&[&a, &b]);

        // Same sign and non-zero => true, else false under ROOT truthiness rules.
        // (a>0 && b>0) OR (a<0 && b<0)
        assert_eq!(got, vec![1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn select_all_scalar_combination() {
        // All three operands constant → Scalar×3 path
        let e = CompiledExpr::compile("1 > 0 ? 42 : 99").unwrap();
        assert!(!bytecode_requires_rowwise(&e.bytecode));
        assert_eq!(e.eval_row(&[]), 42.0);

        let e = CompiledExpr::compile("0 > 1 ? 42 : 99").unwrap();
        assert_eq!(e.eval_row(&[]), 99.0);
    }

    #[test]
    fn select_mixed_col_scalar() {
        // mask=Col, then=Scalar, else=Scalar
        let e = CompiledExpr::compile("a > 0 ? 10 : 20").unwrap();
        assert!(!bytecode_requires_rowwise(&e.bytecode));
        let a = [1.0, -1.0, 2.0, -2.0];
        let got = e.eval_bulk(&[&a]);
        assert_eq!(got, vec![10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn select_col_col_col() {
        // mask=Col, then=Col, else=Col
        let e = CompiledExpr::compile("a > b ? a : b").unwrap();
        assert!(!bytecode_requires_rowwise(&e.bytecode));
        let a = [1.0, 5.0, 3.0];
        let b = [4.0, 2.0, 3.0];
        let got = e.eval_bulk(&[&a, &b]);
        // a>b: [false, true, false] → [4.0, 5.0, 3.0]
        assert_eq!(got, vec![4.0, 5.0, 3.0]);
    }

    #[test]
    fn chunked_matches_bulk() {
        let exprs = [
            "a + b * 2",
            "sqrt(abs(a)) + max(a, b)",
            "a > 0 ? a : -a",
            "(a > 0 && b > 0) ? pow(a, 2) : min(a, b)",
        ];

        let n = 200_000usize;
        let mut state = 987654321u64;
        let mut a = vec![0.0f64; n];
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            a[i] = rand_f64(&mut state);
            b[i] = rand_f64(&mut state);
        }

        for src in exprs {
            let e = CompiledExpr::compile(src).unwrap();
            let mut cols: Vec<&[f64]> = Vec::new();
            for name in &e.required_branches {
                match name.as_str() {
                    "a" => cols.push(&a),
                    "b" => cols.push(&b),
                    other => panic!("unexpected branch: {other}"),
                }
            }

            let full = e.eval_bulk(&cols);
            // Use a small chunk size (1000) to exercise boundary conditions
            let chunked = e.eval_bulk_chunked(&cols, 1000);

            assert_eq!(full.len(), chunked.len(), "length mismatch for {src}");
            for i in 0..full.len() {
                assert!(
                    full[i] == chunked[i] || (full[i].is_nan() && chunked[i].is_nan()),
                    "value mismatch at i={i} for {src}: full={} chunked={}",
                    full[i],
                    chunked[i]
                );
            }
        }
    }

    #[test]
    fn errors_report_line_col() {
        let err = CompiledExpr::compile("x +\n  * 2").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("line 2, col 3"), "unexpected error message: {msg}");
    }

    #[test]
    fn parse_dynamic_index() {
        let expr = CompiledExpr::compile("jet_pt[njet - 1]").unwrap();
        assert!(
            expr.required_jagged_branches.contains(&"jet_pt".to_string()),
            "expected jet_pt in required_jagged_branches: {:?}",
            expr.required_jagged_branches
        );
        assert!(
            expr.required_branches.contains(&"njet".to_string()),
            "expected njet in required_branches: {:?}",
            expr.required_branches
        );
        // DynLoad forces row-wise path
        assert!(
            bytecode_requires_rowwise(&expr.bytecode),
            "dynamic index should require row-wise path"
        );
    }

    #[test]
    fn dynload_and_or_use_short_circuit_bytecode() {
        use crate::branch_reader::JaggedCol;

        let expr = CompiledExpr::compile("njet > 0 && jet_pt[njet - 1] > 25").unwrap();
        assert!(
            expr.bytecode.iter().any(|i| matches!(i, Instr::Jz(_))),
            "expected Jz in bytecode when DynLoad is present: {:?}",
            expr.bytecode
        );
        assert!(
            expr.bytecode.iter().any(|i| matches!(i, Instr::Jmp(_))),
            "expected Jmp in bytecode when DynLoad is present: {:?}",
            expr.bytecode
        );

        // 4 entries; jet_pt is jagged: [10,30], [], [50], [20,100]
        let jet_pt =
            JaggedCol { flat: vec![10.0, 30.0, 50.0, 20.0, 100.0], offsets: vec![0, 2, 2, 3, 5] };
        let njet_col = vec![2.0, 0.0, 1.0, 2.0];

        let cols: Vec<&[f64]> = expr
            .required_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "njet");
                njet_col.as_slice()
            })
            .collect();
        let jagged_refs: Vec<&JaggedCol> = expr
            .required_jagged_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "jet_pt");
                &jet_pt
            })
            .collect();

        let got = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        assert_eq!(got, vec![1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn dynload_ternary_uses_short_circuit_bytecode() {
        use crate::branch_reader::JaggedCol;

        let expr = CompiledExpr::compile("njet > 0 ? jet_pt[njet - 1] : 999").unwrap();
        assert!(
            expr.bytecode.iter().any(|i| matches!(i, Instr::Jz(_))),
            "expected Jz in bytecode when DynLoad is present: {:?}",
            expr.bytecode
        );
        assert!(
            expr.bytecode.iter().any(|i| matches!(i, Instr::Jmp(_))),
            "expected Jmp in bytecode when DynLoad is present: {:?}",
            expr.bytecode
        );
        assert!(
            !expr.bytecode.iter().any(|i| matches!(i, Instr::Select)),
            "ternary should not compile to Select when DynLoad is present: {:?}",
            expr.bytecode
        );

        let jet_pt =
            JaggedCol { flat: vec![10.0, 30.0, 50.0, 20.0, 100.0], offsets: vec![0, 2, 2, 3, 5] };
        let njet_col = vec![2.0, 0.0, 1.0, 2.0];

        let cols: Vec<&[f64]> = expr
            .required_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "njet");
                njet_col.as_slice()
            })
            .collect();
        let jagged_refs: Vec<&JaggedCol> = expr
            .required_jagged_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "jet_pt");
                &jet_pt
            })
            .collect();

        let got = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        assert_eq!(got, vec![30.0, 999.0, 50.0, 100.0]);
    }

    #[test]
    fn static_index_still_works() {
        // jet_pt[0] should remain static (Expr::Index → Var rewrite)
        let expr = CompiledExpr::compile("jet_pt[0]").unwrap();
        assert!(
            expr.required_jagged_branches.is_empty(),
            "static index should not create jagged branches"
        );
        assert!(
            expr.required_branches.contains(&"jet_pt[0]".to_string()),
            "expected jet_pt[0] in required_branches: {:?}",
            expr.required_branches
        );
    }

    #[test]
    fn dynamic_index_eval_with_jagged() {
        use crate::branch_reader::JaggedCol;

        // jet_pt[idx] where idx comes from a scalar column
        let expr = CompiledExpr::compile("jet_pt[idx]").unwrap();

        // 4 entries; jet_pt is jagged: [10,20], [30], [40,50,60], []
        let jagged = JaggedCol {
            flat: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            offsets: vec![0, 2, 3, 6, 6],
        };

        // idx = [1, 0, 2, 0] → expect [20, 30, 60, 0(OOR)]
        let idx_col = vec![1.0, 0.0, 2.0, 0.0];

        let cols: Vec<&[f64]> = expr
            .required_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "idx");
                idx_col.as_slice()
            })
            .collect();

        let jagged_refs: Vec<&JaggedCol> = expr
            .required_jagged_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "jet_pt");
                &jagged
            })
            .collect();

        let result = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        assert_eq!(result, vec![20.0, 30.0, 60.0, 0.0]);
    }

    #[test]
    fn dynamic_index_negative_to_oor() {
        use crate::branch_reader::JaggedCol;

        let expr = CompiledExpr::compile("x[idx]").unwrap();

        let jagged = JaggedCol { flat: vec![100.0, 200.0], offsets: vec![0, 2] };

        // Negative index should behave like out-of-range (ROOT/TTreeFormula numeric convention).
        let idx_col = vec![-1.0];
        let cols: Vec<&[f64]> = expr.required_branches.iter().map(|_| idx_col.as_slice()).collect();
        let jagged_refs: Vec<&JaggedCol> =
            expr.required_jagged_branches.iter().map(|_| &jagged).collect();

        let result = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn dynamic_index_nan_to_oor() {
        use crate::branch_reader::JaggedCol;

        let expr = CompiledExpr::compile("x[idx]").unwrap();

        let jagged = JaggedCol { flat: vec![100.0, 200.0], offsets: vec![0, 2] };

        let idx_col = vec![f64::NAN];
        let cols: Vec<&[f64]> = expr.required_branches.iter().map(|_| idx_col.as_slice()).collect();
        let jagged_refs: Vec<&JaggedCol> =
            expr.required_jagged_branches.iter().map(|_| &jagged).collect();

        let result = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn dynamic_index_float_truncates() {
        use crate::branch_reader::JaggedCol;

        let expr = CompiledExpr::compile("x[idx]").unwrap();

        let jagged = JaggedCol { flat: vec![10.0, 20.0, 30.0], offsets: vec![0, 3] };

        // 1.7 → truncated to 1 → x[1] = 20.0
        let idx_col = vec![1.7];
        let cols: Vec<&[f64]> = expr.required_branches.iter().map(|_| idx_col.as_slice()).collect();
        let jagged_refs: Vec<&JaggedCol> =
            expr.required_jagged_branches.iter().map(|_| &jagged).collect();

        let result = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        assert_eq!(result, vec![20.0]);
    }

    #[test]
    fn dynamic_index_expression() {
        use crate::branch_reader::JaggedCol;

        // jet_pt[njet - 1]: index by expression
        let expr = CompiledExpr::compile("jet_pt[njet - 1]").unwrap();

        // 3 entries; jet_pt: [10,20,30], [40,50], [60]
        let jagged =
            JaggedCol { flat: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], offsets: vec![0, 3, 5, 6] };

        // njet = [3, 2, 1] → index = [2, 1, 0] → expect [30, 50, 60]
        let njet_col = vec![3.0, 2.0, 1.0];

        let cols: Vec<&[f64]> = expr
            .required_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "njet");
                njet_col.as_slice()
            })
            .collect();

        let jagged_refs: Vec<&JaggedCol> = expr
            .required_jagged_branches
            .iter()
            .map(|name| {
                assert_eq!(name, "jet_pt");
                &jagged
            })
            .collect();

        let result = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        assert_eq!(result, vec![30.0, 50.0, 60.0]);
    }
}
