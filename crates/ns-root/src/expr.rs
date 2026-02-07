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
//! - Parsed bracket indexing `x[0]` (currently rejected with a clear error;
//!   vector branches are not supported by the ROOT reader yet).

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
    Index { base: Box<Expr>, index: usize, span: Span },
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
    Exp,
    Pow,
    Min,
    Max,
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
    And,
    Or,
    Abs,
    Sqrt,
    Log,
    Exp,
    Pow,
    Min,
    Max,
    /// Jump to absolute instruction index if condition (popped) is <= 0.
    Jz(usize),
    /// Unconditional jump to absolute instruction index.
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
            return Err(expr_err(input, t.span, format!("unexpected token after expression: {:?}", t.kind)));
        }
        // Indexing is supported only as syntactic sugar for selecting a scalar "view" of
        // a branch, i.e. `jet_pt[0]` becomes a required branch named `jet_pt[0]`.
        //
        // The ROOT reader currently provides only scalar columns; vector-branch extraction
        // is handled elsewhere by materializing such `name[idx]` columns.
        rewrite_indexing(input, &mut ast, &mut parser.branches)?;
        prune_unused_branches(&mut ast, &mut parser.branches);
        let mut bytecode = Vec::new();
        compile_bytecode(input, &ast, &mut bytecode)?;
        let branches = std::mem::take(&mut parser.branches);
        Ok(CompiledExpr { ast, bytecode, required_branches: branches })
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
}

// ── Evaluation ─────────────────────────────────────────────────

#[cfg(test)]
fn eval_expr(e: &Expr, vals: &[f64]) -> f64 {
    match e {
        Expr::Number(n) => *n,
        Expr::Var(i) => vals[*i],
        Expr::UnaryNeg(a) => -eval_expr(a, vals),
        Expr::UnaryNot(a) => {
            if eval_expr(a, vals) > 0.0 {
                0.0
            } else {
                1.0
            }
        }
        Expr::BinOp(op, a, b) => {
            let lhs = eval_expr(a, vals);
            let rhs = eval_expr(b, vals);
            match op {
                BinOp::Add => lhs + rhs,
                BinOp::Sub => lhs - rhs,
                BinOp::Mul => lhs * rhs,
                BinOp::Div => lhs / rhs,
                BinOp::Eq => {
                    if (lhs - rhs).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Ne => {
                    if (lhs - rhs).abs() >= f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Lt => {
                    if lhs < rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Le => {
                    if lhs <= rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Gt => {
                    if lhs > rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Ge => {
                    if lhs >= rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::And => {
                    if lhs > 0.0 && rhs > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinOp::Or => {
                    if lhs > 0.0 || rhs > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        }
        Expr::Ternary(c, t, f) => {
            if eval_expr(c, vals) > 0.0 { eval_expr(t, vals) } else { eval_expr(f, vals) }
        }
        Expr::Call(f, args) => {
            let a0 = || eval_expr(&args[0], vals);
            let a1 = || eval_expr(&args[1], vals);
            match f {
                Func::Abs => a0().abs(),
                Func::Sqrt => a0().sqrt(),
                Func::Log => a0().ln(),
                Func::Exp => a0().exp(),
                Func::Pow => a0().powf(a1()),
                Func::Min => a0().min(a1()),
                Func::Max => a0().max(a1()),
            }
        }
        Expr::Index { .. } => {
            // Rejected in compile(), but keep evaluation total.
            f64::NAN
        }
    }
}

fn compile_bytecode(input: &str, e: &Expr, out: &mut Vec<Instr>) -> Result<()> {
    match e {
        Expr::Number(n) => out.push(Instr::Const(*n)),
        Expr::Var(i) => out.push(Instr::Load(*i)),
        Expr::UnaryNeg(a) => {
            compile_bytecode(input, a, out)?;
            out.push(Instr::Neg);
        }
        Expr::UnaryNot(a) => {
            compile_bytecode(input, a, out)?;
            out.push(Instr::Not);
        }
        Expr::BinOp(op, a, b) => {
            compile_bytecode(input, a, out)?;
            compile_bytecode(input, b, out)?;
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
                BinOp::And => Instr::And,
                BinOp::Or => Instr::Or,
            });
        }
        Expr::Ternary(c, t, f) => {
            compile_bytecode(input, c, out)?;
            let jz_pos = out.len();
            out.push(Instr::Jz(usize::MAX));
            compile_bytecode(input, t, out)?;
            let jmp_pos = out.len();
            out.push(Instr::Jmp(usize::MAX));
            let else_ip = out.len();
            out[jz_pos] = Instr::Jz(else_ip);
            compile_bytecode(input, f, out)?;
            let end_ip = out.len();
            out[jmp_pos] = Instr::Jmp(end_ip);
        }
        Expr::Call(func, args) => {
            for a in args {
                compile_bytecode(input, a, out)?;
            }
            out.push(match func {
                Func::Abs => Instr::Abs,
                Func::Sqrt => Instr::Sqrt,
                Func::Log => Instr::Log,
                Func::Exp => Instr::Exp,
                Func::Pow => Instr::Pow,
                Func::Min => Instr::Min,
                Func::Max => Instr::Max,
            });
        }
        Expr::Index { span, .. } => {
            // Should be rejected by `reject_indexing`, but keep a clear error if it ever gets here.
            return Err(expr_err(input, *span, "vector branches are not supported".to_string()));
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
                stack.push(if a > 0.0 { 0.0 } else { 1.0 });
            }
            Instr::Add => bin2(&mut stack, |a, b| a + b),
            Instr::Sub => bin2(&mut stack, |a, b| a - b),
            Instr::Mul => bin2(&mut stack, |a, b| a * b),
            Instr::Div => bin2(&mut stack, |a, b| a / b),
            Instr::Eq => bin2(&mut stack, |a, b| if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 }),
            Instr::Ne => bin2(&mut stack, |a, b| if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 }),
            Instr::Lt => bin2(&mut stack, |a, b| if a < b { 1.0 } else { 0.0 }),
            Instr::Le => bin2(&mut stack, |a, b| if a <= b { 1.0 } else { 0.0 }),
            Instr::Gt => bin2(&mut stack, |a, b| if a > b { 1.0 } else { 0.0 }),
            Instr::Ge => bin2(&mut stack, |a, b| if a >= b { 1.0 } else { 0.0 }),
            Instr::And => bin2(&mut stack, |a, b| if a > 0.0 && b > 0.0 { 1.0 } else { 0.0 }),
            Instr::Or => bin2(&mut stack, |a, b| if a > 0.0 || b > 0.0 { 1.0 } else { 0.0 }),
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
            Instr::Exp => {
                let a = stack.pop().unwrap();
                stack.push(a.exp());
            }
            Instr::Pow => bin2(&mut stack, |a, b| a.powf(b)),
            Instr::Min => bin2(&mut stack, |a, b| a.min(b)),
            Instr::Max => bin2(&mut stack, |a, b| a.max(b)),
            Instr::Jz(target) => {
                let c = stack.pop().unwrap();
                if c <= 0.0 {
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
    // Falls back to row-wise for expressions with control flow (ternary → Jz/Jmp).
    if bytecode_has_control_flow(code) {
        eval_bytecode_bulk_rowwise(code, cols)
    } else {
        eval_bytecode_bulk_vectorized(code, cols)
    }
}

fn eval_bytecode_bulk_rowwise(code: &[Instr], cols: &[&[f64]]) -> Vec<f64> {
    let n = cols[0].len();
    let mut out = Vec::with_capacity(n);
    let mut stack: Vec<f64> = Vec::with_capacity(16);

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
                    stack.push(if a > 0.0 { 0.0 } else { 1.0 });
                }
                Instr::Add => bin2(&mut stack, |a, b| a + b),
                Instr::Sub => bin2(&mut stack, |a, b| a - b),
                Instr::Mul => bin2(&mut stack, |a, b| a * b),
                Instr::Div => bin2(&mut stack, |a, b| a / b),
                Instr::Eq => bin2(&mut stack, |a, b| if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 }),
                Instr::Ne => bin2(&mut stack, |a, b| if (a - b).abs() >= f64::EPSILON { 1.0 } else { 0.0 }),
                Instr::Lt => bin2(&mut stack, |a, b| if a < b { 1.0 } else { 0.0 }),
                Instr::Le => bin2(&mut stack, |a, b| if a <= b { 1.0 } else { 0.0 }),
                Instr::Gt => bin2(&mut stack, |a, b| if a > b { 1.0 } else { 0.0 }),
                Instr::Ge => bin2(&mut stack, |a, b| if a >= b { 1.0 } else { 0.0 }),
                Instr::And => bin2(&mut stack, |a, b| if a > 0.0 && b > 0.0 { 1.0 } else { 0.0 }),
                Instr::Or => bin2(&mut stack, |a, b| if a > 0.0 || b > 0.0 { 1.0 } else { 0.0 }),
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
                Instr::Exp => {
                    let a = stack.pop().unwrap();
                    stack.push(a.exp());
                }
                Instr::Pow => bin2(&mut stack, |a, b| a.powf(b)),
                Instr::Min => bin2(&mut stack, |a, b| a.min(b)),
                Instr::Max => bin2(&mut stack, |a, b| a.max(b)),
                Instr::Jz(target) => {
                    let c = stack.pop().unwrap();
                    if c <= 0.0 {
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
        out.push(stack.pop().unwrap_or(f64::NAN));
    }

    out
}

fn bytecode_has_control_flow(code: &[Instr]) -> bool {
    code.iter().any(|i| matches!(i, Instr::Jz(_) | Instr::Jmp(_)))
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

impl<'a> BulkEvalState<'a> {
    fn new(n: usize) -> Self {
        Self {
            arena: Vec::new(),
            slots: Vec::with_capacity(16),
            stack: Vec::with_capacity(16),
            n,
        }
    }

    fn push(&mut self, s: Slot<'a>) {
        self.slots.push(s);
        self.stack.push(self.slots.len() - 1);
    }

    fn pop(&mut self) -> Slot<'a> {
        let idx = self.stack.pop().unwrap();
        self.slots[idx]
    }

    fn unary(&mut self, v: Slot<'a>, f: fn(f64) -> f64) -> Slot<'a> {
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

    fn binary(&mut self, a: Slot<'a>, b: Slot<'a>, f: fn(f64, f64) -> f64) -> Slot<'a> {
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
                let rhs = self.arena[j].clone();
                for (x, y) in self.arena[i].iter_mut().zip(rhs.iter()) {
                    *x = f(*x, *y);
                }
                Slot::Owned(i)
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

    // Control-flow is not supported in the vectorized path.
    debug_assert!(!bytecode_has_control_flow(code));

    let mut st = BulkEvalState::new(n);

    for instr in code {
        match *instr {
            Instr::Const(v) => st.push(Slot::Scalar(v)),
            Instr::Load(i) => st.push(Slot::Col(cols[i])),
            Instr::Neg => {
                let a = st.pop();
                let r = st.unary(a, |x| -x);
                st.push(r);
            }
            Instr::Not => {
                let a = st.pop();
                let r = st.unary(a, |x| if x > 0.0 { 0.0 } else { 1.0 });
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
            Instr::Exp => {
                let a = st.pop();
                let r = st.unary(a, f64::exp);
                st.push(r);
            }
            Instr::Add => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x + y);
                st.push(r);
            }
            Instr::Sub => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x - y);
                st.push(r);
            }
            Instr::Mul => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x * y);
                st.push(r);
            }
            Instr::Div => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x / y);
                st.push(r);
            }
            Instr::Eq => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| {
                    if (x - y).abs() < f64::EPSILON { 1.0 } else { 0.0 }
                });
                st.push(r);
            }
            Instr::Ne => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| {
                    if (x - y).abs() >= f64::EPSILON { 1.0 } else { 0.0 }
                });
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
            Instr::And => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| {
                    if x > 0.0 && y > 0.0 { 1.0 } else { 0.0 }
                });
                st.push(r);
            }
            Instr::Or => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| {
                    if x > 0.0 || y > 0.0 { 1.0 } else { 0.0 }
                });
                st.push(r);
            }
            Instr::Pow => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x.powf(y));
                st.push(r);
            }
            Instr::Min => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x.min(y));
                st.push(r);
            }
            Instr::Max => {
                let b = st.pop();
                let a = st.pop();
                let r = st.binary(a, b, |x, y| x.max(y));
                st.push(r);
            }
            Instr::Jz(_) | Instr::Jmp(_) => unreachable!("control flow handled by row-wise path"),
        }
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
                            RootError::Expression("internal error: branch index out of bounds".to_string())
                        })?;
                        let indexed = format!("{name}[{index}]");
                        let new_i = branches
                            .iter()
                            .position(|s| s == &indexed)
                            .unwrap_or_else(|| {
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
                let n: f64 = s
                    .parse()
                    .map_err(|_| expr_err(input, Span { start, end: i }, format!("invalid number: '{s}'")))?;
                TokenKind::Num(n)
            }
            _ if (b as char).is_ascii_alphabetic() || b == b'_' => {
                i += 1;
                while i < bytes.len() {
                    let c = bytes[i] as char;
                    if c.is_ascii_alphanumeric() || bytes[i] == b'_' {
                        i += 1;
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

        loop {
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::LBracket) => {
                    let lb = self.advance().unwrap();
                    let idx_tok = self.advance().ok_or_else(|| {
                        expr_err(self.input, lb.span, "expected index after '['".to_string())
                    })?;
                    let index = match idx_tok.kind {
                        TokenKind::Num(n) if n.fract() == 0.0 && n >= 0.0 => n as usize,
                        _ => {
                            return Err(expr_err(
                                self.input,
                                idx_tok.span,
                                "index must be a non-negative integer literal".to_string(),
                            ));
                        }
                    };
                    let rb = self.advance().ok_or_else(|| {
                        expr_err(self.input, lb.span, "expected ']'".to_string())
                    })?;
                    if rb.kind != TokenKind::RBracket {
                        return Err(expr_err(self.input, rb.span, format!("expected ']', got {:?}", rb.kind)));
                    }
                    let span = Span { start: lb.span.start, end: rb.span.end };
                    e = Expr::Index { base: Box::new(e), index, span };
                }
                _ => break,
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
                        Err(expr_err(
                            self.input,
                            t.span,
                            format!("expected ')', got {:?}", t.kind),
                        ))
                    }
                } else {
                    Err(expr_err(self.input, span, "expected ')'".to_string()))
                }
            }
            Some(Token { kind: TokenKind::Ident(name), span, .. }) => {
                // Check for function call
                if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::LParen)) {
                    self.advance(); // consume '('
                    let func = match name.as_str() {
                        "abs" => Func::Abs,
                        "sqrt" => Func::Sqrt,
                        "log" => Func::Log,
                        "exp" => Func::Exp,
                        "pow" => Func::Pow,
                        "min" => Func::Min,
                        "max" => Func::Max,
                        _ => {
                            return Err(expr_err(
                                self.input,
                                span,
                                format!("unknown function: '{name}'"),
                            ));
                        }
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
    fn comparison_and_boolean() {
        let e = CompiledExpr::compile("njet >= 4 && pt_lead > 25.0").unwrap();
        assert_eq!(e.required_branches, vec!["njet", "pt_lead"]);
        assert!((e.eval_row(&[4.0, 30.0]) - 1.0).abs() < 1e-10);
        assert!((e.eval_row(&[3.0, 30.0]) - 0.0).abs() < 1e-10);
        assert!((e.eval_row(&[4.0, 20.0]) - 0.0).abs() < 1e-10);
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
    fn errors_report_line_col() {
        let err = CompiledExpr::compile("x +\n  * 2").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("line 2, col 3"), "unexpected error message: {msg}");
    }
}
