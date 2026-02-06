//! Expression engine for evaluating selections and weights from string
//! expressions over TTree branches.
//!
//! Supports arithmetic (+, -, *, /), comparisons (==, !=, <, <=, >, >=),
//! boolean operators (&&, ||, !), and built-in functions (abs, sqrt, log,
//! exp, pow, min, max).

use crate::error::{Result, RootError};

// ── AST ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Expr {
    Number(f64),
    Var(usize), // index into required_branches
    UnaryNeg(Box<Expr>),
    UnaryNot(Box<Expr>),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
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

// ── Compiled expression ────────────────────────────────────────

/// A compiled expression ready for evaluation.
///
/// Variable identifiers in the expression are mapped to branch names.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    ast: Expr,
    /// Branch names referenced by this expression (ordered by first occurrence).
    pub required_branches: Vec<String>,
}

impl CompiledExpr {
    /// Parse and compile an expression string.
    ///
    /// Variable names in the expression correspond to TTree branch names.
    pub fn compile(input: &str) -> Result<Self> {
        let tokens = tokenize(input)?;
        let mut parser = Parser::new(&tokens);
        let ast = parser.parse_or()?;
        if parser.pos < parser.tokens.len() {
            return Err(RootError::Expression(format!(
                "unexpected token after expression: {:?}",
                parser.tokens[parser.pos]
            )));
        }
        let branches = std::mem::take(&mut parser.branches);
        Ok(CompiledExpr { ast, required_branches: branches })
    }

    /// Evaluate the expression for a single row.
    ///
    /// `values` must have the same length and order as `required_branches`.
    pub fn eval_row(&self, values: &[f64]) -> f64 {
        eval_expr(&self.ast, values)
    }

    /// Evaluate the expression for all rows (column-wise).
    ///
    /// `columns` must have the same length and order as `required_branches`;
    /// each column must have the same number of entries.
    pub fn eval_bulk(&self, columns: &[&[f64]]) -> Vec<f64> {
        if columns.is_empty() {
            // Constant expression — evaluate once
            return vec![eval_expr(&self.ast, &[])];
        }
        let n = columns[0].len();
        let mut row = vec![0.0f64; columns.len()];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            for (j, col) in columns.iter().enumerate() {
                row[j] = col[i];
            }
            out.push(eval_expr(&self.ast, &row));
        }
        out
    }
}

// ── Evaluation ─────────────────────────────────────────────────

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
    }
}

// ── Tokenizer ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
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

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Two-character operators
        if i + 1 < chars.len() {
            let two = &input[i..i + 2];
            let tok = match two {
                "&&" => Some(Token::And),
                "||" => Some(Token::Or),
                "==" => Some(Token::Eq),
                "!=" => Some(Token::Ne),
                "<=" => Some(Token::Le),
                ">=" => Some(Token::Ge),
                _ => None,
            };
            if let Some(t) = tok {
                tokens.push(t);
                i += 2;
                continue;
            }
        }

        match c {
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '<' => {
                tokens.push(Token::Lt);
                i += 1;
            }
            '>' => {
                tokens.push(Token::Gt);
                i += 1;
            }
            '!' => {
                tokens.push(Token::Not);
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && i > start
                            && (chars[i - 1] == 'e' || chars[i - 1] == 'E')))
                {
                    i += 1;
                }
                let s = &input[start..i];
                let n: f64 = s
                    .parse()
                    .map_err(|_| RootError::Expression(format!("invalid number: '{}'", s)))?;
                tokens.push(Token::Num(n));
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                tokens.push(Token::Ident(input[start..i].to_string()));
            }
            _ => {
                return Err(RootError::Expression(format!("unexpected character: '{}'", c)));
            }
        }
    }

    Ok(tokens)
}

// ── Parser (recursive descent) ─────────────────────────────────

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    branches: Vec<String>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0, branches: Vec::new() }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        match self.advance() {
            Some(t) if t == expected => Ok(()),
            other => {
                Err(RootError::Expression(format!("expected {:?}, got {:?}", expected, other)))
            }
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

    fn parse_or(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Some(Token::Or)) {
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::BinOp(BinOp::Or, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_cmp()?;
        while matches!(self.peek(), Some(Token::And)) {
            self.advance();
            let rhs = self.parse_cmp()?;
            lhs = Expr::BinOp(BinOp::And, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Expr> {
        let lhs = self.parse_add()?;
        let op = match self.peek() {
            Some(Token::Eq) => BinOp::Eq,
            Some(Token::Ne) => BinOp::Ne,
            Some(Token::Lt) => BinOp::Lt,
            Some(Token::Le) => BinOp::Le,
            Some(Token::Gt) => BinOp::Gt,
            Some(Token::Ge) => BinOp::Ge,
            _ => return Ok(lhs),
        };
        self.advance();
        let rhs = self.parse_add()?;
        Ok(Expr::BinOp(op, Box::new(lhs), Box::new(rhs)))
    }

    fn parse_add(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_mul()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    let rhs = self.parse_mul()?;
                    lhs = Expr::BinOp(BinOp::Add, Box::new(lhs), Box::new(rhs));
                }
                Some(Token::Minus) => {
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
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    lhs = Expr::BinOp(BinOp::Mul, Box::new(lhs), Box::new(rhs));
                }
                Some(Token::Slash) => {
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
        match self.peek() {
            Some(Token::Minus) => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::UnaryNeg(Box::new(e)))
            }
            Some(Token::Not) => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::UnaryNot(Box::new(e)))
            }
            _ => self.parse_atom(),
        }
    }

    fn parse_atom(&mut self) -> Result<Expr> {
        match self.advance().cloned() {
            Some(Token::Num(n)) => Ok(Expr::Number(n)),
            Some(Token::LParen) => {
                let e = self.parse_or()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            Some(Token::Ident(name)) => {
                // Check for function call
                if matches!(self.peek(), Some(Token::LParen)) {
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
                            return Err(RootError::Expression(format!(
                                "unknown function: '{}'",
                                name
                            )));
                        }
                    };
                    let mut args = vec![self.parse_or()?];
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.advance();
                        args.push(self.parse_or()?);
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call(func, args))
                } else {
                    // Variable reference
                    let idx = self.resolve_var(&name);
                    Ok(Expr::Var(idx))
                }
            }
            other => Err(RootError::Expression(format!(
                "expected number, identifier, or '(', got {:?}",
                other
            ))),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_arithmetic() {
        let e = CompiledExpr::compile("2 + 3 * 4").unwrap();
        assert!(e.required_branches.is_empty());
        assert!((e.eval_row(&[]) - 14.0).abs() < 1e-10);
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
}
