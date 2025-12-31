use crate::domain::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Pow, Mod,
    Eq, Neq, Lt, Gt, Le, Ge,
    And, Or,
    Pipe,    // |> operator
    Compose, // . operator
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Value),
    Var(String),
    BinaryOp(Box<Expr>, BinaryOp, Box<Expr>),
    Call(String, Vec<Expr>),
    Vector(Vec<Expr>),
    Matrix(Vec<Vec<Expr>>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Let(String, Box<Expr>, Box<Expr>),
    Lambda(Vec<String>, Box<Expr>),
    List(Vec<Expr>),
    Guarded(Vec<Guard>), // | cond = e | ...
}

/// A guard is a condition-result pair
#[derive(Debug, Clone, PartialEq)]
pub struct Guard {
    pub condition: Expr,
    pub result: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<String>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopLevel {
    FunctionDef(FunctionDef),
}
