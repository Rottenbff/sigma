use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::fmt;
use std::collections::HashMap;
use std::rc::Rc;

/// Persistent linked list node
#[derive(Debug, Clone, PartialEq)]
pub enum ListNode {
    Nil,
    Cons(Value, Rc<ListNode>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Complex(Complex64),
    Vector(Rc<Array1<f64>>),
    Matrix(Rc<Array2<f64>>),
    Function(String),
    Closure {
        func_name: String,
        captured: HashMap<String, Value>,
    },
    PartialApp {
        func_name: String,
        captured: HashMap<String, Value>,
        applied_args: Vec<Value>,
    },
    List(Rc<ListNode>),
    // Future: Tuple(Vec<Value>)
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Complex(c) => write!(f, "{}", c),
            Value::Vector(v) => write!(f, "vec {}", v),
            Value::Matrix(m) => write!(f, "mat {}", m),
            Value::Function(name) => write!(f, "<function {}>", name),
            Value::Closure { func_name, .. } => write!(f, "<closure {}>", func_name),
            Value::PartialApp { func_name, applied_args, .. } => {
                write!(f, "<partial {} ({} args applied)>", func_name, applied_args.len())
            }
            Value::List(node) => {
                write!(f, "[")?;
                let mut current = node.as_ref();
                let mut first = true;
                while let ListNode::Cons(val, next) = current {
                    if !first { write!(f, ", ")?; }
                    write!(f, "{}", val)?;
                    first = false;
                    current = next.as_ref();
                }
                write!(f, "]")
            }
        }
    }
}
