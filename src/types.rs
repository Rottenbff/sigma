use std::fmt;


#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Complex,
    String,
    Unit,
    
    Var(usize),
    
    List(Box<Type>),
    Vector(Box<Type>),
    Matrix(Box<Type>),
    Function(Box<Type>, Box<Type>),
}

impl Type {
    pub fn function(args: Vec<Type>, ret: Type) -> Type {
        // Right fold for currying
        let mut t = ret;
        for arg in args.into_iter().rev() {
            t = Type::Function(Box::new(arg), Box::new(t));
        }
        t
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::Complex => write!(f, "Complex"),
            Type::String => write!(f, "String"),
            Type::Unit => write!(f, "()"),
            Type::Var(id) => write!(f, "t{}", id),
            Type::List(t) => write!(f, "[{}]", t),
            Type::Vector(t) => write!(f, "Vector<{}>", t),
            Type::Matrix(t) => write!(f, "Matrix<{}>", t),
            Type::Function(arg, ret) => {
                match **arg {
                    Type::Function(_, _) => write!(f, "({}) -> {}", arg, ret),
                    _ => write!(f, "{} -> {}", arg, ret),
                }
            }
        }
    }
}
