use crate::domain::Value;
use std::collections::HashMap;
use crate::vm::NativeFunc;

pub fn create_stdlib() -> (HashMap<String, NativeFunc>, HashMap<String, Value>) {
    let mut map: HashMap<String, NativeFunc> = HashMap::new();
    let mut globals: HashMap<String, Value> = HashMap::new();
    
    map.insert("dot".to_string(), native_dot);
    map.insert("mul".to_string(), native_mul);
    map.insert("vadd".to_string(), native_vadd);
    map.insert("transpose".to_string(), native_transpose);
    map.insert("vec".to_string(), native_vec);
    map.insert("mat".to_string(), native_mat);
    map.insert("abs".to_string(), native_abs);
    map.insert("sigmoid".to_string(), native_sigmoid);

    // Math
    map.insert("sin".to_string(), |args| math_op(args, f64::sin));
    map.insert("cos".to_string(), |args| math_op(args, f64::cos));
    map.insert("tan".to_string(), |args| math_op(args, f64::tan));
    map.insert("asin".to_string(), |args| math_op(args, f64::asin));
    map.insert("acos".to_string(), |args| math_op(args, f64::acos));
    map.insert("atan".to_string(), |args| math_op(args, f64::atan));
    map.insert("sinh".to_string(), |args| math_op(args, f64::sinh));
    map.insert("cosh".to_string(), |args| math_op(args, f64::cosh));
    map.insert("sqrt".to_string(), |args| math_op(args, f64::sqrt));
    map.insert("exp".to_string(), |args| math_op(args, f64::exp));
    map.insert("log".to_string(), |args| math_op(args, f64::ln));
    map.insert("log10".to_string(), |args| math_op(args, f64::log10));
    
    globals.insert("pi".to_string(), Value::Float(std::f64::consts::PI));
    
    map.insert("cons".to_string(), native_cons);
    map.insert("head".to_string(), native_head);
    map.insert("tail".to_string(), native_tail);
    map.insert("isEmpty".to_string(), native_is_empty);

    (map, globals)
}

fn native_cons(args: Vec<Value>) -> Result<Value, String> {
    use std::rc::Rc;
    use crate::domain::ListNode;
    
    if args.len() != 2 { return Err("cons expects 2 arguments".to_string()); }
    let head = args[0].clone();
    match &args[1] {
        Value::List(tail) => {
            // O(1) cons
            let new_node = ListNode::Cons(head, Rc::clone(tail));
            Ok(Value::List(Rc::new(new_node)))
        },
        _ => Err("cons expects second argument to be a list".to_string())
    }
}

fn native_head(args: Vec<Value>) -> Result<Value, String> {
    use crate::domain::ListNode;
    
    if args.len() != 1 { return Err("head expects 1 argument".to_string()); }
    match &args[0] {
        Value::List(node) => {
            match node.as_ref() {
                ListNode::Nil => Err("head of empty list".to_string()),
                ListNode::Cons(val, _) => Ok(val.clone()),
            }
        },
        _ => Err("head expects a list".to_string())
    }
}

fn native_tail(args: Vec<Value>) -> Result<Value, String> {
    use std::rc::Rc;
    use crate::domain::ListNode;
    
    if args.len() != 1 { return Err("tail expects 1 argument".to_string()); }
    match &args[0] {
        Value::List(node) => {
            match node.as_ref() {
                ListNode::Nil => Err("tail of empty list".to_string()),
                ListNode::Cons(_, tail) => Ok(Value::List(Rc::clone(tail))), // O(1)
            }
        },
        _ => Err("tail expects a list".to_string())
    }
}

fn native_is_empty(args: Vec<Value>) -> Result<Value, String> {
    use crate::domain::ListNode;
    
    if args.len() != 1 { return Err("isEmpty expects 1 argument".to_string()); }
    match &args[0] {
        Value::List(node) => {
            match node.as_ref() {
                ListNode::Nil => Ok(Value::Bool(true)),
                ListNode::Cons(_, _) => Ok(Value::Bool(false)),
            }
        },
        _ => Err("isEmpty expects a list".to_string())
    }
}


fn math_op(args: Vec<Value>, op: fn(f64) -> f64) -> Result<Value, String> {
    if args.len() != 1 { return Err("math op expects 1 argument".to_string()); }
    match args[0] {
        Value::Int(i) => Ok(Value::Float(op(i as f64))),
        Value::Float(f) => Ok(Value::Float(op(f))),
        _ => Err("math op expects number".to_string())
    }
}

fn native_dot(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("dot expects 2 arguments".to_string()); }
    match (&args[0], &args[1]) {
        (Value::Vector(v1), Value::Vector(v2)) => {
            if v1.len() != v2.len() {
                return Err(format!("Vector dimension mismatch: {} vs {}", v1.len(), v2.len()));
            }
            Ok(Value::Float(v1.as_ref().dot(v2.as_ref())))
        },
        _ => Err("dot expects two vectors".to_string())
    }
}

fn native_mul(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("mul expects 2 arguments".to_string()); }
    match (&args[0], &args[1]) {
        (Value::Matrix(m1), Value::Matrix(m2)) => {
             use std::rc::Rc;
             Ok(Value::Matrix(Rc::new(m1.as_ref().dot(m2.as_ref()))))
        },
         (Value::Matrix(m), Value::Vector(v)) => {
            use std::rc::Rc;
            Ok(Value::Vector(Rc::new(m.as_ref().dot(v.as_ref()))))
        },
        _ => Err("mul expects Matrix*Matrix or Matrix*Vector".to_string())
    }
}

fn native_transpose(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("transpose expects 1 argument".to_string()); }
    match &args[0] {
        Value::Matrix(m) => {
             use std::rc::Rc;
             // to_owned() needed because .t() returns a View, but we own the Value
             Ok(Value::Matrix(Rc::new(m.as_ref().t().to_owned())))
        },
        _ => Err("transpose expects a matrix".to_string())
    }
}
fn native_vec(args: Vec<Value>) -> Result<Value, String> {
    use std::rc::Rc;
    use ndarray::Array1;
    
    if args.len() != 1 { return Err("vec expects 1 argument (list)".to_string()); }
    match &args[0] {
        Value::List(node) => {
            // Convert list to vector
            let mut elements = Vec::new();
            let mut curr = node.as_ref();
            while let crate::domain::ListNode::Cons(val, next) = curr {
                match val {
                    Value::Int(i) => elements.push(*i as f64),
                    Value::Float(f) => elements.push(*f),
                    _ => return Err("vec expects list of numbers".to_string()),
                }
                curr = next.as_ref();
            }
            Ok(Value::Vector(Rc::new(Array1::from(elements))))
        },
        _ => Err("vec expects a list".to_string())
    }
}

fn native_mat(args: Vec<Value>) -> Result<Value, String> {
    use std::rc::Rc;
    use ndarray::Array2;
    
    if args.len() != 1 { return Err("mat expects 1 argument (list of lists)".to_string()); }
    match &args[0] {
        Value::List(rows_node) => {
            let mut rows = Vec::new();
            let mut curr_row = rows_node.as_ref();
            let mut cols = 0;
            let mut first = true;
            
            while let crate::domain::ListNode::Cons(row_val, next_row) = curr_row {
                if let Value::List(col_node) = row_val {
                    let mut row_data = Vec::new();
                    let mut curr_col = col_node.as_ref();
                    while let crate::domain::ListNode::Cons(val, next_col) = curr_col {
                        match val {
                            Value::Int(i) => row_data.push(*i as f64),
                            Value::Float(f) => row_data.push(*f),
                            _ => return Err("mat expects list of numbers".to_string()),
                        }
                        curr_col = next_col.as_ref();
                    }
                    
                    if first {
                        cols = row_data.len();
                        first = false;
                    } else if row_data.len() != cols {
                        return Err("mat expects all rows to have same length".to_string());
                    }
                    rows.extend(row_data);
                } else {
                    return Err("mat expects list of lists".to_string());
                }
                curr_row = next_row.as_ref();
            }
            
            let n_rows = if cols > 0 { rows.len() / cols } else { 0 };
            let arr = Array2::from_shape_vec((n_rows, cols), rows)
                    .map_err(|e| format!("Failed to create matrix: {}", e))?;
            Ok(Value::Matrix(Rc::new(arr)))
        },
        _ => Err("mat expects a list".to_string())
    }
}
fn native_abs(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("abs expects 1 argument".to_string()); }
    match &args[0] {
        Value::Int(i) => Ok(Value::Int(i.abs())),
        Value::Float(f) => Ok(Value::Float(f.abs())),
        Value::Complex(c) => Ok(Value::Float(c.norm())),
        _ => Err("abs expects a number".to_string())
    }
}

fn native_vadd(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("vadd expects 2 arguments".to_string()); }
    match (&args[0], &args[1]) {
        (Value::Vector(v1), Value::Vector(v2)) => {
             use std::rc::Rc;
             if v1.len() != v2.len() {
                 return Err("Vector dimension mismatch for addition".to_string());
             }
             Ok(Value::Vector(Rc::new(v1.as_ref() + v2.as_ref())))
        },
        _ => Err("vadd expects two vectors".to_string())
    }
}

fn native_sigmoid(args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("sigmoid expects 1 argument".to_string()); }
    match &args[0] {
        Value::Vector(v) => {
             use std::rc::Rc;
             // sigmoid(x) = 1 / (1 + exp(-x))
             let res = v.as_ref().mapv(|x| 1.0 / (1.0 + (-x).exp()));
             Ok(Value::Vector(Rc::new(res)))
        },
        Value::Float(f) => {
             Ok(Value::Float(1.0 / (1.0 + (-f).exp())))
        },
        _ => Err("sigmoid expects vector or float".to_string())
    }
}
