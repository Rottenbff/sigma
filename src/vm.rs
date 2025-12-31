use crate::domain::Value;
use num_complex::Complex64;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum OpCode {
    Push(Value),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Load(String),
    Store(String),
    Move(String),
    Call(usize),
    Ret,
    MakeVector(usize),
    MakeList(usize),
    MakeMatrix(usize, usize),
    MakeClosure(String, Vec<String>),
    Jump(usize),
    JumpIfFalse(usize),
    Eq, Neq, Lt, Gt, Le, Ge,
    Map,
    Filter,
    Foldl,
}

#[derive(Debug, Clone)]
pub struct FunctionObj {
    pub name: String,
    pub args: Vec<String>,
    pub code: std::rc::Rc<Vec<OpCode>>,
}

struct CallFrame {
    _func_name: String,
    ip: usize,
    locals: HashMap<String, Value>,
    code: std::rc::Rc<Vec<OpCode>>,
}

pub type NativeFunc = fn(Vec<Value>) -> Result<Value, String>;

pub struct VM {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    functions: HashMap<String, FunctionObj>,
    native_functions: HashMap<String, NativeFunc>,
    globals: HashMap<String, Value>,
}

impl VM {
    pub fn new(functions: HashMap<String, FunctionObj>, native_functions: HashMap<String, NativeFunc>, globals: HashMap<String, Value>) -> Self {
        Self {
            stack: Vec::new(),
            frames: Vec::new(),
            functions,
            native_functions,
            globals,
        }
    }

    pub fn run_function(&mut self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        let func = self.functions.get(name).ok_or_else(|| format!("Function '{}' not found", name))?;
        
        if args.len() != func.args.len() {
            return Err(format!("Arity mismatch: {} expects {} args, got {}", name, func.args.len(), args.len()));
        }

        let mut locals = HashMap::new();
        for (i, arg_name) in func.args.iter().enumerate() {
            locals.insert(arg_name.clone(), args[i].clone());
        }

        let frame = CallFrame {
            _func_name: name.to_string(),
            ip: 0,
            locals,
            code: std::rc::Rc::clone(&func.code),
        };

        self.frames.push(frame);
        self.run_loop()
    }

    fn run_loop(&mut self) -> Result<Value, String> {
        while !self.frames.is_empty() {
            let frame_idx = self.frames.len() - 1;
            
            if self.frames[frame_idx].ip >= self.frames[frame_idx].code.len() {
                 let _ = self.frames.pop();
                 if self.frames.is_empty() {
                     return self.stack.pop().ok_or_else(|| "Stack empty after implicit return".to_string());
                 }
                 continue;
            }

            let op = self.frames[frame_idx].code[self.frames[frame_idx].ip].clone();
            self.frames[frame_idx].ip += 1;

            self.execute_op(op, frame_idx)?;
            
            if self.frames.is_empty() {
                 return self.stack.pop().ok_or_else(|| "Stack empty after return".to_string());
            }
        }
        
        Err("VM halted without result".to_string())
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack.pop().ok_or_else(|| "Stack underflow".to_string())
    }

    fn compare(a: Value, b: Value) -> Result<i8, String> {
        match (a, b) {
            (Value::Int(i1), Value::Int(i2)) => {
                if i1 < i2 { Ok(-1) } else if i1 > i2 { Ok(1) } else { Ok(0) }
            },
             (Value::Float(f1), Value::Float(f2)) => {
                 if f1 < f2 { Ok(-1) } else if f1 > f2 { Ok(1) } else { Ok(0) }
            },
             (Value::Int(i), Value::Float(f)) => {
                 let f1 = i as f64;
                 if f1 < f { Ok(-1) } else if f1 > f { Ok(1) } else { Ok(0) }
            },
             (Value::Float(f), Value::Int(i)) => {
                 let f2 = i as f64;
                 if f < f2 { Ok(-1) } else if f > f2 { Ok(1) } else { Ok(0) }
            },
            // Complex numbers are unordered
            (Value::Complex(_), _) | (_, Value::Complex(_)) => {
                Err("Complex numbers cannot be compared with <, >, <=, >=".to_string())
            }
            _ => Err("Invalid types for comparison".to_string())
        }
    }

    fn add(a: Value, b: Value) -> Result<Value, String> {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector(std::rc::Rc::new(a.as_ref() + b.as_ref()))),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix(std::rc::Rc::new(a.as_ref() + b.as_ref()))),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a + b)),
            (Value::Complex(a), Value::Int(b)) => Ok(Value::Complex(a + b as f64)),
            (Value::Int(a), Value::Complex(b)) => Ok(Value::Complex(a as f64 + b)),
            (Value::Complex(a), Value::Float(b)) => Ok(Value::Complex(a + b)),
            (Value::Float(a), Value::Complex(b)) => Ok(Value::Complex(a + b)),
            _ => Err("Type mismatch for Add".to_string()),
        }
    }

    fn sub(a: Value, b: Value) -> Result<Value, String> {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - b as f64)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a - b)),
            (Value::Complex(a), Value::Int(b)) => Ok(Value::Complex(a - b as f64)),
            (Value::Int(a), Value::Complex(b)) => Ok(Value::Complex(a as f64 - b)),
            (Value::Complex(a), Value::Float(b)) => Ok(Value::Complex(a - b)),
            (Value::Float(a), Value::Complex(b)) => Ok(Value::Complex(a - b)),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector(std::rc::Rc::new(a.as_ref() - b.as_ref()))),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix(std::rc::Rc::new(a.as_ref() - b.as_ref()))),
            _ => Err("Type mismatch for Sub".to_string()),
        }
    }

    fn mul(a: Value, b: Value) -> Result<Value, String> {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * b as f64)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a * b)),
            (Value::Complex(a), Value::Int(b)) => Ok(Value::Complex(a * b as f64)),
            (Value::Int(a), Value::Complex(b)) => Ok(Value::Complex(a as f64 * b)),
            (Value::Complex(a), Value::Float(b)) => Ok(Value::Complex(a * b)),
            (Value::Float(a), Value::Complex(b)) => Ok(Value::Complex(a * b)),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector(std::rc::Rc::new(a.as_ref() * b.as_ref()))),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix(std::rc::Rc::new(a.as_ref() * b.as_ref()))),
            _ => Err("Type mismatch for Mul".to_string()),
        }
    }

    fn div(a: Value, b: Value) -> Result<Value, String> {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / b as f64)),
             (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a / b)),
            (Value::Complex(a), Value::Int(b)) => Ok(Value::Complex(a / b as f64)),
            (Value::Int(a), Value::Complex(b)) => Ok(Value::Complex(a as f64 / b)),
            (Value::Complex(a), Value::Float(b)) => Ok(Value::Complex(a / b)),
            (Value::Float(a), Value::Complex(b)) => Ok(Value::Complex(a / b)),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector(std::rc::Rc::new(a.as_ref() / b.as_ref()))),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix(std::rc::Rc::new(a.as_ref() / b.as_ref()))),
            _ => Err("Type mismatch for Div".to_string()),
        }
    }

    fn pow(a: Value, b: Value) -> Result<Value, String> {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => {
                if b < 0 {
                     Ok(Value::Float((a as f64).powf(b as f64)))
                } else {
                     Ok(Value::Int(a.pow(b as u32))) 
                }
            },
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(b))),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float((a as f64).powf(b))),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a.powf(b as f64))),
             (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a.powc(b))),
            (Value::Complex(a), Value::Int(b)) => Ok(Value::Complex(a.powf(b as f64))),
            (Value::Int(a), Value::Complex(b)) => Ok(Value::Complex(Complex64::new(a as f64, 0.0).powc(b))),
            (Value::Complex(a), Value::Float(b)) => Ok(Value::Complex(a.powf(b))),
            (Value::Float(a), Value::Complex(b)) => Ok(Value::Complex(Complex64::new(a, 0.0).powc(b))),
            _ => Err("Type mismatch for Pow".to_string()),
        }
    }
    
    /// Converts a List value to a Vec
    fn list_to_vec(&self, list: &Value) -> Result<Vec<Value>, String> {
        use crate::domain::ListNode;
        
        match list {
            Value::List(node) => {
                let mut elements = Vec::new();
                let mut current = node.as_ref();
                while let ListNode::Cons(val, next) = current {
                    elements.push(val.clone());
                    current = next.as_ref();
                }
                Ok(elements)
            }
            _ => Err("Expected a list".to_string())
        }
    }
    
    /// Invokes a callable value
    fn call_value(&mut self, func: &Value, args: Vec<Value>) -> Result<Value, String> {
        match func {
            Value::Function(name) => {
                if let Some(&native_fn) = self.native_functions.get(name) {
                    return native_fn(args);
                }
                if let Some(func_obj) = self.functions.get(name).cloned() {
                    let mut locals = HashMap::new();
                    for (i, param) in func_obj.args.iter().enumerate() {
                        if i < args.len() {
                            locals.insert(param.clone(), args[i].clone());
                        }
                    }
                    
                    let saved_frames_len = self.frames.len();
                    
                    let frame = CallFrame {
                        _func_name: name.clone(),
                        ip: 0,
                        locals,
                        code: std::rc::Rc::clone(&func_obj.code),
                    };
                    self.frames.push(frame);
                    
                    let result = self.run_until_frame(saved_frames_len)?;
                    Ok(result)
                } else {
                    Err(format!("Function '{}' not found", name))
                }
            }
            Value::Closure { func_name, captured } => {
                if let Some(func_obj) = self.functions.get(func_name).cloned() {
                    let mut locals = captured.clone();
                    for (i, param) in func_obj.args.iter().enumerate() {
                        if i < args.len() {
                            locals.insert(param.clone(), args[i].clone());
                        }
                    }
                    
                    let saved_frames_len = self.frames.len();
                    let frame = CallFrame {
                        _func_name: func_name.clone(),
                        ip: 0,
                        locals,
                        code: std::rc::Rc::clone(&func_obj.code),
                    };
                    self.frames.push(frame);
                    
                    let result = self.run_until_frame(saved_frames_len)?;
                    Ok(result)
                } else {
                    Err(format!("Closure function '{}' not found", func_name))
                }
            }
            Value::PartialApp { func_name, captured, applied_args } => {
                let mut all_args = applied_args.clone();
                all_args.extend(args);
                
                let closure = Value::Closure {
                    func_name: func_name.clone(),
                    captured: captured.clone(),
                };
                self.call_value(&closure, all_args)
            }
            _ => Err(format!("Cannot call non-function value: {:?}", func))
        }
    }
    
    /// Runs the VM until the stack frame depth returns to target_len
    fn run_until_frame(&mut self, target_len: usize) -> Result<Value, String> {
        while self.frames.len() > target_len {
            let frame_idx = self.frames.len() - 1;
            
            if self.frames[frame_idx].ip >= self.frames[frame_idx].code.len() {
                let _ = self.frames.pop();
                if self.frames.len() == target_len {
                    return self.stack.pop().ok_or_else(|| "Stack empty after return".to_string());
                }
                continue;
            }
            
            let op = self.frames[frame_idx].code[self.frames[frame_idx].ip].clone();
            self.frames[frame_idx].ip += 1;
            
            self.execute_op(op, frame_idx)?;
        }
        
        self.stack.pop().ok_or_else(|| "Stack empty after function call".to_string())
    }
    
    /// Executes a single opcode
    fn execute_op(&mut self, op: OpCode, frame_idx: usize) -> Result<(), String> {
        match op {
            OpCode::Push(v) => self.stack.push(v),
            OpCode::Load(name) => {
                if let Some(val) = self.frames[frame_idx].locals.get(&name) {
                    self.stack.push(val.clone());
                } else if let Some(val) = self.globals.get(&name) {
                    self.stack.push(val.clone());
                } else if let Some(func) = self.functions.get(&name) {
                    if func.args.is_empty() {
                         let new_frame = CallFrame {
                            _func_name: name.clone(),
                            ip: 0,
                            locals: HashMap::new(),
                            code: std::rc::Rc::clone(&func.code),
                        };
                        self.frames.push(new_frame);
                    } else {
                        self.stack.push(Value::Function(name));
                    }
                } else if self.native_functions.contains_key(&name) {
                     self.stack.push(Value::Function(name));
                } else {
                    return Err(format!("Variable '{}' not found", name));
                }
            }
            OpCode::Move(name) => {
                if let Some(val) = self.frames[frame_idx].locals.remove(&name) {
                    self.stack.push(val);
                } else {
                    return Err(format!("Variable '{}' not found for move", name));
                }
            }
            OpCode::Store(name) => {
                let val = self.pop()?;
                self.frames[frame_idx].locals.insert(name, val);
            }
            OpCode::Call(arg_count) => {
                let mut args = Vec::new();
                for _ in 0..arg_count {
                    args.push(self.pop()?);
                }
                args.reverse();

                let callee = self.pop()?;
                
                match callee {
                    Value::Function(ref func_name) => {
                        if let Some(func_obj) = self.functions.get(func_name).cloned() {
                            let expected = func_obj.args.len();
                            if arg_count < expected {
                                self.stack.push(Value::PartialApp {
                                    func_name: func_name.clone(),
                                    captured: HashMap::new(),
                                    applied_args: args,
                                });
                            } else if arg_count == expected {
                                let mut locals = HashMap::new();
                                for (i, param) in func_obj.args.iter().enumerate() {
                                    locals.insert(param.clone(), args[i].clone());
                                }
                                let new_frame = CallFrame {
                                    _func_name: func_name.clone(),
                                    ip: 0,
                                    locals,
                                    code: std::rc::Rc::clone(&func_obj.code),
                                };
                                self.frames.push(new_frame);
                            } else {
                                return Err(format!("Too many args for {}: expected {}, got {}", func_name, expected, arg_count));
                            }
                        } else if let Some(&native_func) = self.native_functions.get(func_name) {
                            let res = native_func(args)?;
                            self.stack.push(res);
                        } else {
                            return Err(format!("Function '{}' not found", func_name));
                        }
                    },
                    Value::Closure { func_name, captured } => {
                        if let Some(func_obj) = self.functions.get(&func_name).cloned() {
                            let expected = func_obj.args.len();
                            if arg_count < expected {
                                self.stack.push(Value::PartialApp {
                                    func_name: func_name.clone(),
                                    captured,
                                    applied_args: args,
                                });
                            } else if arg_count == expected {
                                let mut locals = captured;
                                for (i, param) in func_obj.args.iter().enumerate() {
                                    locals.insert(param.clone(), args[i].clone());
                                }
                                let new_frame = CallFrame {
                                    _func_name: func_name.clone(),
                                    ip: 0,
                                    locals,
                                    code: std::rc::Rc::clone(&func_obj.code),
                                };
                                self.frames.push(new_frame);
                            } else {
                                return Err(format!("Too many args for closure {}: expected {}, got {}", func_name, expected, arg_count));
                            }
                        } else {
                            return Err(format!("Closure function '{}' not found", func_name));
                        }
                    },
                    Value::PartialApp { func_name, captured, applied_args } => {
                        let mut all_args = applied_args;
                        all_args.extend(args);
                        
                        if let Some(func_obj) = self.functions.get(&func_name).cloned() {
                            let expected = func_obj.args.len();
                            let total = all_args.len();
                            
                            if total < expected {
                                self.stack.push(Value::PartialApp {
                                    func_name,
                                    captured,
                                    applied_args: all_args,
                                });
                            } else if total == expected {
                                let mut locals = captured;
                                for (i, param) in func_obj.args.iter().enumerate() {
                                    locals.insert(param.clone(), all_args[i].clone());
                                }
                                let new_frame = CallFrame {
                                    _func_name: func_name.clone(),
                                    ip: 0,
                                    locals,
                                    code: std::rc::Rc::clone(&func_obj.code),
                                };
                                self.frames.push(new_frame);
                            } else {
                                return Err(format!("Too many args for partial {}: expected {}, got {}", func_name, expected, total));
                            }
                        } else {
                            return Err(format!("PartialApp function '{}' not found", func_name));
                        }
                    },
                    _ => return Err(format!("Attempt to call non-function value: {:?}", callee)),
                }
            }
            OpCode::Ret => {
                let ret_val = self.pop()?;
                self.frames.pop();
                self.stack.push(ret_val);
            }
            OpCode::Add => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Self::add(a, b)?);
            }
            OpCode::Sub => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Self::sub(a, b)?);
            }
            OpCode::Mul => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Self::mul(a, b)?);
            }
            OpCode::Div => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Self::div(a, b)?);
            }
            OpCode::Pow => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Self::pow(a, b)?);
            }
            OpCode::MakeVector(size) => {
                let mut elements = Vec::with_capacity(size);
                for _ in 0..size {
                    let val = self.pop()?;
                    match val {
                        Value::Float(f) => elements.push(f),
                        Value::Int(i) => elements.push(i as f64),
                        _ => return Err("Vector elements must be numbers".to_string()),
                    }
                }
                elements.reverse();
                self.stack.push(Value::Vector(std::rc::Rc::new(ndarray::Array1::from(elements))));
            }
            OpCode::MakeList(size) => {
                use std::rc::Rc;
                use crate::domain::ListNode;
                let mut elements = Vec::with_capacity(size);
                for _ in 0..size {
                    elements.push(self.pop()?);
                }
                let mut list: Rc<ListNode> = Rc::new(ListNode::Nil);
                for elem in elements {
                    list = Rc::new(ListNode::Cons(elem, list));
                }
                self.stack.push(Value::List(list));
            }
            OpCode::MakeMatrix(rows, cols) => {
                let size = rows * cols;
                let mut elements = Vec::with_capacity(size);
                for _ in 0..size {
                        let val = self.pop()?;
                        match val {
                        Value::Float(f) => elements.push(f),
                        Value::Int(i) => elements.push(i as f64),
                        _ => return Err("Matrix elements must be numbers".to_string()),
                    }
                }
                elements.reverse();
                
                let arr = ndarray::Array2::from_shape_vec((rows, cols), elements)
                    .map_err(|e| format!("Failed to create matrix: {}", e))?;
                self.stack.push(Value::Matrix(std::rc::Rc::new(arr)));
            }
            OpCode::MakeClosure(func_name, captured_names) => {
                let mut captured = HashMap::new();
                for name in captured_names {
                    if let Some(val) = self.frames[frame_idx].locals.get(&name) {
                        captured.insert(name.clone(), val.clone());
                    } else if let Some(val) = self.globals.get(&name) {
                        captured.insert(name.clone(), val.clone());
                    } else if self.functions.contains_key(&name) || self.native_functions.contains_key(&name) {
                        captured.insert(name.clone(), Value::Function(name));
                    }
                }
                self.stack.push(Value::Closure { func_name, captured });
            }
            OpCode::Jump(offset) => {
                self.frames[frame_idx].ip += offset;
            }
            OpCode::JumpIfFalse(offset) => {
                let cond = self.pop()?;
                if let Value::Bool(b) = cond {
                    if !b {
                            self.frames[frame_idx].ip += offset;
                    }
                } else {
                    return Err("Condition must be a boolean".to_string());
                }
            }
            OpCode::Eq => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Value::Bool(a == b));
            }
            OpCode::Neq => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Value::Bool(a != b));
            }
            OpCode::Lt => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Value::Bool(Self::compare(a, b)? < 0));
            }
            OpCode::Gt => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Value::Bool(Self::compare(a, b)? > 0));
            }
            OpCode::Le => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Value::Bool(Self::compare(a, b)? <= 0));
            }
            OpCode::Ge => {
                let b = self.pop()?;
                let a = self.pop()?;
                self.stack.push(Value::Bool(Self::compare(a, b)? >= 0));
            }
            OpCode::Map => {
                use std::rc::Rc;
                use crate::domain::ListNode;
                
                let func = self.pop()?;
                let list = self.pop()?;
                
                let elements = self.list_to_vec(&list)?;
                
                let mut results = Vec::new();
                for elem in elements {
                    let result = self.call_value(&func, vec![elem])?;
                    results.push(result);
                }
                
                let mut result_list: Rc<ListNode> = Rc::new(ListNode::Nil);
                for elem in results.into_iter().rev() {
                    result_list = Rc::new(ListNode::Cons(elem, result_list));
                }
                self.stack.push(Value::List(result_list));
            }
            OpCode::Filter => {
                use std::rc::Rc;
                use crate::domain::ListNode;
                
                let func = self.pop()?;
                let list = self.pop()?;
                
                let elements = self.list_to_vec(&list)?;
                
                let mut results = Vec::new();
                for elem in elements {
                    let cond = self.call_value(&func, vec![elem.clone()])?;
                    if let Value::Bool(true) = cond {
                        results.push(elem);
                    }
                }
                
                let mut result_list: Rc<ListNode> = Rc::new(ListNode::Nil);
                for elem in results.into_iter().rev() {
                    result_list = Rc::new(ListNode::Cons(elem, result_list));
                }
                self.stack.push(Value::List(result_list));
            }
            OpCode::Foldl => {
                let func = self.pop()?;
                let initial = self.pop()?;
                let list = self.pop()?;
                
                let elements = self.list_to_vec(&list)?;
                
                let mut acc = initial;
                for elem in elements {
                    acc = self.call_value(&func, vec![acc, elem])?;
                }
                self.stack.push(acc);
            }
        }
        Ok(())
    }
}
