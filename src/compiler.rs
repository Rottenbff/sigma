use crate::ast::{BinaryOp, Expr, TopLevel};
use crate::vm::{OpCode, FunctionObj};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static GLOBAL_LAMBDA_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub struct Compiler {
    // Current compilation context
    instructions: Vec<OpCode>,
    lambdas: Vec<FunctionObj>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            lambdas: Vec::new(),
        }
    }

    /// Compiles a program to a map of FunctionObjs
    pub fn compile_program(program: Vec<TopLevel>) -> HashMap<String, FunctionObj> {
        let mut functions = HashMap::new();
        
        for item in program {
            match item {
                TopLevel::FunctionDef(def) => {
                    let mut compiler = Compiler::new();
                    compiler.compile_expr(def.body);
                    compiler.emit(OpCode::Ret);
                    
                    functions.insert(def.name.clone(), FunctionObj {
                        name: def.name,
                        args: def.args,
                        code: std::rc::Rc::new(compiler.instructions),
                    });
                    
                    // Lambdas
                    for lambda in compiler.lambdas {
                        functions.insert(lambda.name.clone(), lambda);
                    }
                }
            }
        }
        functions
    }

    // Backward compatibility for single expressions
    pub fn compile(mut self, expr: Expr) -> (Vec<OpCode>, Vec<FunctionObj>) {
        self.compile_expr(expr);
        self.emit(OpCode::Ret);
        (self.instructions, self.lambdas)
    }

    fn compile_expr(&mut self, expr: Expr) {
        match expr {
            Expr::Literal(v) => { self.emit(OpCode::Push(v)); },
            Expr::Var(name) => { self.emit(OpCode::Load(name)); },
            Expr::BinaryOp(lhs, op, rhs) => {
                // Pipe: x |> f => f(x)
                if matches!(op, BinaryOp::Pipe) {
                    // Compile: push function, push arg, call
                    self.compile_expr(*rhs); // function
                    self.compile_expr(*lhs); // argument
                    self.emit(OpCode::Call(1));
                    return;
                }
                
                // Compose: f . g => \x -> f(g(x))
                if matches!(op, BinaryOp::Compose) {
                    // Generate a synthetic lambda
                    let id = GLOBAL_LAMBDA_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let _lambda_name = format!("__compose_{}", id);
                    
                    // Capture f/g and apply: g(x) |> f
                    
                    // Alternative: compile to a proper lambda that uses pipe
                    // f . g = \x -> x |> g |> f
                    let composed_body = Expr::BinaryOp(
                        Box::new(Expr::BinaryOp(
                            Box::new(Expr::Var("__x".to_string())),
                            BinaryOp::Pipe,
                            rhs
                        )),
                        BinaryOp::Pipe,
                        lhs
                    );
                    let lambda = Expr::Lambda(vec!["__x".to_string()], Box::new(composed_body));
                    self.compile_expr(lambda);
                    return;
                }
                
                self.compile_expr(*lhs);
                self.compile_expr(*rhs);
                match op {
                    BinaryOp::Add => { self.emit(OpCode::Add); },
                    BinaryOp::Sub => { self.emit(OpCode::Sub); },
                    BinaryOp::Mul => { self.emit(OpCode::Mul); },
                    BinaryOp::Div => { self.emit(OpCode::Div); },
                    BinaryOp::Pow => { self.emit(OpCode::Pow); },
                    BinaryOp::Eq => { self.emit(OpCode::Eq); },
                    BinaryOp::Neq => { self.emit(OpCode::Neq); },
                    BinaryOp::Lt => { self.emit(OpCode::Lt); },
                    BinaryOp::Gt => { self.emit(OpCode::Gt); },
                    BinaryOp::Le => { self.emit(OpCode::Le); },
                    BinaryOp::Ge => { self.emit(OpCode::Ge); },
                    BinaryOp::And | BinaryOp::Or => unimplemented!("Logic And/Or"),
                    _ => unimplemented!("Op {:?} not implemented yet", op),
                }
            }
            Expr::If(cond, then_branch, else_branch) => {
                self.compile_expr(*cond);
                // JumpIfFalse placeholder
                let jump_else_idx = self.emit(OpCode::JumpIfFalse(0));
                
                self.compile_expr(*then_branch);
                // Jump placeholder (skip else)
                let jump_end_idx = self.emit(OpCode::Jump(0));
                
                let else_start = self.instructions.len();
                self.compile_expr(*else_branch);
                let end = self.instructions.len();
                
                // Patch jumps
                let offset_else = else_start - (jump_else_idx + 1);
                self.instructions[jump_else_idx] = OpCode::JumpIfFalse(offset_else);
                
                let offset_end = end - (jump_end_idx + 1);
                self.instructions[jump_end_idx] = OpCode::Jump(offset_end);
            }
            Expr::Call(name, args) => {
                // Built-in HOFs
                match name.as_str() {
                    "map" if args.len() == 2 => {
                        // map func list -> [list, func] then Map opcode
                        self.compile_expr(args[1].clone()); // list first
                        self.compile_expr(args[0].clone()); // func second
                        self.emit(OpCode::Map);
                    }
                    "filter" if args.len() == 2 => {
                        self.compile_expr(args[1].clone()); // list
                        self.compile_expr(args[0].clone()); // func
                        self.emit(OpCode::Filter);
                    }
                    "foldl" if args.len() == 3 => {
                        // foldl func init list -> [list, init, func] then Foldl
                        self.compile_expr(args[2].clone()); // list
                        self.compile_expr(args[1].clone()); // init
                        self.compile_expr(args[0].clone()); // func
                        self.emit(OpCode::Foldl);
                    }
                    _ => {
                        // Regular function call
                        self.emit(OpCode::Load(name));
                        let arg_count = args.len();
                        for arg in args {
                            self.compile_expr(arg);
                        }
                        self.emit(OpCode::Call(arg_count));
                    }
                }
            }
            Expr::List(elements) => {
                let size = elements.len();
                for elem in elements {
                    self.compile_expr(elem);
                }
                self.emit(OpCode::MakeList(size));
            }
            Expr::Vector(elements) => {
                let size = elements.len();
                for elem in elements {
                    self.compile_expr(elem);
                }
                self.emit(OpCode::MakeVector(size));
            }
            Expr::Matrix(rows) => {
                let row_count = rows.len();
                let col_count = if row_count > 0 { rows[0].len() } else { 0 };
                // Flattening
                for row in rows {
                    for elem in row {
                        self.compile_expr(elem);
                    }
                }
                self.emit(OpCode::MakeMatrix(row_count, col_count));
            }
            Expr::Let(name, init, body) => {
                self.compile_expr(*init);
                self.emit(OpCode::Store(name));
                self.compile_expr(*body);
            }
            Expr::Lambda(args, body) => {
                let id = GLOBAL_LAMBDA_COUNTER.fetch_add(1, Ordering::Relaxed);
                let lambda_name = format!("__lambda_{}", id);
                
                // Closure capture
                let body_vars = Self::free_vars(&body);
                let arg_set: std::collections::HashSet<String> = args.iter().cloned().collect();
                let captured: Vec<String> = body_vars.difference(&arg_set).cloned().collect();
                
                // Compile body in a new compiler context
                let mut sub_compiler = Compiler::new();
                sub_compiler.compile_expr(*body);
                sub_compiler.emit(OpCode::Ret);
                
                // Collect sub-lambdas
                self.lambdas.extend(sub_compiler.lambdas);
                
                // Create FunctionObj
                let func_obj = FunctionObj {
                    name: lambda_name.clone(),
                    args,
                    code: std::rc::Rc::new(sub_compiler.instructions),
                };
                
                self.lambdas.push(func_obj);
                
                // Emit MakeClosure to capture environment at runtime
                self.emit(OpCode::MakeClosure(lambda_name, captured));
            }
            Expr::Guarded(guards) => {
                // Desugar to nested if/else
                let desugared = self.desugar_guards(guards);
                self.compile_expr(desugared);
            }
        }
    }

    fn emit(&mut self, op: OpCode) -> usize {
        self.instructions.push(op);
        self.instructions.len() - 1
    }
    
    /// Returns all free variables in an expression
    fn free_vars(expr: &Expr) -> std::collections::HashSet<String> {
        use std::collections::HashSet;
        match expr {
            Expr::Literal(_) => HashSet::new(),
            Expr::Var(name) => {
                let mut set = HashSet::new();
                set.insert(name.clone());
                set
            }
            Expr::BinaryOp(lhs, _, rhs) => {
                let mut set = Self::free_vars(lhs);
                set.extend(Self::free_vars(rhs));
                set
            }
            Expr::Call(name, args) => {
                let mut set = HashSet::new();
                set.insert(name.clone());
                for arg in args {
                    set.extend(Self::free_vars(arg));
                }
                set
            }
            Expr::If(cond, then_b, else_b) => {
                let mut set = Self::free_vars(cond);
                set.extend(Self::free_vars(then_b));
                set.extend(Self::free_vars(else_b));
                set
            }
            Expr::Let(name, init, body) => {
                let mut set = Self::free_vars(init);
                let mut body_vars = Self::free_vars(body);
                body_vars.remove(name); // `name` is bound in body
                set.extend(body_vars);
                set
            }
            Expr::Lambda(params, body) => {
                let mut set = Self::free_vars(body);
                for p in params {
                    set.remove(p); // params are bound
                }
                set
            }
            Expr::Vector(elems) | Expr::List(elems) => {
                let mut set = HashSet::new();
                for e in elems {
                    set.extend(Self::free_vars(e));
                }
                set
            }
            Expr::Matrix(rows) => {
                let mut set = HashSet::new();
                for row in rows {
                    for e in row {
                        set.extend(Self::free_vars(e));
                    }
                }
                set
            }
            Expr::Guarded(guards) => {
                let mut set = HashSet::new();
                for g in guards {
                    set.extend(Self::free_vars(&g.condition));
                    set.extend(Self::free_vars(&g.result));
                }
                set
            }
        }
    }
    
    /// Desugar guards to nested if/else expressions
    /// | cond1 = e1 | cond2 = e2 | otherwise = e3
    /// becomes: if cond1 then e1 else (if cond2 then e2 else e3)
    fn desugar_guards(&self, guards: Vec<crate::ast::Guard>) -> Expr {
        use crate::domain::Value;
        
        // Process reverse for nesting
        let guards_rev: Vec<_> = guards.into_iter().rev().collect();
        
        // Fallback result
        let mut result = Expr::Literal(Value::Int(0)); 
        
        for guard in guards_rev {
            // Check if condition is 'otherwise' (treat as always true)
            let is_otherwise = match &guard.condition {
                Expr::Var(name) if name == "otherwise" => true,
                _ => false,
            };
            
            if is_otherwise {
                // 'otherwise' means this becomes the else branch
                result = guard.result;
            } else {
                // Regular guard: if condition then result else (previous result)
                result = Expr::If(
                    Box::new(guard.condition),
                    Box::new(guard.result),
                    Box::new(result)
                );
            }
        }
        
        result
    }
}

