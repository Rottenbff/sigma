use std::collections::HashMap;
use crate::ast::{Expr, BinaryOp, TopLevel};
use crate::types::Type;
use crate::domain::Value; // To check literal types

#[derive(Debug, Clone)]
pub struct Substitution(HashMap<usize, Type>);

impl Substitution {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn insert(&mut self, id: usize, t: Type) {
        self.0.insert(id, t);
    }

    pub fn apply(&self, t: &Type) -> Type {
        match t {
            Type::Var(id) => {
                if let Some(s) = self.0.get(id) {
                    self.apply(s) // Recursive apply
                } else {
                    Type::Var(*id)
                }
            },
            Type::List(inner) => Type::List(Box::new(self.apply(inner))),
            Type::Vector(inner) => Type::Vector(Box::new(self.apply(inner))),
            Type::Matrix(inner) => Type::Matrix(Box::new(self.apply(inner))),
            Type::Function(arg, ret) => Type::Function(
                Box::new(self.apply(arg)),
                Box::new(self.apply(ret))
            ),
            _ => t.clone(),
        }
    }
}

pub struct TypeChecker {
    next_id: usize,
    subst: Substitution,
    env: HashMap<String, Type>, // Global environment
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut checker = Self {
            next_id: 0,
            subst: Substitution::new(),
            env: HashMap::new(),
        };
        checker.init_stdlib();
        checker
    }
    
    fn init_stdlib(&mut self) {
        // Arithmetic handling is polymorphic in unification
        
        // map: (a -> b) -> [a] -> [b]
        let a = self.fresh_var();
        let b = self.fresh_var();
        let map_type = Type::function(
            vec![
                Type::function(vec![a.clone()], b.clone()),
                Type::List(Box::new(a.clone()))
            ],
            Type::List(Box::new(b.clone()))
        );
        self.env.insert("map".to_string(), map_type);
        
        // filter: (a -> Bool) -> [a] -> [a]
        let a = self.fresh_var();
        let filter_type = Type::function(
            vec![
                Type::function(vec![a.clone()], Type::Bool),
                Type::List(Box::new(a.clone()))
            ],
            Type::List(Box::new(a.clone()))
        );
        self.env.insert("filter".to_string(), filter_type);

        // Standard Math constants
        self.env.insert("pi".to_string(), Type::Float);

        // Standard Math functions
        let float_unary = Type::function(vec![Type::Float], Type::Float);
        self.env.insert("sin".to_string(), float_unary.clone());
        self.env.insert("cos".to_string(), float_unary.clone());
        self.env.insert("tan".to_string(), float_unary.clone());
        self.env.insert("sqrt".to_string(), float_unary.clone());
        self.env.insert("exp".to_string(), float_unary.clone());
        self.env.insert("log".to_string(), float_unary.clone());

        // List Utils
        // cons: a -> [a] -> [a]
        let a = self.fresh_var();
        let cons_type = Type::function(
            vec![a.clone(), Type::List(Box::new(a.clone()))],
            Type::List(Box::new(a.clone()))
        );
        self.env.insert("cons".to_string(), cons_type);

        // head: [a] -> a
        let a = self.fresh_var();
        self.env.insert("head".to_string(), Type::function(
            vec![Type::List(Box::new(a.clone()))],
            a.clone()
        ));

        // tail: [a] -> [a]
        let a = self.fresh_var();
        self.env.insert("tail".to_string(), Type::function(
            vec![Type::List(Box::new(a.clone()))],
            Type::List(Box::new(a.clone()))
        ));

        // isEmpty: [a] -> Bool
        let a = self.fresh_var();
        self.env.insert("isEmpty".to_string(), Type::function(
            vec![Type::List(Box::new(a.clone()))],
            Type::Bool
        ));

        // Linalg
        // vec: [Float] -> Vector
        self.env.insert("vec".to_string(), Type::function(
            vec![Type::List(Box::new(Type::Float))],
            Type::Vector(Box::new(Type::Float))
        ));

        // mat: [[Float]] -> Matrix
        self.env.insert("mat".to_string(), Type::function(
            vec![Type::List(Box::new(Type::List(Box::new(Type::Float))))],
            Type::Matrix(Box::new(Type::Float))
        ));
        
        // dot: Vector -> Vector -> Float
        self.env.insert("dot".to_string(), Type::function(
            vec![Type::Vector(Box::new(Type::Float)), Type::Vector(Box::new(Type::Float))],
            Type::Float
        ));
        
        // transpose: Matrix -> Matrix
        self.env.insert("transpose".to_string(), Type::function(
            vec![Type::Matrix(Box::new(Type::Float))],
            Type::Matrix(Box::new(Type::Float))
        ));

        // mul: Overloaded (Matrix->Matrix or Matrix->Vector). Use generic for now: a -> b -> c
        let a = self.fresh_var();
        let b = self.fresh_var();
        let c = self.fresh_var();
        self.env.insert("mul".to_string(), Type::function(
            vec![a.clone(), b.clone()],
            c.clone()
        ));

        // abs: a -> Float
        let a = self.fresh_var();
        self.env.insert("abs".to_string(), Type::function(
            vec![a],
            Type::Float
        ));

        // vadd: Vector -> Vector -> Vector
        self.env.insert("vadd".to_string(), Type::function(
            vec![Type::Vector(Box::new(Type::Float)), Type::Vector(Box::new(Type::Float))],
            Type::Vector(Box::new(Type::Float))
        ));
        
        // sigmoid: a -> a (Generic to support Float and Vector)
        let a = self.fresh_var();
        self.env.insert("sigmoid".to_string(), Type::function(
            vec![a.clone()],
            a.clone()
        ));
    }
    
    fn fresh_var(&mut self) -> Type {
        let t = Type::Var(self.next_id);
        self.next_id += 1;
        t
    }

    pub fn check_program(&mut self, program: &Vec<TopLevel>) -> Result<(), String> {
        // Pass 1: Signatures
        let mut func_sigs: HashMap<String, (Vec<Type>, Type)> = HashMap::new();
        
        for item in program {
            match item {
                TopLevel::FunctionDef(def) => {
                    let mut arg_types = Vec::new();
                    for _ in &def.args {
                        arg_types.push(self.fresh_var());
                    }
                    let ret_type = self.fresh_var();
                    
                    let func_type = Type::function(arg_types.clone(), ret_type.clone());
                    self.env.insert(def.name.clone(), func_type);
                    
                    func_sigs.insert(def.name.clone(), (arg_types, ret_type));
                }
            }
        }
        
        // Pass 2: Inference
        for item in program {
            match item {
                TopLevel::FunctionDef(def) => {
                     let (arg_types, ret_type) = func_sigs.get(&def.name).unwrap();
                     
                     let mut local_env = self.env.clone();
                     for (i, arg_name) in def.args.iter().enumerate() {
                         local_env.insert(arg_name.clone(), arg_types[i].clone());
                     }
                     
                     let body_type = self.infer_expr(&def.body, &mut local_env)?;
                     self.unify(&body_type, ret_type)?;
                }
            }
        }
        Ok(())
    }

    fn infer_expr(&mut self, expr: &Expr, env: &mut HashMap<String, Type>) -> Result<Type, String> {
        match expr {
            Expr::Literal(val) => match val {
                Value::Int(_) => Ok(Type::Int),
                Value::Float(_) => Ok(Type::Float),
                Value::Bool(_) => Ok(Type::Bool),
                Value::Complex(_) => Ok(Type::Complex),
                _ => Ok(Type::Unit), // Fallback
            },
            Expr::Var(name) => {
                if let Some(t) = env.get(name) {
                    Ok(self.subst.apply(t)) // Always apply subst to get latest knowledge
                } else if let Some(t) = self.env.get(name) {
                    Ok(self.subst.apply(t))
                } else {
                    Err(format!("Unknown variable: {}", name))
                }
            },
            Expr::BinaryOp(lhs, op, rhs) => {
                let t_lhs = self.infer_expr(lhs, env)?;
                let t_rhs = self.infer_expr(rhs, env)?;
                
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow => {
                        self.unify(&t_lhs, &t_rhs)?;
                        Ok(self.subst.apply(&t_lhs))
                    },
                    BinaryOp::Eq | BinaryOp::Neq => {
                        self.unify(&t_lhs, &t_rhs)?;
                        Ok(Type::Bool)
                    },
                    BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge => {
                        // Ordered types only? Int/Float
                        self.unify(&t_lhs, &t_rhs)?; // Must compare same types
                        Ok(Type::Bool)
                    },
                    BinaryOp::Pipe => {
                        // lhs |> rhs  == rhs(lhs)
                        // rhs must be Function(lhs -> Ret)
                        let t_ret = self.fresh_var();
                        let t_func = Type::Function(Box::new(t_lhs), Box::new(t_ret.clone()));
                        self.unify(&t_rhs, &t_func)?;
                        Ok(self.subst.apply(&t_ret))
                    },
                    BinaryOp::Compose => {
                         // lhs . rhs == \x -> lhs(rhs(x))
                         // lhs: B -> C
                         // rhs: A -> B
                         // result: A -> C
                         
                         let a = self.fresh_var();
                         let b = self.fresh_var();
                         let c = self.fresh_var();
                         
                         let t_lhs_expect = Type::Function(Box::new(b.clone()), Box::new(c.clone()));
                         let t_rhs_expect = Type::Function(Box::new(a.clone()), Box::new(b.clone()));
                         
                         self.unify(&t_lhs, &t_lhs_expect)?;
                         self.unify(&t_rhs, &t_rhs_expect)?;
                         
                         Ok(Type::Function(Box::new(self.subst.apply(&a)), Box::new(self.subst.apply(&c))))
                    },
                    _ => Err(format!("Operator {:?} not fully implemented in TyChecker", op)),
                }
            },
            Expr::List(elements) => {
                let elem_type = self.fresh_var();
                for e in elements {
                    let t = self.infer_expr(e, env)?;
                    self.unify(&elem_type, &t)?;
                }
                Ok(Type::List(Box::new(self.subst.apply(&elem_type))))
            },
             Expr::Vector(elements) => {
                // Enforce vector of floats for now
                for e in elements {
                    let t = self.infer_expr(e, env)?;
                    self.unify(&Type::Float, &t)?; // Strict: Must be Float (or Int if promoted)
                }
                Ok(Type::Vector(Box::new(Type::Float)))
            },
            Expr::Matrix(rows) => {
                 // Enforce matrix of floats
                 for row in rows {
                     for e in row {
                         let t = self.infer_expr(e, env)?;
                         self.unify(&Type::Float, &t)?;
                     }
                 }
                 Ok(Type::Matrix(Box::new(Type::Float)))
            },
            Expr::If(cond, then_b, else_b) => {
                let t_cond = self.infer_expr(cond, env)?;
                self.unify(&t_cond, &Type::Bool)?;
                
                let t_then = self.infer_expr(then_b, env)?;
                let t_else = self.infer_expr(else_b, env)?;
                self.unify(&t_then, &t_else)?;
                
                Ok(self.subst.apply(&t_then))
            },
            Expr::Let(name, init, body) => {
                let t_init = self.infer_expr(init, env)?;
                // Generic let-polymorphism would generalize here.
                // For simple HM, we just add to env.
                let mut new_env = env.clone();
                new_env.insert(name.clone(), t_init);
                self.infer_expr(body, &mut new_env)
            },
            Expr::Lambda(args, body) => {
                 let mut new_env = env.clone();
                 let mut arg_types = Vec::new();
                 
                 for arg in args {
                     let v = self.fresh_var();
                     new_env.insert(arg.clone(), v.clone());
                     arg_types.push(v);
                 }
                 
                 let body_type = self.infer_expr(body, &mut new_env)?;
                 
                 // Construct function type
                 Ok(Type::function(arg_types, body_type))
            },
            Expr::Call(name, args) => {
                // Look up function
                let func_type = if let Some(t) = env.get(name) {
                     self.subst.apply(t)
                } else if let Some(t) = self.env.get(name) {
                    // Generic instantiation
                    self.instantiate(t.clone())
                } else {
                     return Err(format!("Unknown function in call: {}", name));
                };
                
                let ret_type = self.fresh_var();
                
                // Construct function chain
                let mut call_chain = ret_type.clone();
                for arg_expr in args.iter().rev() {
                    let t_arg = self.infer_expr(arg_expr, env)?;
                    call_chain = Type::Function(Box::new(t_arg), Box::new(call_chain));
                }
                
                self.unify(&func_type, &call_chain)?;
                Ok(self.subst.apply(&ret_type))
            },
             _ => Ok(Type::Unit), // Fallback for unimplemented
        }
    }
    
    // Replace all Vars in t with new fresh vars (Generic Instantiation)
    fn instantiate(&mut self, t: Type) -> Type {
         // In a full HM, we'd only instantiate "Quantified" variables. 
         // Here we assume all Vars in global env types are generic.
         // We need a mapping from old_id -> new_id
         let mut map = HashMap::new();
         self.instantiate_recursive(&t, &mut map)
    }
    
    fn instantiate_recursive(&mut self, t: &Type, map: &mut HashMap<usize, Type>) -> Type {
        match t {
            Type::Var(id) => {
                if let Some(new_var) = map.get(id) {
                    new_var.clone()
                } else {
                    let new_var = self.fresh_var();
                    map.insert(*id, new_var.clone());
                    new_var
                }
            },
            Type::List(inner) => Type::List(Box::new(self.instantiate_recursive(inner, map))),
            Type::Function(a, b) => Type::Function(
                Box::new(self.instantiate_recursive(a, map)),
                Box::new(self.instantiate_recursive(b, map))
            ),
             _ => t.clone()
        }
    }

    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), String> {
        let t1 = self.subst.apply(t1);
        let t2 = self.subst.apply(t2);

        if t1 == t2 { return Ok(()); }

        match (t1, t2) {
            (Type::Var(id), t) | (t, Type::Var(id)) => {
                // Occurs check? Skip for simplicity unless we get infinite loops
                self.subst.insert(id, t);
                Ok(())
            },
            (Type::List(a), Type::List(b)) => self.unify(&a, &b),
            (Type::Function(a1, r1), Type::Function(a2, r2)) => {
                self.unify(&a1, &a2)?;
                self.unify(&r1, &r2)
            },
            (Type::Int, Type::Int) => Ok(()),
            (Type::Float, Type::Float) => Ok(()),
            (Type::Complex, Type::Complex) => Ok(()),
            
            // Numeric Promotion
            (Type::Int, Type::Float) => Ok(()),
            (Type::Float, Type::Int) => Ok(()),
            (Type::Int, Type::Complex) => Ok(()),
            (Type::Complex, Type::Int) => Ok(()),
            (Type::Float, Type::Complex) => Ok(()),
            (Type::Complex, Type::Float) => Ok(()),
            (t1, t2) => Err(format!("Type Mismatch: Expected {}, got {}", t1, t2)),
        }
    }
}
