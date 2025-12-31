pub mod domain;
pub mod ast;
pub mod parser;
pub mod vm;
pub mod compiler;
pub mod stdlib;
pub mod types;
pub mod type_checker;

use chumsky::Parser; 
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple, PyBool};
use numpy::{ToPyArray, PyArrayMethods};
use std::collections::HashMap;
use std::rc::Rc;

use domain::Value;
use vm::{VM, FunctionObj};
use compiler::Compiler;
use type_checker::TypeChecker;

/// Main entry point for Python integration
#[pyclass(unsendable)]
struct SigmaContext {
    functions: HashMap<String, FunctionObj>,
    globals: HashMap<String, Value>,
    native_funcs: HashMap<String, vm::NativeFunc>,
}

#[pymethods]
impl SigmaContext {
    #[new]
    fn new() -> Self {
        let (native_funcs, globals) = stdlib::create_stdlib();
        SigmaContext {
            functions: HashMap::new(),
            globals,
            native_funcs,
        }
    }

    /// Loads a Sigma file
    fn load(&mut self, path: String) -> PyResult<()> {
        let source = std::fs::read_to_string(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read file '{}': {}", path, e)))?;
        self.eval(source)
    }

    /// Evaluates Sigma source code
    fn eval(&mut self, source: String) -> PyResult<()> {
        let normalized = source.replace("\r\n", "\n");
        let chunks: Vec<&str> = normalized.split("\n\n").collect();
        let mut program = Vec::new();

        for chunk in chunks {
            if chunk.trim().is_empty() { continue; }
            match parser::parser().parse(chunk) {
                Ok(mut items) => program.append(&mut items),
                Err(errs) => return Err(pyo3::exceptions::PySyntaxError::new_err(format!("{:?}", errs))),
            }
        }

        if !program.is_empty() {
            // Type check
            let mut checker = TypeChecker::new();
            // Context-local check
            if let Err(e) = checker.check_program(&program) {
                 return Err(pyo3::exceptions::PyTypeError::new_err(e));
            }

            // Compile
            let new_functions = Compiler::compile_program(program);
            
            // Merge into context
            self.functions.extend(new_functions);
        }
        
        Ok(())
    }

    /// Dynamic function access
    fn __getattr__(&self, py: Python, name: String) -> PyResult<Py<PyAny>> {
        if self.functions.contains_key(&name) {
            // Return a wrapper callable
            let wrapper = SigmaCallable {
                ctx: self.clone_ref(),
                func_name: name,
            };
            Ok(Bound::new(py, wrapper)?.into_any().unbind())
        } else {
            Err(pyo3::exceptions::PyAttributeError::new_err(format!("Function '{}' not found", name)))
        }
    }
}

impl SigmaContext {
    fn clone_ref(&self) -> SigmaContext {
        // Snapshot current state
        SigmaContext {
            functions: self.functions.clone(),
            globals: self.globals.clone(),
            native_funcs: self.native_funcs.clone(),
        }
    }
}


#[pyclass(unsendable)]
struct SigmaCallable {
    ctx: SigmaContext,
    func_name: String,
}

#[pymethods]
impl SigmaCallable {
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let mut vm_args = Vec::new();
        for arg in args {
            vm_args.push(py_to_value(&arg)?);
        }

        let mut vm = VM::new(
            self.ctx.functions.clone(),
            self.ctx.native_funcs.clone(),
            self.ctx.globals.clone()
        );

        match vm.run_function(&self.func_name, vm_args) {
            Ok(val) => value_to_py(py, val),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
}

// Helpers

fn value_to_py(py: Python, val: Value) -> PyResult<Py<PyAny>> {
    match val {
        Value::Int(i) => Ok(i.into_pyobject(py).unwrap().into_any().unbind()),
        Value::Float(f) => Ok(f.into_pyobject(py).unwrap().into_any().unbind()),
        Value::Bool(b) => Ok(PyBool::new(py, b).as_any().clone().unbind()),
        Value::Complex(c) => Ok(c.into_pyobject(py).unwrap().into_any().unbind()),
        Value::Vector(v) => {
            // Zero-copy conversion
            Ok(v.as_ref().to_pyarray(py).into_any().unbind())
        },
        Value::Matrix(m) => {
            Ok(m.as_ref().to_pyarray(py).into_any().unbind())
        },
        Value::List(node) => {
            // Convert linked list to Python list
            let mut elements = Vec::new();
            let mut curr = node.as_ref();
            while let domain::ListNode::Cons(v, next) = curr {
                elements.push(value_to_py(py, v.clone())?);
                curr = next.as_ref();
            }
            Ok(elements.into_pyobject(py).unwrap().into_any().unbind())
        },
        _ => Ok(format!("{}", val).into_pyobject(py).unwrap().into_any().unbind()),
    }
}

fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float(f));
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(c) = obj.extract::<num_complex::Complex64>() {
        return Ok(Value::Complex(c));
    }
    
    // Numpy checks
    if let Ok(array) = obj.downcast::<numpy::PyArray1<f64>>() {
        let v = array.to_owned_array();
        return Ok(Value::Vector(Rc::new(v)));
    }
    if let Ok(array) = obj.downcast::<numpy::PyArray2<f64>>() {
        let m = array.to_owned_array();
        return Ok(Value::Matrix(Rc::new(m)));
    }

    // List
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut val_list = Rc::new(domain::ListNode::Nil);
        for item in list.iter().rev() {
            let v = py_to_value(&item)?;
            val_list = Rc::new(domain::ListNode::Cons(v, val_list));
        }
        return Ok(Value::List(val_list));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!("Unsupported type: {}", obj)))
}

// Module definition
#[pymodule]
fn sigma(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SigmaContext>()?;
    Ok(())
}

pub fn run_file(source: &str) -> String {
    let mut ctx = SigmaContext::new();
    match ctx.eval(source.to_string()) {
        Ok(_) => {
             if ctx.functions.contains_key("main") {
                 let mut vm = VM::new(ctx.functions, ctx.native_funcs, ctx.globals);
                 match vm.run_function("main", vec![]) {
                     Ok(val) => format!("{}", val),
                     Err(e) => format!("Runtime Error: {}", e),
                 }
             } else {
                 "Loaded successfully (no main)".to_string()
             }
        },
        Err(e) => format!("Error: {}", e)
    }
}
