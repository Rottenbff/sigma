use chumsky::prelude::*;
use crate::ast::{BinaryOp, Expr, FunctionDef, TopLevel};
use crate::domain::Value;

// Expression parser interface
pub fn parse_expr_str(src: &str) -> Result<Expr, Vec<Simple<char>>> {
    expr_parser().parse(src)
}

fn is_reserved(s: &String) -> bool {
    s == "if" || s == "then" || s == "else" || s == "vec" || s == "mat" || s == "True" || s == "False" || s == "let" || s == "in" || s == "guard"
}

fn expr_parser() -> impl Parser<char, Expr, Error = Simple<char>> {
     recursive(|expr| {
        let ident = text::ident()
            .try_map(|s, span| {
                if is_reserved(&s) {
                    Err(Simple::custom(span, "Reserved keyword used as identifier"))
                } else {
                    Ok(s)
                }
            })
            .padded();
            
        let val = text::int(10)
            .then(just('.').then(text::digits(10)).or_not())
            .map(|(int_part, frac_part)| {
                let s = match frac_part {
                    Some((_, frac)) => format!("{}.{}", int_part, frac),
                    None => int_part,
                };
                s
            });

        let number = just('-').or_not()
            .then(val)
            .then(just('i').or_not())
            .try_map(|((sign, s), imaginary), span| {
                let full_str = if sign.is_some() { format!("-{}", s) } else { s };
                let is_imaginary = imaginary.is_some();
                
                if is_imaginary {
                     let val: f64 = full_str.parse()
                        .map_err(|_| Simple::custom(span, "Invalid complex number format"))?;
                     Ok(Expr::Literal(Value::Complex(num_complex::Complex64::new(0.0, val))))
                } else {
                    if full_str.contains('.') {
                        let val = full_str.parse()
                            .map_err(|_| Simple::custom(span, "Invalid float format"))?;
                        Ok(Expr::Literal(Value::Float(val)))
                    } else {
                        let val = full_str.parse()
                            .map_err(|_| Simple::custom(span, "Invalid integer format"))?;
                        Ok(Expr::Literal(Value::Int(val)))
                    }
                }
            })
            .padded();

        // Helper for [e1, e2, ...]
        let bracketed_exprs = expr.clone()
            .separated_by(just(',').padded())
            .allow_trailing()
            .delimited_by(just('['), just(']'))
            .padded();

        let vec_literal = text::keyword("vec")
            .ignore_then(bracketed_exprs.clone())
            .map(Expr::Vector);

        let mat_literal = text::keyword("mat")
            .ignore_then(
                bracketed_exprs.clone()
                    .separated_by(just(',').padded())
                    .allow_trailing()
                    .delimited_by(just('['), just(']'))
                    .padded()
            )
            .map(Expr::Matrix);

        let list_literal = bracketed_exprs.clone().map(Expr::List);

        let boolean = text::keyword("True").to(true)
            .or(text::keyword("False").to(false))
            .map(|b| Expr::Literal(Value::Bool(b)));
            
        let atom = vec_literal
            .or(mat_literal)
            .or(list_literal)
            .or(boolean)
            .or(number)
            .or(ident.map(Expr::Var))
            .or(expr.clone().delimited_by(just('('), just(')')))
            .padded();

        // Function call
        let call = ident
            .then(atom.clone().repeated().at_least(1))
            .map(|(name, args)| Expr::Call(name, args))
            .or(atom); 

        let op = |c: char, op: BinaryOp| just(c).padded().to(op);

        // Composition
        let compose = call.clone()
            .then(
                just('.').padded().to(BinaryOp::Compose)
                .then(call.clone())
                .repeated()
            )
            .foldl(|lhs, (op, rhs)| Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs)));

        // Power
        let power = compose.clone()
            .then(op('^', BinaryOp::Pow).then(compose.clone()).repeated())
            .foldl(|lhs, (op, rhs)| Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs)));

        let product = power.clone()
            .then(op('*', BinaryOp::Mul).or(op('/', BinaryOp::Div)).then(power.clone()).repeated())
            .foldl(|lhs, (op, rhs)| Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs)));

        let sum = product.clone()
            .then(op('+', BinaryOp::Add).or(op('-', BinaryOp::Sub)).then(product.clone()).repeated())
            .foldl(|lhs, (op, rhs)| Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs)));

        let compare = sum.clone()
             .then(
                 just("==").padded().to(BinaryOp::Eq)
                 .or(just("!=").padded().to(BinaryOp::Neq))
                 .or(just("<=").padded().to(BinaryOp::Le))
                 .or(just(">=").padded().to(BinaryOp::Ge))
                 .or(just('<').padded().to(BinaryOp::Lt))
                 .or(just('>').padded().to(BinaryOp::Gt))
                 .then(sum.clone())
                 .repeated()
             )
             .foldl(|lhs, (op, rhs)| Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs)));

        // Pipe operator |> (lowest precedence among arithmetic/logical ops)
        let pipe = compare.clone()
             .then(
                 just("|>").padded().to(BinaryOp::Pipe)
                 .then(compare.clone())
                 .repeated()
             )
             .foldl(|lhs, (op, rhs)| Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs)));

        // Conditionals
        let if_expr = text::keyword("if").padded()
            .ignore_then(expr.clone())
            .then_ignore(text::keyword("then").padded())
            .then(expr.clone())
            .then_ignore(text::keyword("else").padded())
            .then(expr.clone())
            .map(|((cond, then_b), else_b)| Expr::If(Box::new(cond), Box::new(then_b), Box::new(else_b)));

        let let_expr = text::keyword("let").padded()
            .ignore_then(ident)
            .then_ignore(just('=').padded())
            .then(expr.clone())
            .then_ignore(text::keyword("in").padded())
            .then(expr.clone())
            .map(|((name, init), body)| Expr::Let(name, Box::new(init), Box::new(body)));


        let lambda = just('\\').padded()
            .ignore_then(ident.repeated())
            .then_ignore(just("->").padded())
            .then(expr.clone())
            .map(|(args, body)| Expr::Lambda(args, Box::new(body)));

        // Guards
        let guard_clause = just('|').padded()
            .ignore_then(expr.clone())
            .then_ignore(just("->").padded())
            .then(expr.clone())
            .map(|(condition, result)| crate::ast::Guard { condition, result });
        
        let guard_expr = text::keyword("guard").padded()
            .ignore_then(guard_clause.repeated().at_least(1))
            .map(Expr::Guarded);

        if_expr.or(let_expr).or(lambda).or(guard_expr).or(pipe)
    })
}

pub fn parser() -> impl Parser<char, Vec<TopLevel>, Error = Simple<char>> {
    let ident = text::ident()
        .try_map(|s: String, span| {
            if is_reserved(&s) {
                Err(Simple::custom(span, "Reserved keyword used as identifier"))
            } else {
                Ok(s)
            }
        })
        .padded();
    
    // Function definitions
    let function_def = ident.clone()
        .then(ident.repeated()) // args
        .then_ignore(just('=').padded())
        .then(expr_parser())
        .map(|((name, args), body)| TopLevel::FunctionDef(FunctionDef { name, args, body }));

    function_def.repeated().then_ignore(end())
}

