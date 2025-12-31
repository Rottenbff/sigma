# Sigma User Guide

Welcome to the **Sigma** User Guide. This document bridges the gap between the formal specification and practical usage, teaching you how to write effective Sigma code.

## 1. Getting Started

A Sigma program is a series of function definitions. The entry point is often a function named `main` if running via CLI, or any function you choose to export when using Python.

### Hello World (Math Style)
Sigma is math-focused, so "Hello World" is usually a calculation.
```haskell
add x y = x + y

main = add 10 32
```

## 2. Syntax & Concepts

### 2.1 Variables & Immutability
Sigma is **purely functional**. You don't "change" variables; you define bindings.
```haskell
# Correct
x = 10
y = x + 5

# Incorrect - Reassignment is not allowed
x = 10
x = 20 # Error!
```

### 2.2 Collections
Sigma supports lists (linked) and vectors (contiguous arrays).

*   **Lists**: Good for recursion and variable length.
    ```haskell
    my_list = [1, 2, 3, 4]
    ```
*   **Vectors**: optimized for math.
    ```haskell
    my_vec = vec [1.0, 2.0, 3.0]
    ```
*   **Matrices**: optimized 2D arrays.
    ```haskell
    my_mat = mat [[1.0, 0.0], [0.0, 1.0]]
    ```

### 2.3 Functions & Currying
All functions are "curried". `f x y` is actually `(f x) y`.
```haskell
multiply a b = a * b

double = multiply 2  # Partial application!
result = double 10   # Returns 20
```

### 2.4 Lambdas (Anonymous Functions)
Use `\` (backslash) to define quick throw-away functions, often for `map`.
```haskell
# Square every number in a list
squares = map (\x -> x * x) [1, 2, 3]
```

## 3. Control Flow

### If-Then-Else
Standard conditional logic.
```haskell
abs x = if x < 0 then 0 - x else x
```

### Guards
Great for multiple conditions (piecewise functions).
```haskell
sign x = guard
  | x > 0 -> 1
  | x < 0 -> -1
  | otherwise -> 0
```

## 4. Standard Library Reference

### Math
*   `pi` : 3.14159...
*   `sin`, `cos`, `tan` : Trigonometry
*   `sqrt`, `exp`, `log` : Roots and Exponentials
*   `abs` : Absolute value (works on Complex numbers too!)

### Linear Algebra
*   `vec [list]` : Convert list to Vector
*   `mat [[list]]` : Convert list of lists to Matrix
*   `dot v1 v2` : Dot product
*   `mul A B` : Matrix multiplication (or Matrix * Vector)
*   `vadd v1 v2` : Vector addition
*   `transpose M` : Matrix transpose
*   `sigmoid v` : Element-wise Sigmoid activation

### List Utilities
*   `map f list` : Apply `f` to every element
*   `head list` : First element
*   `tail list` : Rest of the list
*   `cons x list` : Prepend `x` to `list`
*   `isEmpty list` : Check if empty

## 5. Python Interop

Use the `sigma` python package to load and run your code.

```python
import sigma
ctx = sigma.load("math_lib.sg")
result = ctx.my_function(10)
```
