# Sigma Programming Language

> [!WARNING]
> **Heavy Development Phase**: Sigma is currently in active pre-alpha development. Features, syntax, and APIs are subject to breaking changes without notice. Use with caution.

**Sigma** is a high-performance, strictly-typed functional programming language designed specifically for mathematical computation and linear algebra. It combines the safety and elegance of functional programming with the raw speed of Rust and the ease of use of Python.

For a detailed manual and reference, please consult the **[User Guide](USER_GUIDE.md)**.

## Release Cycle

This project follows a **quarterly release cycle**. The repository is updated with substantial new features and stability improvements upon every stable release. Between releases, the `main` branch may contain experimental features.

## Key Features

*   **Math-First Primitives**: `Complex`, `Vector`, and `Matrix` types are built-in first-class citizens, not external libraries.
*   **Functional Core**: Immutable data, higher-order functions (`map`, `filter`, `fold`), and currying by default.
*   **Safety**: Hindley-Milner type inference ensures type safety at compile-time without cluttering your code with annotations.
*   **Rust Runtime**: A stack-based bytecode VM implemented in Rust for performance.
*   **Python Integration**: Seamlessly call Sigma code from Python, passing Numpy arrays with **Zero-Copy** overhead.

## Installation & Building

Sigma is built with Rust. You'll need the latest stable Rust toolchain.

```bash
# Clone the repository
git clone https://github.com/rottenbff/sigma.git
cd sigma

# Build the CLI and Shared Library
cargo build --release
```

## Usage

### Command Line Interface

You can run `.sg` files directly using the CLI:

```bash
# Run a script
cargo run --bin sigma-cli -- path/to/script.sg

# Example
cargo run --bin sigma-cli -- demo_neural.sg
```

### Python Integration

Sigma is designed to be embedded in Python workflows.

1.  Build the project as a dynamic library (creates `sigma.dll` or `sigma.so`).
2.  Ensure the library is in your Python path (or rename to `sigma.pyd` on Windows).

```python
import sigma
import numpy as np

# Load a Sigma script
ctx = sigma.load("my_model.sg")

# Create data (Numpy arrays are passed efficiently)
data = np.array([1.0, 2.0, 3.0])

# Call a function defined in 'my_model.sg'
result = ctx.process_data(data)

print(result)
```

## Language Example

**Signal Processing (Gaussian Smoothing)**

```haskell
# Define a Gaussian function
gaussian sigma x = 
    (1.0 / (sqrt (2.0 * pi) * sigma)) * exp (-(x * x) / (2.0 * sigma * sigma))

# Apply smoothing to a vector
smooth data = 
    let kernel = map (gaussian 1.0) [-2.0, -1.0, 0.0, 1.0, 2.0] in
    # ... logic to convolve kernel with data ...
    data # Placeholder return
```

## Project Structure

*   `src/`: Rust source code for Compiler, VM, and Type Checker.
*   `src/stdlib.rs`: Native implementations of standard library functions.
*   `demo_*.sg`: Example scripts showcasing language features.

## Version History Note
*Development of this project began in 2023.* 
Prior to December 2025, the source code was managed and hosted self-hosted 
using **Clay**, a custom version control system I developed for my local 
Dell Optiplex server. The commit history on GitHub represents the migration 
snapshot and subsequent development.

## License

This project is licensed under the MIT License.
