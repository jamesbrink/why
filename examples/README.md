# Error Examples

These scripts produce various error messages that can be analyzed with `why`.

## Usage

Run a script and pipe its stderr to `why`:

```bash
# Python examples
python examples/python_error1.py 2>&1 | why
python examples/python_key_error.py 2>&1 | why
python examples/python_recursion.py 2>&1 | why
python examples/python_import.py 2>&1 | why

# Bash examples
./examples/bash_unbound.sh 2>&1 | why
./examples/shell_command_not_found.sh 2>&1 | why

# JavaScript/Node
node examples/js_undefined.js 2>&1 | why

# Ruby
ruby examples/ruby_nil.rb 2>&1 | why

# Compiled languages (compile first, then run)
gcc examples/c_segfault.c -o /tmp/segfault && /tmp/segfault 2>&1 | why
go run examples/go_nil_panic.go 2>&1 | why
javac examples/java_npe.java && java -cp examples java_npe 2>&1 | why

# Rust (compiler error, no runtime needed)
# Note: .rs.txt extension avoids cargo trying to compile it as an example
rustc examples/rust_borrow.rs.txt 2>&1 | why

# TypeScript (type checker error)
tsc --strict examples/typescript_type_error.ts 2>&1 | why
```

## Examples

| File | Language | Error Type |
|------|----------|------------|
| `python_error1.py` | Python | AttributeError (metaclass trap) |
| `python_key_error.py` | Python | KeyError (missing dict key) |
| `python_recursion.py` | Python | RecursionError (infinite recursion) |
| `python_import.py` | Python | ModuleNotFoundError (circular import) |
| `bash_unbound.sh` | Bash | Unbound variable with `set -u` |
| `shell_command_not_found.sh` | Bash | Command not found (typo) |
| `js_undefined.js` | JavaScript | TypeError (undefined property) |
| `ruby_nil.rb` | Ruby | NoMethodError on nil |
| `c_segfault.c` | C | Segmentation fault (NULL deref) |
| `go_nil_panic.go` | Go | Panic (nil pointer dereference) |
| `java_npe.java` | Java | NullPointerException |
| `rust_borrow.rs.txt` | Rust | Borrow checker (use after move) |
| `typescript_type_error.ts` | TypeScript | Type error (null assignment) |
