#!/usr/bin/env python3
"""RecursionError - stack overflow, Python style."""

def fibonacci(n):
    # Missing base case for n <= 1
    # This will recurse forever for positive n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    # Oops, wrong comparison - should be n <= 1
    if n == 0:
        return 1
    # For negative numbers, this recurses forever
    return n * factorial(n - 1)

# Both of these will cause RecursionError
print(f"fib(10) = {fibonacci(10)}")
