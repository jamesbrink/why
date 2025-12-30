#!/usr/bin/env python3
"""Circular import error - a Python rite of passage."""

# This simulates what happens when module A imports B and B imports A

# Pretend this is module_a.py importing from module_b
# which in turn tries to import from module_a

import sys
from types import ModuleType

# Create a fake circular import scenario
fake_module = ModuleType("circular_dependency")
fake_module.__dict__["__name__"] = "circular_dependency"

# Simulate the error that occurs during circular imports
exec("""
from circular_dependency import helper  # This would fail in real circular import

def process():
    return helper.do_thing()
""", {"__name__": "__main__"})
