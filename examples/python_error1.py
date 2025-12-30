#!/usr/bin/env python3
"""A script designed to produce a confusing error message."""

class MetaConfusion(type):
    def __getattribute__(cls, name):
        if name == "value":
            raise AttributeError(
                f"'NoneType' object has no attribute 'get' "
                f"(but actually the problem is in __getattribute__ of {cls.__name__})"
            )
        return super().__getattribute__(name)

class DataProcessor(metaclass=MetaConfusion):
    value = 42

    def process(self):
        # Accessing via the class, not instance, triggers the metaclass trap
        return DataProcessor.value

def recursive_wrapper(depth=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if depth > 0:
                return recursive_wrapper(depth - 1)(func)(*args, **kwargs)
            return func(*args, **kwargs)
        wrapper.__name__ = f"wrapper_level_{depth}_for_{func.__name__}"
        return wrapper
    return decorator

@recursive_wrapper()
def fetch_user_data(user_id):
    config = None
    # This looks like it should fail on config.get(), but that's not the real error
    database_url = config.get("db_url") if config else "default"

    # The real error is hidden here
    processor = DataProcessor()
    return processor.process()

class AsyncishResult:
    def __init__(self, callback):
        self._callback = callback
        self._value = None

    def __bool__(self):
        # Surprise! Checking truthiness triggers the callback
        self._value = self._callback()
        return self._value is not None

    def unwrap(self):
        return self._value

def main():
    result = AsyncishResult(lambda: fetch_user_data(12345))

    # This innocent-looking if statement triggers everything
    if result:
        print(f"Got result: {result.unwrap()}")
    else:
        print("No result")

if __name__ == "__main__":
    main()
