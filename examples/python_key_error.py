#!/usr/bin/env python3
"""Classic KeyError from missing dictionary key."""

import json

def parse_api_response(data):
    """Parse user data from API response."""
    user = data["user"]
    # API sometimes returns "user_name" instead of "username"
    return {
        "id": user["id"],
        "name": user["username"],  # KeyError here!
        "email": user["email"]
    }

def main():
    # Simulated API response with inconsistent field naming
    api_response = {
        "user": {
            "id": 12345,
            "user_name": "jsmith",  # Oops, should be "username"
            "email": "jsmith@example.com"
        },
        "status": "ok"
    }

    try:
        user_data = parse_api_response(api_response)
        print(f"Welcome, {user_data['name']}!")
    except KeyError as e:
        # Re-raise with traceback for why to analyze
        raise

if __name__ == "__main__":
    main()
