#!/bin/bash

# Run tests for backend-online
echo "Running tests for backend-online..."
cd /app
python -m pytest /app/tests -v --cov=/app/app

# Generate coverage report
python -m pytest /app/tests -v --cov=/app/app --cov-report=html:/app/coverage

echo "Tests completed. Coverage report available in /app/coverage directory."
