# test_interpreter.py
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

# Test a package that was giving you issues
try:
    import uvicorn
    print("✓ uvicorn imported successfully")
except ImportError as e:
    print(f"✗ uvicorn import failed: {e}")