"""Verifies that all files and functions referenced in 05_CODE_WALKTHROUGH.md exist in the codebase."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WALKTHROUGH = ROOT / "docs" / "guide" / "05_CODE_WALKTHROUGH.md"

def main():
    content = WALKTHROUGH.read_text(encoding="utf-8")
    
    # 1. Check all file paths mentioned as paths
    # Matches strings like `src/agent_runtime/...` or `domains/...` or `tests/...`
    file_patterns = re.findall(r'`([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9_]+)`', content)
    
    errors = []
    checked_files = set()
    
    for f in file_patterns:
        # Ignore things like .json, .yaml when they are general format names or doc references unless intended
        p = ROOT / f
        if any(f.startswith(prefix) for prefix in ["src/", "domains/", "tests/", "scripts/", "data/"]):
            checked_files.add(f)
            if not p.exists():
                errors.append(f"Referenced file does not exist: {f}")
                
    # 2. Check key function definitions mentioned in WALKTHROUGH
    # Functions mentioned with () e.g. `execute_read_only()` or `resolve_fact()`
    func_patterns = set(re.findall(r'`([a-zA-Z_][a-zA-Z0-9_]*)\(\)`', content))
    
    # Known special syntax or builtins
    skip_funcs = {"print", "len", "set", "range", "round", "super"}
    
    for fn in func_patterns:
        if fn in skip_funcs:
            continue
        # Search for def fn or async def fn across src/ and domains/
        found = False
        for py_file in (ROOT / "src").glob("**/*.py"):
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if f"def {fn}(" in text or f"def {fn} (" in text:
                found = True
                break
        if not found:
            for py_file in (ROOT / "domains").glob("**/*.py"):
                text = py_file.read_text(encoding="utf-8", errors="ignore")
                if f"def {fn}(" in text or f"def {fn} (" in text:
                    found = True
                    break
        if not found:
            for py_file in (ROOT / "scripts").glob("**/*.py"):
                text = py_file.read_text(encoding="utf-8", errors="ignore")
                if f"def {fn}(" in text or f"def {fn} (" in text:
                    found = True
                    break
        if not found:
            errors.append(f"Referenced function '{fn}()' not found in src/, domains/, or scripts/")
            
    if errors:
        print(f"FAILED: Found {len(errors)} walkthrough discrepancies:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"PASSED: All {len(checked_files)} files and {len(func_patterns)} functions in 05_CODE_WALKTHROUGH.md exist.")

if __name__ == "__main__":
    main()
