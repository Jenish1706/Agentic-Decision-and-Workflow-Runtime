"""Validates that all relative links in README.md and docs/ resolve to real files."""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

def check_file(md_path: Path) -> list[str]:
    errors = []
    content = md_path.read_text(encoding="utf-8")
    
    # Remove code blocks so code snippets don't look like links
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    
    for match in LINK_PATTERN.finditer(content):
        text, target = match.group(1), match.group(2).strip()
        
        # Skip external URLs, mailto, etc.
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
            
        # Strip anchor fragment
        clean_target = target.split("#")[0].strip()
        if not clean_target:
            continue
            
        # Resolve relative to the markdown file or relative to repository root
        cand1 = (md_path.parent / clean_target).resolve()
        cand2 = (ROOT / clean_target).resolve()
        
        if not (cand1.exists() or cand2.exists()):
            errors.append(f"{md_path.relative_to(ROOT)}: Link to '{target}' does not resolve to an existing file.")
            
    return errors

def main():
    files_to_check = [ROOT / "README.md"] + list((ROOT / "docs").glob("**/*.md"))
    all_errors = []
    
    for md_file in files_to_check:
        if md_file.is_file():
            errs = check_file(md_file)
            all_errors.extend(errs)
            
    if all_errors:
        print(f"FAILED: Found {len(all_errors)} broken links:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"PASSED: All relative links in {len(files_to_check)} markdown files resolve successfully.")

if __name__ == "__main__":
    main()
