# Operational Runbook: Windows Setup, Pipeline Commands & Troubleshooting

This runbook documents the exact commands needed to set up, run, evaluate, and troubleshoot the runtime on Windows (and POSIX systems).

---

## 1. Environment Setup

### 1.1 Prerequisites
- Python 3.11+ (CPython 3.11.15 x86_64 tested)
- [uv](https://docs.astral.sh/uv/) (v0.11+) or standard `pip`
- Git (v2.40+)
- Windows PowerShell 5.1 or PowerShell Core 7+

### 1.2 Initial Setup Commands
```powershell
# Clone the repository
git clone https://github.com/org/agent-runtime.git
cd agent-runtime

# Create virtual environment and sync dependencies with uv
uv sync

# Set PYTHONPATH in current session
$env:PYTHONPATH="src;."
```

---

## 2. Pipeline Execution Commands & Runtimes

| Task | Command | Expected Runtime | Expected Output |
| :--- | :--- | :--- | :--- |
| **Full Test Suite** | `uv run pytest` | ~90–120s | `272 passed, 7 warnings in ~100s` |
| **Targeted Flow Tests** | `uv run pytest tests/test_credit_risk_flow.py tests/test_enterprise_operations_flow.py` | ~15–25s | All domain flow tests pass |
| **Verify README Numbers** | `uv run python scripts/render_readme.py --check` | ~10–15s | `OK: All documentation tables match verified measurements.` |
| **Regenerate Tables** | `uv run python scripts/render_readme.py` | ~15–20s | Overwrites README & EVALUATION tables with fresh verified numbers |
| **Measure Verifier Benchmark** | `uv run python scripts/measure_verifier.py` | ~8–12s | Console benchmark table across 7 categories (6,000+ trials) |
| **Measure Citation Benchmark** | `uv run python scripts/measure_citations.py` | ~10–15s | False positive & held-out precision/recall outputs |
| **Fresh Clone Sanity Check** | Clone to temp dir & `uv run pytest` | ~90s | Clean environment installation & 272 tests passed |

---

## 3. How to Regenerate All README Numbers

To re-run the benchmark measurements and update all public documentation:
```powershell
# 1. Run verifier measurement
uv run python scripts/measure_verifier.py

# 2. Run citation verification measurement
uv run python scripts/measure_citations.py

# 3. Render verified tables directly into README.md and EVALUATION.md
uv run python scripts/render_readme.py

# 4. Verify that documents match code
uv run python scripts/render_readme.py --check
```

---

## 4. Troubleshooting Real Errors Hit During Development

### 4.1 PowerShell Syntax Error: Token '&&' is not a valid separator
- **Symptom:** `ParserError: The token '&&' is not a valid statement separator in this version.`
- **Cause:** Standard Windows PowerShell 5.1 does not support `&&` chaining.
- **Fix:** Use semicolon `;` as a statement separator instead:
  ```powershell
  # Instead of: cmd1 && cmd2
  cmd1; cmd2
  ```

### 4.2 Windows Asyncio / Socketpair Hang in Sandboxed Execution
- **Symptom:** `pytest` hangs indefinitely during `test_agent_loop.py` or MCP subprocess launches.
- **Cause:** On Windows, Python's `asyncio.ProactorEventLoop` uses local loopback socketpairs for inter-thread wakeups. Inside strict sandbox isolation, local socketpair handshakes can be blocked or delayed.
- **Fix:** Run execution commands with appropriate local permissions or pass unbuffered process flags.

### 4.3 Missing `permit_tools` in Manifest
- **Symptom:** A valid record lookup from an MCP tool produces `REQUIRES_APPROVAL` instead of `ALLOWED`.
- **Cause:** The domain manifest (`domain_manifest.yaml`) did not whitelist the tool name in its `permit_tools` section.
- **Fix:** Check `DomainSpec.permit_tools` for the active domain. Add the authoritative tool name to the manifest:
  ```yaml
  permit_tools:
    - get_account_profile
  ```

### 4.4 Hugging Face Hub Offline Exceptions
- **Symptom:** `test_mcp_retriever.py` crashes in air-gapped CI with `requests.exceptions.ConnectionError`.
- **Cause:** The knowledge server instantiates sentence-transformers on import.
- **Fix:** Wrapped in `pytest.skip(..., allow_module_level=True)` so that air-gapped runs skip gracefully with 0 errors.

### 4.5 Multiline `python -c` Quoting in PowerShell
- **Symptom:** `Missing closing ')' in expression` or syntax error when passing nested quotes to `python -c`.
- **Cause:** PowerShell strips inner quotes from strings passed across process boundaries.
- **Fix:** Write complex scratch scripts into a `.py` file and execute `uv run python script.py` rather than escaping multi-level quotes in inline CLI commands.
