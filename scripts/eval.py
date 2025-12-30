#!/usr/bin/env python3
"""
Evaluation script for the `why` CLI.

Runs through error examples from YAML, pipes them to `why`,
and reports on the quality of explanations with formatted markdown output.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ANSI colors
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    NC = "\033[0m"  # No Color
    # Background
    BG_DARK = "\033[48;5;236m"

    @classmethod
    def disable(cls):
        """Disable colors for non-terminal output."""
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


@dataclass
class ErrorCase:
    """Represents an error test case."""

    id: str
    language: str
    error_type: str
    error_text: str


@dataclass
class EvalResult:
    """Result of evaluating a single error case."""

    case: ErrorCase
    success: bool
    why_output: dict = field(default_factory=dict)
    inference_time_ms: int = 0
    has_summary: bool = False
    has_explanation: bool = False
    has_suggestion: bool = False
    no_error_detected: bool = False
    error_message: str = ""


def load_errors_from_yaml(yaml_path: Path) -> list[ErrorCase]:
    """Load error cases from YAML file."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        ErrorCase(
            id=item["id"],
            language=item["language"],
            error_type=item["error_type"],
            error_text=item["error_text"].rstrip("\n"),
        )
        for item in data
    ]


def run_why(
    why_binary: Path, error_input: str, use_json: bool = True, stats: bool = True
) -> tuple[dict, int]:
    """Run the why CLI with the given error input."""
    cmd = [str(why_binary)]
    if use_json:
        cmd.append("--json")
    if stats:
        cmd.append("--stats")

    start = time.time()
    result = subprocess.run(
        cmd,
        input=error_input,
        capture_output=True,
        text=True,
    )
    elapsed_ms = int((time.time() - start) * 1000)

    if use_json and result.stdout.strip():
        try:
            return json.loads(result.stdout), elapsed_ms
        except json.JSONDecodeError:
            return {"raw_output": result.stdout, "parse_error": True}, elapsed_ms

    return {"raw_output": result.stdout, "stderr": result.stderr}, elapsed_ms


def evaluate_case(case: ErrorCase, why_binary: Path) -> EvalResult:
    """Evaluate a single error case."""
    result = EvalResult(case=case, success=False)

    try:
        why_output, elapsed_ms = run_why(why_binary, case.error_text)
        result.why_output = why_output
        result.inference_time_ms = elapsed_ms

        if why_output.get("no_error"):
            result.no_error_detected = True
            result.success = False
        elif why_output.get("parse_error"):
            result.success = False
            result.error_message = "Failed to parse JSON output"
        else:
            result.has_summary = bool(why_output.get("summary", "").strip())
            result.has_explanation = bool(why_output.get("explanation", "").strip())
            result.has_suggestion = bool(why_output.get("suggestion", "").strip())
            result.success = result.has_summary or result.has_explanation

    except Exception as e:
        result.error_message = f"why failed: {e}"

    return result


def render_markdown(text: str, indent: str = "") -> str:
    """Render markdown text with ANSI colors."""
    c = Colors
    lines = []
    in_code_block = False
    code_block_lines = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Handle code block delimiters
        if stripped.startswith("```"):
            if in_code_block:
                # End code block - render accumulated lines
                for code_line in code_block_lines:
                    lines.append(f"{indent}  {c.CYAN}{code_line}{c.NC}")
                code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_block_lines.append(line)
            continue

        # Process inline markdown
        processed = line

        # Inline code: `code`
        processed = re.sub(
            r"`([^`]+)`", lambda m: f"{c.CYAN}{m.group(1)}{c.NC}", processed
        )

        # Bold: **text**
        processed = re.sub(
            r"\*\*([^*]+)\*\*", lambda m: f"{c.BOLD}{m.group(1)}{c.NC}", processed
        )

        # Italic: *text*
        processed = re.sub(
            r"\*([^*]+)\*", lambda m: f"{c.ITALIC}{m.group(1)}{c.NC}", processed
        )

        lines.append(f"{indent}{processed}")

    # Handle unclosed code block
    if in_code_block:
        for code_line in code_block_lines:
            lines.append(f"{indent}  {c.CYAN}{code_line}{c.NC}")

    return "\n".join(lines)


def print_result_summary(result: EvalResult, verbose: bool = False):
    """Print a single result line."""
    c = Colors

    # Status icon
    if result.no_error_detected:
        icon = f"{c.YELLOW}?{c.NC}"
        status = f"{c.YELLOW}no error detected{c.NC}"
    elif result.success:
        icon = f"{c.GREEN}✓{c.NC}"
        parts = []
        if result.has_summary:
            parts.append("summary")
        if result.has_explanation:
            parts.append("explanation")
        if result.has_suggestion:
            parts.append("suggestion")
        status = f"{c.GREEN}{', '.join(parts)}{c.NC}"
    else:
        icon = f"{c.RED}✗{c.NC}"
        status = f"{c.RED}failed{c.NC}"
        if result.error_message:
            status += f" ({result.error_message})"

    # Timing
    timing = ""
    if result.inference_time_ms > 0:
        timing = f" {c.DIM}({result.inference_time_ms}ms){c.NC}"

    print(
        f"  {icon} {c.BOLD}{result.case.id}{c.NC} [{result.case.language}] - {status}{timing}"
    )

    if verbose and result.success:
        output = result.why_output
        if output.get("summary"):
            summary = output["summary"][:80]
            print(f"      {c.CYAN}Summary:{c.NC} {summary}...")


def print_detailed_result(result: EvalResult, show_input: bool = True):
    """Print detailed output for a single result with markdown rendering."""
    c = Colors

    print(f"\n{c.BOLD}{'═' * 70}{c.NC}")
    print(f"{c.BOLD}{result.case.id}{c.NC} [{result.case.language}]")
    print(f"{c.DIM}Error type: {result.case.error_type}{c.NC}")
    print(f"{c.BOLD}{'─' * 70}{c.NC}")

    # Show the raw error input (truncated)
    if show_input:
        print(f"\n{c.BLUE}▸ Input Error:{c.NC}")
        lines = result.case.error_text.strip().split("\n")
        for line in lines[:8]:
            print(f"  {c.DIM}{line[:90]}{c.NC}")
        if len(lines) > 8:
            print(f"  {c.DIM}... ({len(lines) - 8} more lines){c.NC}")

    # Show why output with markdown rendering
    print(f"\n{c.BLUE}▸ Why Output:{c.NC}")

    if result.no_error_detected:
        print(f"  {c.YELLOW}No error detected in input{c.NC}")
    elif result.why_output.get("parse_error"):
        print(f"  {c.RED}Failed to parse JSON output{c.NC}")
        raw = result.why_output.get("raw_output", "")[:200]
        print(f"  {c.DIM}{raw}{c.NC}")
    elif result.success:
        output = result.why_output

        if output.get("summary"):
            print(f"\n  {c.WHITE}{c.BOLD}Summary:{c.NC}")
            print(render_markdown(output["summary"], "    "))

        if output.get("explanation"):
            print(f"\n  {c.BLUE}{c.BOLD}Explanation:{c.NC}")
            print(render_markdown(output["explanation"], "    "))

        if output.get("suggestion"):
            print(f"\n  {c.GREEN}{c.BOLD}Suggestion:{c.NC}")
            print(render_markdown(output["suggestion"], "    "))

        # Stats if available
        if output.get("stats"):
            stats = output["stats"]
            print(f"\n  {c.MAGENTA}Stats:{c.NC}")
            print(f"    Backend: {stats.get('backend', 'unknown')}")
            print(
                f"    Tokens: prompt {stats.get('prompt_tokens', 0)}, gen {stats.get('generated_tokens', 0)}"
            )
            print(f"    Speed: {stats.get('gen_tok_per_s', 0):.1f} tok/s")
    else:
        print(f"  {c.RED}Failed: {result.error_message}{c.NC}")


def find_why_binary() -> Optional[Path]:
    """Find the why binary to use."""
    candidates = [
        Path("./result/bin/why"),
        Path("./why-embedded"),
        Path("./target/release/why"),
        Path("./target/debug/why"),
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    which_result = shutil.which("why")
    if which_result:
        return Path(which_result)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the `why` CLI against error test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Run all cases
  %(prog)s -v                   # Verbose output
  %(prog)s -d                   # Detailed output with markdown
  %(prog)s -f python            # Filter by language
  %(prog)s -f nix -d            # Show Nix errors in detail
  %(prog)s --id rust_borrow -d  # Run specific case
  %(prog)s --limit 10           # Run first 10 cases
        """,
    )
    parser.add_argument(
        "--binary",
        "-b",
        type=Path,
        help="Path to why binary (auto-detected if not specified)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to errors YAML (default: eval/errors.yaml)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        help="Filter cases by language or id (substring match)",
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Run only case with this exact id",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        help="Limit number of cases to run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show more detail in summary output",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show full detailed output with markdown rendering",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--hide-input",
        action="store_true",
        help="Hide input error in detailed view",
    )

    args = parser.parse_args()

    # Handle colors
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    c = Colors

    # Find why binary
    if args.binary:
        why_binary = args.binary.resolve()
        if not why_binary.exists():
            print(
                f"{c.RED}Error:{c.NC} Binary not found: {why_binary}", file=sys.stderr
            )
            sys.exit(1)
    else:
        why_binary = find_why_binary()
        if not why_binary:
            print(f"{c.RED}Error:{c.NC} Could not find why binary.", file=sys.stderr)
            print(f"{c.DIM}Try: nix build && ./scripts/eval.py{c.NC}", file=sys.stderr)
            sys.exit(1)

    # Find data file
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    if args.data:
        data_path = args.data
    else:
        data_path = project_root / "eval" / "errors.yaml"

    if not data_path.exists():
        print(f"{c.RED}Error:{c.NC} Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load cases
    cases = load_errors_from_yaml(data_path)

    # Filter cases
    if args.id:
        cases = [case for case in cases if case.id == args.id]
        if not cases:
            print(f"{c.RED}Error:{c.NC} No case with id: {args.id}", file=sys.stderr)
            sys.exit(1)
    elif args.filter:
        filter_lower = args.filter.lower()
        cases = [
            case
            for case in cases
            if filter_lower in case.id.lower() or filter_lower in case.language.lower()
        ]
        if not cases:
            print(
                f"{c.RED}Error:{c.NC} No cases match filter: {args.filter}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Limit cases
    if args.limit:
        cases = cases[: args.limit]

    # Header
    if not args.json:
        print(f"\n{c.BOLD}why eval{c.NC}")
        print(f"{c.DIM}Binary: {why_binary}{c.NC}")
        print(f"{c.DIM}Cases: {len(cases)}{c.NC}\n")

    # Run evaluations
    results: list[EvalResult] = []

    for case in cases:
        if not args.json and not args.detailed:
            print(f"  {c.DIM}Running {case.id}...{c.NC}", end="\r")

        result = evaluate_case(case, why_binary)
        results.append(result)

        if args.detailed:
            print_detailed_result(result, show_input=not args.hide_input)
        elif not args.json:
            print_result_summary(result, args.verbose)

    # Output
    if args.json:
        output = {
            "binary": str(why_binary),
            "data": str(data_path),
            "total": len(results),
            "results": [],
        }
        for r in results:
            output["results"].append(
                {
                    "id": r.case.id,
                    "language": r.case.language,
                    "error_type": r.case.error_type,
                    "success": r.success,
                    "no_error_detected": r.no_error_detected,
                    "has_summary": r.has_summary,
                    "has_explanation": r.has_explanation,
                    "has_suggestion": r.has_suggestion,
                    "inference_time_ms": r.inference_time_ms,
                    "why_output": r.why_output,
                }
            )

        output["stats"] = {
            "success": sum(1 for r in results if r.success),
            "failed": sum(
                1 for r in results if not r.success and not r.no_error_detected
            ),
            "no_error_detected": sum(1 for r in results if r.no_error_detected),
            "avg_inference_ms": (
                sum(r.inference_time_ms for r in results if r.inference_time_ms > 0)
                // max(1, sum(1 for r in results if r.inference_time_ms > 0))
            ),
        }

        print(json.dumps(output, indent=2))
    else:
        # Print summary stats
        success = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success and not r.no_error_detected)
        no_error = sum(1 for r in results if r.no_error_detected)
        total_time = sum(
            r.inference_time_ms for r in results if r.inference_time_ms > 0
        )
        avg_time = total_time // max(
            1, sum(1 for r in results if r.inference_time_ms > 0)
        )

        print(f"\n{c.BOLD}Summary{c.NC}")
        print(f"  {c.GREEN}Success:{c.NC} {success}/{len(results)}")
        if failed > 0:
            print(f"  {c.RED}Failed:{c.NC} {failed}")
        if no_error > 0:
            print(f"  {c.YELLOW}No error detected:{c.NC} {no_error}")
        print(f"  {c.MAGENTA}Avg inference:{c.NC} {avg_time}ms")
        print()


if __name__ == "__main__":
    main()
