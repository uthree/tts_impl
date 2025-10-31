#!/usr/bin/env python3
"""Fix Union type annotations."""

import re
from pathlib import Path


def replace_union_in_line(line: str) -> str:
    """Replace Union[A, B, C] with A | B | C."""

    def find_matching_bracket(s: str, start: int) -> int:
        """Find the matching closing bracket."""
        depth = 1
        i = start + 1
        while i < len(s) and depth > 0:
            if s[i] == '[':
                depth += 1
            elif s[i] == ']':
                depth -= 1
            i += 1
        return i - 1 if depth == 0 else -1

    result = line
    while 'Union[' in result:
        match = re.search(r'Union\[', result)
        if not match:
            break

        start = match.start()
        bracket_start = match.end() - 1
        bracket_end = find_matching_bracket(result, bracket_start)

        if bracket_end == -1:
            break

        # Extract the content between brackets
        content = result[bracket_start + 1:bracket_end]

        # Split by comma, but respect nested brackets
        types = []
        current = ""
        depth = 0
        for char in content:
            if char == '[':
                depth += 1
                current += char
            elif char == ']':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                types.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            types.append(current.strip())

        # Join with |
        replacement = " | ".join(types)

        # Replace in result
        result = result[:start] + replacement + result[bracket_end + 1:]

    return result


def process_file(file_path: Path) -> bool:
    """Process a single file."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = [replace_union_in_line(line) for line in lines]
        new_content = "".join(new_lines)

        original_content = "".join(lines)
        if new_content != original_content:
            file_path.write_text(new_content, encoding="utf-8")
            print(f"Updated: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function."""
    src_path = Path(__file__).parent.parent.parent / "src"
    test_path = Path(__file__).parent.parent.parent / "test"

    updated_count = 0

    # Process all Python files
    for py_file in src_path.rglob("*.py"):
        if process_file(py_file):
            updated_count += 1

    for py_file in test_path.rglob("*.py"):
        if process_file(py_file):
            updated_count += 1

    print(f"\nTotal files updated: {updated_count}")


if __name__ == "__main__":
    main()
