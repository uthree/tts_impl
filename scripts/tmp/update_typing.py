#!/usr/bin/env python3
"""Update typing annotations to use built-in types (PEP 585 & PEP 604)."""

import re
from pathlib import Path


def update_typing_imports(content: str) -> str:
    """Update typing imports to remove List, Tuple, Optional, Union."""
    # Find the typing import line
    import_pattern = r"from typing import ([^\n]+)"

    def replace_import(match):
        imports = match.group(1)
        # Split by comma and clean up
        import_list = [item.strip() for item in imports.split(",")]

        # Remove List, Tuple, Optional, Union
        new_imports = [
            item for item in import_list
            if item not in ["List", "Tuple", "Optional", "Union"]
        ]

        # If no imports left, remove the line entirely
        if not new_imports:
            return ""

        # Return updated import
        return f"from typing import {', '.join(new_imports)}"

    content = re.sub(import_pattern, replace_import, content)

    # Remove empty lines that were left behind
    content = re.sub(r"\n\n\n+", "\n\n", content)

    return content


def update_type_annotations(content: str) -> str:
    """Update type annotations to use built-in types."""

    # Replace List[X] with list[X]
    content = re.sub(r"\bList\[", "list[", content)

    # Replace Tuple[X, Y] with tuple[X, Y]
    content = re.sub(r"\bTuple\[", "tuple[", content)

    # Replace Optional[X] with X | None
    # Handle nested cases carefully
    def replace_optional(match):
        inner_type = match.group(1)
        return f"{inner_type} | None"

    # Simple Optional[Type]
    content = re.sub(r"Optional\[([^\[\]]+)\]", replace_optional, content)

    # Nested Optional - repeat a few times to handle nesting
    for _ in range(3):
        content = re.sub(
            r"Optional\[([^\[\]]*\[[^\]]+\][^\[\]]*)\]",
            replace_optional,
            content
        )

    # Replace Union[A, B] with A | B
    def replace_union(match):
        types = match.group(1)
        # Split by comma but be careful with nested brackets
        type_list = []
        depth = 0
        current = ""
        for char in types:
            if char == "[":
                depth += 1
                current += char
            elif char == "]":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                type_list.append(current.strip())
                current = ""
            else:
                current += char
        if current:
            type_list.append(current.strip())

        return " | ".join(type_list)

    # Handle nested Union - repeat to handle nesting
    for _ in range(3):
        content = re.sub(r"Union\[([^\[\]]*(?:\[[^\]]*\])?[^\[\]]*)\]", replace_union, content)

    return content


def process_file(file_path: Path) -> bool:
    """Process a single file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content

        # Update imports
        content = update_typing_imports(content)

        # Update annotations
        content = update_type_annotations(content)

        # Only write if changed
        if content != original:
            file_path.write_text(content, encoding="utf-8")
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

    # Process all Python files in src
    for py_file in src_path.rglob("*.py"):
        if process_file(py_file):
            updated_count += 1

    # Process all Python files in test
    for py_file in test_path.rglob("*.py"):
        if process_file(py_file):
            updated_count += 1

    print(f"\nTotal files updated: {updated_count}")


if __name__ == "__main__":
    main()
