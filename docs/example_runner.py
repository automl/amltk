"""Generate the code reference pages and navigation https://mkdocstrings.github.io/recipes/."""
from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path

import mkdocs_gen_files
from markdown_exec.formatters.python import _sessions
from more_itertools import first, first_true, peekable

logger = logging.getLogger(__name__)

nav = mkdocs_gen_files.Nav()  # pyright: reportPrivateImportUsage=false

ENV_VAR = "AMLTK_DOC_RENDER_EXAMPLES"


@dataclass
class CodeSegment:
    lines: list[str]
    session: str
    exec: bool

    def code(self, code: list[str]) -> str:
        points_start = first_true(code, pred=lambda _l: _l.startswith("# 1."))
        if points_start is not None:
            points_start_index = code.index(points_start)

            points = [""]
            points.extend([_l.lstrip("#")[1:] for _l in code[points_start_index:]])
            points.append("")

            body = code[:points_start_index]
        else:
            points = []
            body = code

        # Trim off any excess leading lines which only have whitespace
        while body and body[0].strip() == "":
            body.pop(0)

        hl_lines = []
        for i, line in enumerate(body):
            _l = line.rstrip()
            if "<!>" in _l:
                hl_lines.append(str(i + 1))
                _l = _l.replace("<!>", "").rstrip()

            for strip in ["# !", "#!", "#"]:
                if _l.endswith(strip):
                    _l = _l[: -len(strip)]

            body[i] = _l

        if any(hl_lines):
            hl_lines = " ".join(hl_lines)
            hl_string = f'hl_lines="{hl_lines}"'
        else:
            hl_string = ""

        # We generate two tabs if executing
        if self.exec:
            indented_body = "\n".join(f"    {_l}" for _l in body)

            code_annotations = " ".join(
                [
                    "{",
                    ".python",
                    ".annotate",
                    hl_string,
                    "}",
                ],
            )
            tab1 = "\n".join(
                [
                    '=== "Code"',
                    "",
                    f"    ``` {code_annotations}",
                    indented_body,
                    "    ```",
                    *[f"    {point}" for point in points],
                    "",
                ],
            )

            # HACK(eddiebergman): to make `__name__` work
            session_object = _sessions.setdefault(self.session, {})
            if "__name__" not in session_object:
                session_object["__name__"] = "__main__"


            run_annotations = " ".join(
                [
                    "{",
                    ".python",
                    f"session='{self.session}'",
                    'exec="True"',
                    'result="python"',
                    "}",
                ],
            )

            tab2 = "\n".join(
                [
                    '=== "Run"',
                    "",
                    f"    ``` {run_annotations}",
                    indented_body,
                    "    ```",
                ],
            )

            return "\n".join([tab1, "", tab2])

        annotations = " ".join(["{", ".python", ".annotate", hl_string, "}"])
        top = f"```{annotations}"
        bottom = "```"

        s = [top, *body, bottom, *points]
        body = "\n".join(s)
        return body

    def __str__(self) -> str:
        return self.code(self.lines)


@dataclass
class CommentSegment:
    lines: list[str]

    def __str__(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Example:
    name: str
    filepath: Path
    description: str
    segments: list[CodeSegment | CommentSegment]

    @classmethod
    def should_execute(cls, name: str) -> bool:
        env_var = os.environ.get(ENV_VAR, None)
        if env_var is None:
            return False
        if env_var == "all":
            return True

        examples_to_exec = [
            example.lstrip().rstrip() for example in env_var.lower().split(",")
        ]
        return name.lower() in examples_to_exec

    @classmethod
    def from_file(cls, path: Path) -> Example:
        with path.open() as f:
            lines = f.readlines()

        lines = iter(lines)

        # First line is the name of the example to show
        name = first(lines).strip().replace('"""', "")

        # Lines leading up to the second triple quote are the description
        description = "".join(takewhile(lambda _l: not _l.startswith('"""'), lines))

        segments: list[CodeSegment | CommentSegment] = []

        # The rest is interspersed with triple quotes and code blocks
        # We need to wrap the code blocks in triple backticks while
        # removing the triple quotes for the comment blocks
        remaining = peekable(lines)
        while remaining.peek(None) is not None:
            # If we encounter triple backticks we remove them and just add the lines
            # in, up until the point we hit the next set of backticks
            if remaining.peek().startswith('"""'):
                # Skip the triple quotes
                next(remaining)
                ls = list(takewhile(lambda _l: not _l.startswith('"""'), remaining))
                comment_segment = CommentSegment([line.rstrip() for line in ls])
                segments.append(comment_segment)

            # Otherwise we wrap the line in triple backticks until we hit the next
            # set of triple quotes
            else:
                ls = list(takewhile(lambda _l: not _l.startswith('"""'), remaining))
                code_segment = CodeSegment(
                    [line.rstrip() for line in ls],
                    session=name,
                    exec=cls.should_execute(name),
                )
                segments.append(code_segment)

                remaining.prepend('"""')  # Stick back in so we can find it next itr

        return cls(name, path, description, segments)

    def header(self) -> str:
        return f"# {self.name}"

    def description_header(self) -> str:
        return "\n".join(
            [
                "## Description",
                self.description,
            ],
        )

    def generate_doc(self) -> str:
        return "\n".join(
            [
                self.header(),
                self.copy_section(),
                self.description_header(),
                *map(str, self.segments),
            ],
        )

    def copy_section(self) -> str:
        body = "\n".join(
            [
                "```python",
                *[
                    "\n".join(segment.lines)
                    for segment in self.segments
                    if isinstance(segment, CodeSegment)
                ],
                "```",
            ],
        )
        indented_body = textwrap.indent(body, " " * 4)
        header = (
            f'??? quote "Expand to copy'
            f' `{self.filepath}` :material-content-copy: (top right)"'
        )
        return "\n".join(
            [
                header,
                "",
                indented_body,
                "",
            ],
        )


for path in sorted(Path("examples").rglob("*.py")):
    module_path = path.relative_to("examples").with_suffix("")
    doc_path = path.relative_to("examples").with_suffix(".md")
    full_doc_path = Path("examples", doc_path)

    parts = tuple(module_path.parts)
    filename = parts[-1]

    if filename.startswith("_"):
        continue

    example = Example.from_file(path)
    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        f.write(example.generate_doc())

    toc_name = example.name
    nav[toc_name] = doc_path.as_posix()
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("examples/SUMMARY.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())  #
