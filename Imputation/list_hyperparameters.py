"""
Print every method's hyperparameter defaults, with the source cited beside each.

    python Imputation/list_hyperparameters.py            # plain text
    python Imputation/list_hyperparameters.py --markdown # tables, for docs

Read straight from the `_DEFAULTS` dict at the top of each impute_*.py, so this
cannot drift from what the code actually runs. The values are the ones reported
in each method's original paper and are meant to stay fixed -- override at
runtime with `--hp KEY=VALUE` rather than editing them.

Parsed with `ast` rather than imported: that keeps the inline comments, which
are where the citations live, and avoids importing torch just to read a dict.
"""
import argparse
import ast
import io
import tokenize
from pathlib import Path

METHODS = [("Mean", "Mean/impute_mean.py"),
           ("Fed-MIWAE", "Miwae/impute_miwae.py"),
           ("Fed-ReMasker", "Remasker/impute_remasker.py"),
           ("Fed-CAFE", "CAFE/impute_cafe.py"),
           ("Fed-HF", "FedHF/impute_fedhf.py")]


def defaults_with_comments(path):
    """[(key, value, comment)] for the file's _DEFAULTS, in source order."""
    src = path.read_text(encoding="utf-8")
    node = next((n for n in ast.parse(src).body
                 if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "_DEFAULTS"), None)
    if node is None:
        return [], ""

    # Comments are not in the AST, so recover them by line from the token stream.
    comments = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comments.setdefault(tok.start[0], []).append(
                tok.string.lstrip("#").strip())
    # A comment on the line above the dict names the source for the whole block.
    header = " ".join(comments.get(node.lineno - 1, []))

    rows = []
    kws = node.value.keywords
    for i, kw in enumerate(kws):
        try:
            value = ast.literal_eval(kw.value)
        except ValueError:
            value = ast.unparse(kw.value)
        # Take comments up to the next key, so two-line justifications are
        # kept whole and none is attributed to the preceding parameter.
        start = kw.value.lineno
        stop = (kws[i + 1].value.lineno - 1 if i + 1 < len(kws)
                else node.end_lineno - 1)
        note = " ".join(c for ln in range(start, stop + 1)
                        for c in comments.get(ln, []))
        rows.append((kw.arg, value, note.strip()))
    return rows, header


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", action="store_true",
                    help="Emit markdown tables instead of plain text")
    args = ap.parse_args()

    root = Path(__file__).parent
    for name, rel in METHODS:
        rows, header = defaults_with_comments(root / rel)
        if not rows:
            continue
        if args.markdown:
            print(f"\n### {name}\n")
            if header:
                print(f"{header}\n")
            print("| Parameter | Default | Source / note |")
            print("|---|---|---|")
            for k, v, note in rows:
                print(f"| `{k}` | `{v}` | {note} |")
        else:
            print(f"\n{name}  ({rel})")
            if header:
                print(f"  {header}")
            w = max(len(k) for k, _, _ in rows)
            for k, v, note in rows:
                print(f"    {k:<{w}}  {str(v):<24}{note}")
    print()


if __name__ == "__main__":
    main()
