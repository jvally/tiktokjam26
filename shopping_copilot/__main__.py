"""Run from the repository root: python3 -m shopping_copilot demo."""

import argparse
import json
from pathlib import Path

from evaluation import evaluate_cases
from retrieval import LexicalRetriever, load_catalog
from shared import SearchState
from .pipeline import ShoppingCopilot

DEMO_CATALOG = Path(__file__).parent / "demo_data" / "catalog.json"
DEMO_CASES = Path(__file__).parent / "demo_data" / "sessions.json"


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam shopping copilot team baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "search", "evaluate", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--catalog", type=Path, default=DEMO_CATALOG,
                             help="Normalized JSON/JSONL catalog; defaults to synthetic demo data")
        command.add_argument("--candidate-limit", type=int, default=200)
        if name == "search":
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--query")
            source.add_argument("--state", type=Path, help="JSON SearchState file")
        elif name == "evaluate":
            command.add_argument("--sessions", type=Path, default=DEMO_CASES)
        elif name == "serve":
            command.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        retriever = LexicalRetriever(load_catalog(str(args.catalog)))
        copilot = ShoppingCopilot(retriever, args.candidate_limit)
        if args.command == "evaluate":
            result = evaluate_cases(retriever, read_json(args.sessions), args.candidate_limit)
            result["dataset"] = str(args.sessions)
            result["synthetic_demo"] = args.sessions.resolve() == DEMO_CASES.resolve()
        elif args.command == "serve":
            from .server import create_server
            server = create_server(copilot, port=args.port)
            print(f"Local development API: http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return
        else:
            if args.command == "demo":
                state = read_json(DEMO_CASES)[0]["state"]
            else:
                state = read_json(args.state) if args.state else SearchState(query=args.query).to_dict()
            result = copilot.search(state)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
