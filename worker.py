import argparse
import asyncio
import json

from app.bootstrap import create_application_context


async def _preload() -> None:
    context = create_application_context()
    await context.preload_all()
    print(json.dumps(context.snapshot(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preload Demucs and WhisperX models.")
    parser.add_argument("command", choices=["preload"], help="Action to execute.")
    args = parser.parse_args()

    if args.command == "preload":
        asyncio.run(_preload())


if __name__ == "__main__":
    main()
