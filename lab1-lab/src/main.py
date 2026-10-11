import os
import argparse
from dotenv import load_dotenv
from src.settings import Settings
from pathlib import Path
import yaml


def export_envs(environment: str = "dev") -> None:
    dotenv_path = Path("config") / f".env.{environment}"

    load_dotenv(dotenv_path=dotenv_path)


def export_secrets(yaml_path: str = "secrets.yaml") -> None:
    with open(yaml_path, "r") as file:
        secrets = yaml.safe_load(file)

    for key, value in secrets.items():
        os.environ[key] = str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load environment variables from specified.env file."
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        help="The environment to load (dev, test, prod)",
    )
    args = parser.parse_args()

    export_envs(args.environment)
    export_secrets()

    settings = Settings()  # type: ignore

    print("APP_NAME: ", settings.APP_NAME)
    print("ENVIRONMENT: ", settings.ENVIRONMENT)
    print("OPENAI_API_KEY: ", settings.OPENAI_API_KEY)
