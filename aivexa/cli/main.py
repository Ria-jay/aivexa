import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aivexa",
        description="AIVEXA AI safety and security evaluation platform",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="AIVEXA 0.1.0",
    )

    parser.parse_args()


if __name__ == "__main__":
    main()
