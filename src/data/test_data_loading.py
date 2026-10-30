import logging

from config.config import create_project_directories

from src.data.load_data import load_and_validate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main():

    create_project_directories()

    df, results = load_and_validate()

    print("\nDATA LOADING SUCCESSFUL")
    print("=" * 50)

    print("\nDataset Preview:")
    print(df.head())

    print("\nValidation Results:")
    for key, value in results.items():

        print(f"{key}: {value}")


if __name__ == "__main__":
    main()