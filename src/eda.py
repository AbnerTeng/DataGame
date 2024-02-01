"""
Get EDA dashboard using sweetviz

sweetviz official document: https://pypi.org/project/sweetviz/
"""
import argparse
import sweetviz
from .utils import load_data


class EDA:
    """
    EDA session
    """
    def __init__(self, path: str) -> None:
        self.data = load_data(path)


    def get_report(self) -> None:
        """
        Get EDA report
        """
        report = sweetviz.analyze(self.data)
        report.show_html()


def arguments() -> argparse.ArgumentParser:
    """
    Get arguments
    """
    parser = argparse.ArgumentParser(description="Get EDA report")
    parser.add_argument("--path", type=str, help="path to data")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    eda = EDA(args.path)
    print(eda.data.head())
    #eda.get_report()
