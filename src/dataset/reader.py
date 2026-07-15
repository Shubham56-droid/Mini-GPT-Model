from pathlib import Path

class TextReader:

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def read(self) -> str:
        """
        Reads the entire text file and returns it as a string.
        """

        with open(self.filepath, "r", encoding="utf-8") as file:
            return file.read()