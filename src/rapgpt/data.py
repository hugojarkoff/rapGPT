from rapgpt.config import Config
from pathlib import Path


class ArtistLyrics:
    @classmethod
    def from_file(cls, filename: Path) -> str:
        return filename.read_text()


class Artist:
    def __init__(self, artist_file: Path) -> None:
        self.name: str = artist_file.name.strip(".txt").capitalize()
        self.lyrics = ArtistLyrics.from_file(artist_file)


class Corpus:
    def __init__(self, data_path: str | Path) -> None:
        self.data_path: Path = Path(data_path)

        # TODO: Move these assertions to Config init?
        assert self.data_path.exists()
        assert self.data_path.glob("*.txt")

        # Dataset is small so fits in memory
        self.artists: list[Artist] = [Artist(f) for f in self.data_path.glob("*.txt")]

    def __len__(self) -> int:
        return len(self.artists)


if __name__ == "__main__":
    # TODO: Put in unit test
    config = Config()
    corpus = Corpus(data_path=config.data.path)
    print("Nb of artists in the corpus: ", len(corpus))
    artist = corpus.artists[0]
    print("First artist name: ", artist.name)
    print("First artist lyrics: ", artist.lyrics[:100])
