from rapgpt.config import Settings
from pathlib import Path


class Corpus:
    def __init__(self, settings: Settings) -> None:
        self.data_path: Path = Path(settings.data.path)
        # TODO: Move this assertion to Settings init?
        assert self.data_path.exists()
        self.artists: list[Path] = list(self.data_path.glob("*.txt"))

    def __len__(self) -> int:
        return len(self.artists)

class ArtistLyrics:
    @classmethod
    def from_file(cls, filename: Path) -> str:
        return filename.read_text()


class Artist:
    def __init__(self, artist_file: Path):
        self.name: str = artist_file.name.strip(".txt").capitalize()
        self.lyrics = ArtistLyrics.from_file(artist_file)


if __name__ == "__main__":
    # TODO: Put in unit test
    settings = Settings()
    corpus = Corpus(settings)
    print("Nb of artists in the corpus: ", len(corpus))
    artist = Artist(corpus.artists[0])
    print("First artist name: ", artist.name)
    print("First artist lyrics: ", artist.lyrics[:100])
