from pathlib import Path
from rapgpt.encoder import Encoder


class ArtistLyrics:
    @classmethod
    def from_file(cls, filename: Path) -> str:
        return filename.read_text()


class Artist:
    def __init__(self, artist_file: Path) -> None:
        self.name: str = self.process_artist_name(artist_file.name.strip(".txt"))
        self.lyrics = ArtistLyrics.from_file(artist_file)
        self._name_token: int = -1

    @property
    def name_token(
        self,
    ) -> int:
        return self._name_token

    @name_token.setter
    def name_token(self, value: int) -> None:
        assert self._name_token < 0, "name_token has already been set"
        self._name_token = value

    def process_artist_name(self, name: str) -> str:
        return name.replace(" ", "_").upper()

class Corpus:
    def __init__(self, data_path: str | Path) -> None:
        self.data_path: Path = Path(data_path)

        # TODO: Move these assertions to Config init?
        assert self.data_path.exists()
        assert self.data_path.glob("*.txt")

        self.artists: list[Artist] = [
            Artist(path) for path in self.data_path.glob("*.txt")
        ]

    def __len__(self) -> int:
        return len(self.artists)

    def encode_artists_names(self, encoder: Encoder) -> None:
        for i, artist in enumerate(self.artists):
            name_token = encoder.vocab_size + i
            artist.name_token = name_token
