from sciwing.data.line import Line, Dict
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from typing import List


class DocLines:
    def __init__(
        self, lines: List[str], begin: List[str], end: List[str], overlap: int, tokenizers: Dict[str, BaseTokenizer]
    ):
        self.lines = [Line(text=line, tokenizers=tokenizers) for line in lines]
        assert len(self.lines) >= overlap, f"Length is {len(self.lines)}"
        self.begin = ["<<<begin>>>"] * (overlap-len(begin))
        self.begin.extend(begin)
        assert len(self.begin) == overlap, f"Length is {len(self.begin)}"
        self.begin = [Line(text=line, tokenizers=tokenizers) for line in self.begin]
        self.end = end
        self.end.extend(["<<<end>>>"] * (overlap-len(end)))
        assert len(self.end) == overlap, f"Length is {len(self.end)}"
        self.end = [Line(text=line, tokenizers=tokenizers) for line in self.end]
