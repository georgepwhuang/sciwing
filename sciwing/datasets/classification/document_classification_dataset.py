from typing import Dict, List, Any
from sciwing.data.doc_lines import DocLines
from sciwing.data.doc_labels import DocLabels
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from torch.utils.data import Dataset
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
from sciwing.numericalizers.numericalizer import Numericalizer
from sciwing.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.utils.class_nursery import ClassNursery
from sciwing.vocab.vocab import Vocab
from collections import defaultdict
import random


class DocumentClassificationDataset(BaseTextClassification, Dataset):
    """ This represents a dataset that is of the form

    line1###label1
    line2###label2
    line3###label3

    line4###label4
    line5###label5
    .
    .
    .

    where lines 1,2,3 have contextual relation and lines 4,5 have contextual relation
    """

    def __init__(
        self, filename: str, tokenizers: Dict[str, BaseTokenizer] = WordTokenizer(),
            batch_size: int = 64, window_size: int = 3, dilution_gap: int = 0
    ):
        super().__init__(filename, tokenizers)
        self.filename = filename
        self.tokenizers = tokenizers
        self.batch_size = batch_size
        self.overlap = int((window_size - 1) / 2.0) * (dilution_gap + 1)
        self.lines, self.labels = self.get_lines_labels()

    def get_lines_labels(self) -> (List[DocLines], List[DocLabels]):
        lines: List[str] = []
        labels: List[str] = []
        doc_lines: List[DocLines] = []
        doc_labels: List[DocLabels] = []
        with open(self.filename) as fp:
            for line in fp:
                if bool(line.strip()):
                    line, label = line.split("###")
                    line = line.strip()
                    label = label.strip()
                    lines.append(line)
                    labels.append(label)
                else:
                    lines_read = len(lines)
                    counter = 0
                    while True:
                        offset = random.randint(1, self.batch_size)
                        if offset >= self.overlap and \
                                (lines_read - offset) % self.batch_size >= self.overlap:
                            break
                        counter += 1
                        assert counter < 10000, "Unable to partition batch"
                    doc_lines_batch = self.create_line_batch(lines, offset)
                    doc_labels_batch = self.create_label_batch(labels, offset)
                    doc_lines.extend(doc_lines_batch)
                    doc_labels.extend(doc_labels_batch)
                    lines_batched = sum([len(doc.lines) for doc in doc_lines_batch])
                    assert lines_read == lines_batched, \
                        f"Lines read ({lines_read}) and lines batched ({lines_batched})do not match"
                    lines = []
                    labels = []
        if len(lines) > 0 and len(labels) > 0:
            offset = random.randint(1, self.batch_size)
            doc_lines_batch = self.create_line_batch(lines, offset)
            doc_labels_batch = self.create_label_batch(labels, offset)
            doc_lines.extend(doc_lines_batch)
            doc_labels.extend(doc_labels_batch)
            lines_read = len(lines)
            lines_batched = sum([len(doc.lines) for doc in doc_lines_batch])
            assert lines_read == lines_batched, \
                f"Lines read ({lines_read}) and lines batched ({lines_batched})do not match"

        return doc_lines, doc_labels

    def create_line_batch(self, ls: List[str], offset: int) -> List[DocLines]:
        doc_lines = []
        lines = ls[0:offset]
        begin = []
        end = ls[offset:offset+self.overlap]
        doc_lines.append(DocLines(lines=lines, begin=begin, end=end, overlap=self.overlap, tokenizers=self.tokenizers))
        for idx in range(offset, len(ls), self.batch_size):
            lines = ls[idx: idx + self.batch_size]
            begin = ls[idx - self.overlap:idx]
            if idx > len(ls) - self.batch_size - self.overlap:
                end = ls[idx + self.batch_size: len(ls)]
            else:
                end = ls[idx + self.batch_size: idx + self.batch_size + self.overlap]
            doc_lines.append(DocLines(lines=lines, begin=begin, end=end, overlap=self.overlap, tokenizers=self.tokenizers))
        return doc_lines

    def create_label_batch(self, ls: List[str], offset: int) -> List[DocLabels]:
        doc_labels = []
        labels = ls[0:offset]
        doc_labels.append(DocLabels(labels=labels))
        for idx in range(offset, len(ls), self.batch_size):
            labels = ls[idx: idx + self.batch_size]
            doc_labels.append(DocLabels(labels=labels))
        return doc_labels

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> (DocLines, DocLabels):
        doc_line, doc_label = self.lines[idx], self.labels[idx]
        return doc_line, doc_label

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value


class DocumentClassificationDatasetManager(DatasetsManager, ClassNursery):
    def __init__(
        self,
        train_filename: str,
        dev_filename: str,
        test_filename: str,
        tokenizers: Dict[str, BaseTokenizer] = None,
        namespace_vocab_options: Dict[str, Dict[str, Any]] = None,
        namespace_numericalizer_map: Dict[str, BaseNumericalizer] = None,
        batch_size: int = 64,
        window_size: int = 3,
        dilution_gap: int = 0
    ):
        """

        Parameters
        ----------
        train_filename: str
            The path where the train file is stored
        dev_filename: str
            The path where the dev file is stored
        test_filename: str
            The path where the test file is stored
        tokenizers: Dict[str, BaseTokenizer]
            A mapping from namespace to the tokenizer
        namespace_vocab_options: Dict[str, Dict[str, Any]]
            A mapping from the name to options
        namespace_numericalizer_map: Dict[str, BaseNumericalizer]
            Every namespace can have a different numericalizer specified
        batch_size: int
            The batch size of the data returned
        """
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.tokenizers = tokenizers or {
            "tokens": WordTokenizer(),
            "char_tokens": CharacterTokenizer(),
        }
        self.namespace_vocab_options = namespace_vocab_options or {
            "char_tokens": {
                "start_token": " ",
                "end_token": " ",
                "pad_token": " ",
                "unk_token": " ",
            },
            "label": {"include_special_vocab": False},
        }
        self.namespace_numericalizer_map = namespace_numericalizer_map or {
            "tokens": Numericalizer(),
            "char_tokens": Numericalizer(),
        }
        self.namespace_numericalizer_map["label"] = Numericalizer()
        self.batch_size = batch_size

        self.train_dataset = DocumentClassificationDataset(
            filename=self.train_filename, tokenizers=self.tokenizers,
            batch_size=batch_size, window_size=window_size, dilution_gap=dilution_gap
        )
        self.dev_dataset = DocumentClassificationDataset(
            filename=self.dev_filename, tokenizers=self.tokenizers,
            batch_size=batch_size, window_size=window_size, dilution_gap=dilution_gap
        )
        self.test_dataset = DocumentClassificationDataset(
            filename=self.test_filename, tokenizers=self.tokenizers,
            batch_size=batch_size, window_size=window_size, dilution_gap=dilution_gap
        )

        super(DocumentClassificationDatasetManager, self).__init__(
            train_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            namespace_vocab_options=self.namespace_vocab_options,
            namespace_numericalizer_map=self.namespace_numericalizer_map,
            batch_size=1,
        )

    def build_vocab(self) -> Dict[str, Vocab]:
        """ Returns a vocab for each of the namespace
        The namespace identifies the kind of tokens
        Some tokens correspond to words
        Some tokens may correspond to characters.
        Some tokens may correspond to Bert style tokens

        Returns
        -------
        Dict[str, Vocab]
            A vocab corresponding to each of the

        """
        lines = []
        for doc in self.train_dataset.lines:
            lines.extend(doc.lines)
            lines.extend(doc.begin)
            lines.extend(doc.end)
        labels = []
        for doc in self.train_dataset.labels:
            labels.extend(doc.labels)

        namespace_to_instances: Dict[str, List[List[str]]] = defaultdict(list)
        for line in lines:
            namespace_tokens = line.tokens
            for namespace, tokens in namespace_tokens.items():
                namespace_to_instances[namespace].append(tokens)
        for label in labels:
            namespace_tokens = label.tokens
            for namespace, tokens in namespace_tokens.items():
                namespace_to_instances[namespace].append(tokens)

        self.label_namespaces = list(labels[0].tokens.keys())

        namespace_to_vocab: Dict[str, Vocab] = {}

        # This always builds a vocab from instances
        for namespace, instances in namespace_to_instances.items():
            namespace_to_vocab[namespace] = Vocab(
                instances=instances, **self.namespace_vocab_options.get(namespace, {})
            )
            namespace_to_vocab[namespace].build_vocab()
        return namespace_to_vocab
