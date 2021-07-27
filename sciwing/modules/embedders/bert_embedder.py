import torch
import torch.nn as nn
from sciwing.modules.embedders.base_embedders import BaseEmbedder
from sciwing.data.datasets_manager import DatasetsManager
from typing import List, Union
import wasabi
import sciwing.constants as constants
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from transformers import AutoTokenizer
from transformers import AutoModel

PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class BertEmbedder(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self,
        datasets_manager: DatasetsManager = None,
        dropout_value: float = 0.1,
        aggregation_type: str = "sum",
        layers: Union[str, List[int]] = "all",
        transformer: str = "bert-base-uncased",
        word_tokens_namespace="tokens",
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """ Bert Embedder that embeds the given instance to BERT embeddings

        Parameters
        ----------
        dropout_value : float
            The amount of dropout to be added after the embedding
        aggregation_type : str
            The kind of aggregation of different layers. BERT produces representations from
            different layers. This specifies the strategy to aggregating them
            One of

            sum
                Sum the representations from all the layers
            average
                Average the representations from all the layers

        transformer : type
            The kind of BERT embedding to be used

            bert-base-uncased
                12 layer transformer trained on lowercased vocab

            bert-large-uncased:
                24 layer transformer trained on lowercased vocab

            bert-base-cased:
                12 layer transformer trained on cased vocab

            bert-large-cased:
                24 layer transformer train on cased vocab

            scibert-base-cased
                12 layer transformer trained on scientific document on cased normal vocab
            scibert-sci-cased
                12 layer transformer trained on scientific documents on cased scientifc vocab

            scibert-base-uncased
                12 layer transformer trained on scientific docments on uncased normal vocab

            scibert-sci-uncased
                12 layer transformer train on scientific documents on ncased scientific vocab

        word_tokens_namespace : str
            The namespace in the liens where the tokens are stored

        device :  Union[torch.device, str]
            The device on which the model is run.
        """
        super(BertEmbedder, self).__init__()

        self.datasets_manager = datasets_manager
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        if isinstance(layers, str):
            if layers == "all":
                self.layers = list(range(1, 13))
            elif layers == "last_four":
                self.layers = [9, 10, 11, 12]
            elif layers == "last":
                self.layers = [12]
            elif layers == "second_to_last":
                self.layers = [11]
            else:
                raise ValueError(f"The layer specification {layers} is not supported")
        else:
            self.layers = layers
        self.bert_type = transformer
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.word_tokens_namespace = word_tokens_namespace
        self.msg_printer = wasabi.Printer()
        self.embedder_name = self.bert_type

        self.scibert_foldername_mapping = {
            "scibert-base-cased": "scibert_basevocab_cased",
            "scibert-sci-cased": "scibert_scivocab_cased",
            "scibert-base-uncased": "scibert_basevocab_uncased",
            "scibert-sci-uncased": "scibert_scivocab_uncased",
        }

        if self.bert_type in self.scibert_foldername_mapping.keys():
            foldername = self.scibert_foldername_mapping[self.bert_type]
            self.model_type_or_folder_url = "allenai/" + foldername

        else:
            self.model_type_or_folder_url = self.bert_type

        # load the bert model
        with self.msg_printer.loading(" Loading Bert tokenizer and model. "):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type_or_folder_url,
                                                           use_fast=False,
                                                           do_basic_tokenize=False)
            self.model = AutoModel.from_pretrained(self.model_type_or_folder_url)
            self.model.to(self.device)

        self.msg_printer.good(f"Finished Loading {self.bert_type} model and tokenizer")
        self.dimension = self.get_dimension()

    def forward(self, lines: List[Line]) -> torch.Tensor:
        """

        Parameters
        ----------
        lines : List[Line]
            A list of lines

        Returns
        -------
        torch.Tensor
            The bert embeddings for all the words in the instances
            The size of the returned embedding is ``[batch_size, max_len_word_tokens, emb_dim]``

        """

        # word_tokenize all the text string in the batch
        transformer_tokens_lengths = []
        word_tokens_lengths = []
        for line in lines:
            text = line.text
            word_tokens = line.tokens[self.word_tokens_namespace]
            word_tokens_lengths.append(len(word_tokens))

            # split every token to subtokens
            for word_token in word_tokens:
                word_piece_tokens = self.tokenizer.tokenize(word_token.text)
                word_token.sub_tokens = word_piece_tokens

            tokenized_text = self.tokenizer.tokenize(text)
            line.tokenizers[self.embedder_name] = self.tokenizer
            line.add_tokens(tokens=tokenized_text, namespace=self.embedder_name)
            transformer_tokens_lengths.append(len(tokenized_text))

        max_len_words = max(word_tokens_lengths)
        tokenized_input = self.tokenizer([line.text for line in lines],
                                         padding=True, truncation=True,
                                         return_tensors="pt")

        output = self.model(**tokenized_input, output_hidden_states=True)

        encoded_layers = output.hidden_states

        if "base" in self.transformer:
            assert len(encoded_layers) == 13, f"{len(encoded_layers)}"
        elif "large" in self.transformer:
            assert len(encoded_layers) == 25, f"{len(encoded_layers)}"

        # batch_size, num_bert_layers, max_len_bert + 2, bert_hidden_dimension
        filtered_layers = [encoded_layers[layer] for layer in self.layers]
        filtered_layers = torch.stack(filtered_layers, dim=1)

        # batch_size, max_len_bert + 2, bert_hidden_dimension
        if self.aggregation_type == "sum":
            encoding = torch.sum(filtered_layers, dim=0)

        elif self.aggregation_type == "average":
            encoding = torch.mean(filtered_layers, dim=0)
        else:
            raise ValueError(f"The aggregation type {self.aggregation_type}")

        # fill up the appropriate embeddings in the tokens of the lines
        batch_embeddings = []
        for idx, line in enumerate(lines):
            word_tokens = line.tokens[self.word_tokens_namespace]  # word tokens
            transformer_tokens_ = line.tokens[self.embedder_name]
            token_embeddings = encoding[idx]  # max_len_bert + 2, bert_hidden_dimension

            len_word_tokens = len(word_tokens)
            len_transformer_tokens = len(transformer_tokens_)
            padding_length_words = max_len_words - len_word_tokens

            attention_mask = tokenized_input.attention_mask[idx]

            token_embeddings = [token_embeddings[i] for i in range(len(attention_mask)) if attention_mask[i] == 1]

            # do not want embeddings for start and end tokens
            token_embeddings = token_embeddings[1:-1]

            # just have embeddings for the bert tokens now
            # without padding and start and end tokens
            assert len(token_embeddings) == len_transformer_tokens, (
                f"bert token embeddings size {len(token_embeddings)} and length of bert tokens "
                f"{len_transformer_tokens}"
            )

            line_embeddings = []
            for token in word_tokens:
                idx = 0
                sub_tokens = token.sub_tokens
                len_sub_tokens = len(sub_tokens)

                # taking the embedding of only the first token
                # TODO: Have different strategies for this
                emb = token_embeddings[idx]
                line_embeddings.append(emb)
                token.set_embedding(name=self.embedder_name, value=emb)
                idx += len_sub_tokens

            for i in range(padding_length_words):
                zeros = torch.zeros(self.embedding_dimension)
                zeros = zeros.to(self.device)
                line_embeddings.append(zeros)

            line_embeddings = torch.stack(line_embeddings)
            batch_embeddings.append(line_embeddings)

        # batch_size, max_len_words, bert_hidden_dimension
        batch_embeddings = torch.stack(batch_embeddings)
        return batch_embeddings

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size
