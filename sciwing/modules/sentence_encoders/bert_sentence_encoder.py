from typing import List, Union
import torch
import torch.nn as nn
import wasabi
from transformers import AutoTokenizer, AutoModel

import sciwing.constants as constants
from sciwing.data.line import Line
from sciwing.modules.attentions.multihead_attention import MultiHeadAttention
from sciwing.utils.class_nursery import ClassNursery

PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class BertSentenceEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        dropout_value: float = 0.1,
        aggregation_type: str = "sum",
        embedding_method: str = "attn_pool",
        transformer: str = "bert-base-cased",
        device: Union[torch.device, str] = torch.device("cpu"),
        sentencepiece: bool = False
    ):
        super(BertSentenceEncoder, self).__init__()

        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.embedding_method = embedding_method
        self.transformer = transformer
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.msg_printer = wasabi.Printer()
        self.embedder_name = self.transformer

        self.scibert_foldername_mapping = {
            "scibert-base-cased": "scibert_basevocab_cased",
            "scibert-sci-cased": "scibert_scivocab_cased",
            "scibert-base-uncased": "scibert_basevocab_uncased",
            "scibert-sci-uncased": "scibert_scivocab_uncased",
        }

        if self.transformer in self.scibert_foldername_mapping.keys():
            foldername = self.scibert_foldername_mapping[self.transformer]
            self.model_type_or_folder_url = "allenai/" + foldername

        else:
            self.model_type_or_folder_url = self.transformer

        # load the bert model
        with self.msg_printer.loading(" Loading transformer tokenizer and model. "):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type_or_folder_url,
                                                           use_fast=False,
                                                           do_basic_tokenize=False)
            self.model = AutoModel.from_pretrained(self.model_type_or_folder_url)

        self.attention = MultiHeadAttention(encode_dim=self.model.config.hidden_size)

        self.msg_printer.good(f"Finished Loading {self.transformer} model and tokenizer")
        self.dimension = self.get_dimension()
        self.sentencepiece = sentencepiece

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
            The size of the returned embedding is ``[batch_size, emb_dim]``

        """

        # word_tokenize all the text string in the batch
        transformer_tokens_lengths = []
        for line in lines:
            text = self.strip_special_tokens(line.text)
            tokenized_text = self.tokenizer.tokenize(text)
            if self.sentencepiece and len(tokenized_text) == 0:
                line.text = "一"
                tokenized_text = self.tokenizer.tokenize(line.text)
            else:
                assert len(tokenized_text) != 0, f"{line.text}"
            line.tokenizers[self.embedder_name] = self.tokenizer
            line.add_tokens(tokens=tokenized_text, namespace=self.embedder_name)
            transformer_tokens_lengths.append(len(tokenized_text))

        tokenized_input = self.tokenizer([self.strip_special_tokens(line.text) for line in lines],
                                         padding=True, truncation=True,
                                         return_tensors="pt").to(self.device)

        model_output = self.model(**tokenized_input)[0]

        # batch_size, bert_hidden_dimension
        if self.embedding_method == "CLS":
            cls_mask = tokenized_input.input_ids.eq(self.tokenizer.cls_token_id)
            sentence_encoding = model_output[cls_mask, :].view(
                model_output.size(0), -1, model_output.size(-1))[:, -1, :]
        elif self.embedding_method == "EOS":
            eos_mask = tokenized_input.input_ids.eq(self.tokenizer.eos_token_id)
            sentence_encoding = model_output[eos_mask, :].view(
                model_output.size(0), -1, model_output.size(-1))[:, -1, :]
        elif self.embedding_method == "mean_pool":
            attention_mask = tokenized_input.attention_mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
            sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_encoding = sum_embeddings / sum_mask
        elif self.embedding_method == "max_pool":
            attention_mask = tokenized_input.attention_mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
            embeddings = model_output
            embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            sentence_encoding = torch.max(embeddings, dim=1)
        elif self.embedding_method == "attn_pool_cls":
            attention_mask = tokenized_input.attention_mask
            cls_mask = tokenized_input.input_ids.eq(self.tokenizer.cls_token_id)
            query = model_output[cls_mask, :].view(
                model_output.size(0), -1, model_output.size(-1))[:, -1, :]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
            key = model_output * input_mask_expanded
            attn = self.attention(query_matrix=query, key_matrix=key)
            attn_unsqueeze = attn.unsqueeze(-2)
            attn_encoding = torch.matmul(attn_unsqueeze, key)
            sentence_encoding = attn_encoding.squeeze(-2)
        elif self.embedding_method == "attn_pool_eos":
            attention_mask = tokenized_input.attention_mask
            eos_mask = tokenized_input.input_ids.eq(self.tokenizer.eos_token_id)
            query = model_output[eos_mask, :].view(
                model_output.size(0), -1, model_output.size(-1))[:, -1, :]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
            key = model_output * input_mask_expanded
            attn = self.attention(query_matrix=query, key_matrix=key)
            attn_unsqueeze = attn.unsqueeze(-2)
            attn_encoding = torch.matmul(attn_unsqueeze, key)
            sentence_encoding = attn_encoding.squeeze(-2)
        else:
            raise ValueError(f"The embedding type {self.embedding_method} does not exist")
        return sentence_encoding

    def get_dimension(self) -> int:
        return self.model.config.hidden_size

    def strip_special_tokens(self, line: str) -> str:
        dt = {self.tokenizer.bos_token: "bos",
              self.tokenizer.eos_token: "eos",
              self.tokenizer.cls_token: "cls",
              self.tokenizer.sep_token: "sep",
              self.tokenizer.mask_token: "mask",
              self.tokenizer.unk_token: "unk",
              self.tokenizer.pad_token: "pad"}
        for key, value in dt.items():
            line = line.replace(key, value)
        return line