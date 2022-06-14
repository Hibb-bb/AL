# -*- coding: utf-8 -*-

# TODO: SentencePiece

from typing import List, Optional, Tuple, Union

from tokenizers import (
    Encoding,
    Tokenizer,
    AddedToken,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
)

from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from transformers.tokenization_utils import PreTrainedTokenizer

# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
# TODO: 1. these tokenizers should inherit from PreTrainedTokenizer.
#       2. check post-processing.
#       3. add encode_batch, decode_batch.


class ByteLevelBPETokenizer(object):

    def __init__(
        self,
        vocab_size: int = 25000,
        min_freq: int = 5,
        lang: str = "en",
        is_tgt: bool = True,
        files: Optional[List[str]] = [None, None]
    ) -> None:
        """

        Args:
            vocab_size: (int)
            min_freq: minimum frequency
            lang: "en", "fr", etc.
            files: (List[str]) ["vocab.json", "merge.txt"]
        """
        super(ByteLevelBPETokenizer, self).__init__()

        self.tokenizer = Tokenizer(BPE(files[0], files[1]))

        self.lang = lang
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=["<pad>", "<s>", "</s>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        # https://huggingface.co/docs/tokenizers/python/latest/components.html#normalizers
        self.tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        # https://huggingface.co/docs/tokenizers/python/latest/components.html#pre-tokenizers
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.ByteLevel()
        ])
        # https://huggingface.co/docs/tokenizers/python/latest/components.html#postprocessor
        if is_tgt:
            self.tokenizer.post_processor = processors.TemplateProcessing(
                single="<s> $A </s>",
                pair="<s> $A </s> $B:1",
                special_tokens=[("<s>", 1), ("</s>", 2)],
            )
        # https://huggingface.co/docs/tokenizers/python/latest/components.html#decoders
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, files=None) -> None:

        if files is None:
            # files: ["test.txt", "train.txt", "valid.txt"]
            files = [
                f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]
            ]

        self.tokenizer.train(files, self.trainer)

    def save(self) -> None:

        # folder
        self.tokenizer.model.save(f"data/tokenizer/{self.lang}")

    def encode(self, input: Union[str, List[str], Tuple[str]]) -> Encoding:

        return self.tokenizer.encode(input)

    def decode(self, input: Encoding) -> str:

        # Note that type(input) == Encoding
        return self.tokenizer.decode(input.ids)


class SentencePiece(PreTrainedTokenizer):
    """https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/tokenization_t5.py"""

    def __init__(self) -> None:
        super().__init__()


def train_bpe():

    tokenizer = ByteLevelBPETokenizer(lang="fr")
    files = [
        "data/wmt14/commoncrawl/commoncrawl.fr-en.fr.shell",
        "data/wmt14/europarl_v7/europarl-v7.fr-en.fr.shell",
        "data/wmt14/giga/giga-fren.release2.fixed.fr.shell",
        "data/wmt14/news-commentary/news-commentary-v9.fr-en.fr.shell",
        "data/wmt14/un/undoc.2000.fr-en.fr.shell"
    ]
    tokenizer.train(files)
    tokenizer.save()
    encoded = tokenizer.encode(
        "Bonjour, vous tous! Comment ça va 😁?")
    # Outputs:
    # ['Ġbon', 'jour', ',', 'Ġvous', 'Ġtous', '!', 'Ġcomment', 'ĠÃ§a', 'Ġva',
    #  'Ġ', 'ð', 'Ł', 'ĺ', 'ģ', '?']
    print(encoded.tokens)
    decoded = tokenizer.decode(encoded)
    # Outputs:
    # bonjour, vous tous! comment ça va 😁?
    print(decoded)


if __name__ == "__main__":
    pass
