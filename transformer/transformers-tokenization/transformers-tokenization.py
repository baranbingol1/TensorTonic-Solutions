import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        unique_words = set()
        
        for text in texts:
            for word in text.lower().split():
                unique_words.add(word)
        
        for word in sorted(unique_words):
            if word not in self.word_to_id.keys():
                self.word_to_id[word] = len(self.word_to_id)

        self.vocab_size = len(self.word_to_id)
        self.id_to_word = {value: key for key, value in self.word_to_id.items()}
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        ids = list()
        for word in text.lower().split():
            if word in self.word_to_id.keys():
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id[self.unk_token])
        return ids
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        tokens = list()
        for id in ids:
            tokens.append(self.id_to_word.get(id, self.unk_token))
        return " ".join(tokens)
