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
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        for i, special_token in enumerate(special_tokens):
            self.word_to_id[special_token] = i

        unique_words = set()
        for text in texts:
            words = text.split()
            unique_words.update(words)

        for i, word in enumerate(sorted(unique_words), start=len(special_tokens)):
            self.word_to_id[word] = i

        for k, v in self.word_to_id.items():
            self.id_to_word[v] = k

        self.vocab_size = len(self.id_to_word)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """

        words = text.split()
        return [self.word_to_id[word] 
          if word in self.word_to_id 
          else self.word_to_id[self.unk_token] 
          for word in words]
            
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = [self.id_to_word[i] for i in ids if self.id_to_word[i] not in {self.pad_token, self.bos_token, self.eos_token}]
        return ' '.join(words)