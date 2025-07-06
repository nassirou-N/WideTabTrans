import re
import warnings
import numpy as np
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

# Opérateurs Solidity
OPERATORS = {
    # 3 caractères
    '<<=', '>>=',
    # 2 caractères
    '->', '++', '--', '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', 
    '%=', '&=', '^=', '|=',
    # 1 caractère
    '(', ')', '[', ']', '.', '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|', '=', ',', '?', ':', ';',
    '{', '}'
}

class FragmentVectorizer:
    def __init__(self, vector_length):
        self.fragments = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0

    @staticmethod
    def tokenize(line):
        """Tokenise une ligne de code Solidity."""
        tokens = []
        current_word = []
        i = 0
        
        while i < len(line):
            char = line[i]
            
            # Ignorer les espaces
            if char == ' ':
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                i += 1
                continue
            
            # Vérifier les opérateurs (3, 2, puis 1 caractère)
            operator_found = False
            for op_len in [3, 2, 1]:
                if i + op_len <= len(line):
                    potential_op = line[i:i+op_len]
                    if potential_op in OPERATORS:
                        if current_word:
                            tokens.append(''.join(current_word))
                            current_word = []
                        tokens.append(potential_op)
                        i += op_len
                        operator_found = True
                        break
            
            if not operator_found:
                current_word.append(char)
                i += 1
        
        # Ajouter le dernier mot
        if current_word:
            tokens.append(''.join(current_word))
        
        return [token for token in tokens if token]

    @staticmethod
    def tokenize_fragment(fragment):
        """Tokenise un fragment entier."""
        tokenized = []
        function_regex = re.compile(r'function\d+')
        is_backward_slice = False
        
        for line in fragment:
            tokens = FragmentVectorizer.tokenize(line)
            tokenized.extend(tokens)
            
            # Détecter si c'est un backward slice
            if any(function_regex.match(token) for token in tokens):
                is_backward_slice = True
        
        return tokenized, is_backward_slice

    def add_fragment(self, fragment):
        """Ajoute un fragment à la liste pour l'entraînement."""
        tokenized_fragment, is_backward = FragmentVectorizer.tokenize_fragment(fragment)
        self.fragments.append(tokenized_fragment)
        
        if is_backward:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

    def vectorize(self, fragment):
        """Vectorise un fragment en utilisant le modèle Word2Vec entraîné."""
        tokenized_fragment, is_backward = FragmentVectorizer.tokenize_fragment(fragment)
        
        # Matrice de vecteurs (100 tokens max)
        vectors = np.zeros((100, self.vector_length))
        
        # Remplir la matrice selon le sens (forward ou backward)
        for i in range(min(len(tokenized_fragment), 100)):
            if is_backward:
                # Backward: remplir depuis la fin
                vectors[100 - 1 - i] = self.embeddings[tokenized_fragment[len(tokenized_fragment) - 1 - i]]
            else:
                # Forward: remplir depuis le début
                vectors[i] = self.embeddings[tokenized_fragment[i]]
        
        return vectors

    def train_model(self):
        """Entraîne le modèle Word2Vec et conserve les embeddings."""
        model = Word2Vec(
            self.fragments, 
            min_count=1, 
            vector_size=self.vector_length, 
            sg=0,  # CBOW
            window=5,
            workers=4
        )
        
        self.embeddings = model.wv
        
        # Libérer la mémoire
        del model
        del self.fragments