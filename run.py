import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from config.fragment_vectorizer import FragmentVectorizer
from config.models.wide_tabtransformer import WIDE_TabTransformer
import pandas as pd
from config.arg_parser import parameter_parser
import warnings
import numpy as np

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

args = parameter_parser()

def parse_file(filename):
    """Parse le fichier et extrait les fragments avec leurs valeurs."""
    print(f'Parsing file: {filename}')
    fragments = []
    
    with open(filename, "r", encoding="utf8") as file:
        current_fragment = []
        fragment_val = 0
        
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Séparateur de fragments
            if "-" * 40 in line:
                if current_fragment:
                    fragments.append((current_fragment, fragment_val))
                    current_fragment = []
                    fragment_val = 0
            # Ligne avec valeur numérique
            elif stripped.isdigit():
                fragment_val = int(stripped)
            # Ligne de code
            else:
                current_fragment.append(stripped)
        
        # Ajouter le dernier fragment
        if current_fragment:
            fragments.append((current_fragment, fragment_val))
    
    return fragments

def create_vectors_dataframe(filename, vec_length):
    """Crée un DataFrame avec les vecteurs des fragments."""
    fragments_data = parse_file(filename)
    vectorizer = FragmentVectorizer(vec_length)
    
    # Collecter tous les fragments
    print("Collecting fragments...")
    for fragment, val in fragments_data:
        vectorizer.add_fragment(fragment)
    
    print(f'Found {len(fragments_data)} fragments')
    print("Training Word2Vec model...")
    vectorizer.train_model()
    
    # Vectoriser les fragments
    print("Vectorizing fragments...")
    vectors_data = []
    for fragment, val in fragments_data:
        vector = vectorizer.vectorize(fragment)
        vectors_data.append({"vector": vector, "val": val})
    
    return pd.DataFrame(vectors_data)

def main():
    filename = args.filename
    base_name = os.path.splitext(os.path.basename(filename))[0]
    dataset_path = f"config/train_data/{base_name}_fragment_vectors_tabtransformer.pkl"
    
    print(f"Dataset path: {dataset_path}")
    
    # Charger ou créer le dataset
    if os.path.exists(dataset_path):
        print("Loading existing vector dataframe...")
        df = pd.read_pickle(dataset_path)
    else:
        print('Generating vector dataframe...')
        df = create_vectors_dataframe(filename, args.vec_length)
        
        # Sauvegarder le dataset
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df.to_pickle(dataset_path)
        print(f"Dataframe saved to {dataset_path}")
    
    # Informations sur le dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Vector shape: {df.iloc[0, 0].shape}")
    print(f"Labels distribution:\n{df.iloc[:, 1].value_counts()}")
    
    # Entraîner le modèle
    print("\nInitializing model...")
    model = WIDE_TabTransformer(df, args)
    
    print("\nModel architecture:")
    model.model.summary()
    
    print("\nStarting training...")
    history = model.train()
    model.plot_training_curves(history)
    
    print("\nEvaluating model...")
    model.test()

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()