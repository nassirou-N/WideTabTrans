import argparse

def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contract Vulnerability Detection Using Wide and TabTransformer Neural Network')

    parser.add_argument('filename', type=str, help="name of file to process")
    parser.add_argument('-vt', type=str, choices=['ts', 're', 'io'])
        
    # Optimiseur et learning rate
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (optimized for stability)')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate (balanced for regularization)')
    
    # Dimensions vectorielles
    parser.add_argument('--vec_length', type=int, default=200, help='vector dimension')
    
    # Paramètres d'entraînement
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs (increased for better convergence)')
    parser.add_argument('-b', '--batch_size', type=int, default=24, help='batch size (optimized for memory/performance)')
    
    # Paramètres spécifiques au TabTransformer
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads in TabTransformer')
    parser.add_argument('--key_dim', type=int, default=64, help='dimension of keys/values in TabTransformer')
    parser.add_argument('--ff_dim', type=int, default=256, help='feed-forward dimension in TabTransformer')
    parser.add_argument('--num_layers', type=int, default=3, help='number of transformer layers (reduced for stability)')
    
    # Paramètres pour la division Wide/TabTransformer
    parser.add_argument('--wide_features', type=int, default=40, help='number of features for wide component (optimized)')
    
    # Nouveaux paramètres pour l'optimisation
    parser.add_argument('--focal_alpha', type=float, default=0.3, help='alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma parameter for focal loss')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='gradient clipping norm')
    parser.add_argument('--test_time_aug', type=int, default=5, help='number of test-time augmentation runs')
    
    # Paramètres pour la validation
    parser.add_argument('--val_split', type=float, default=0.25, help='validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.2, help='test split ratio')
    
    # Paramètres pour l'augmentation des données
    parser.add_argument('--augment_ratio', type=float, default=1.0, help='ratio of augmented samples to generate')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='alpha parameter for mixup augmentation')
    parser.add_argument('--noise_std', type=float, default=0.01, help='standard deviation for noise augmentation')

    return parser.parse_args()