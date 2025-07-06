import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from config.transformer.TabTransformer import TabTransformer
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

class WIDE_TabTransformer:
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.dropout = args.dropout

        # Préparation des données
        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values
        
        # Augmentation des données
        self.vectors, self.labels = self.augment_data(self.vectors, self.labels)
        
        # Division train/test
        self.prepare_data()
        
        # Construction du modèle
        self.model = self.build_model()

    def augment_data(self, vectors, labels):
        """Augmentation équilibrée des données."""
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        
        print(f"Original: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        # Identifier la classe minoritaire
        if len(positive_indices) < len(negative_indices):
            minority_indices = positive_indices
            minority_label = 1
            samples_needed = len(negative_indices) - len(positive_indices)
        else:
            minority_indices = negative_indices
            minority_label = 0
            samples_needed = len(positive_indices) - len(negative_indices)
        
        # Limiter l'augmentation
        samples_needed = min(samples_needed, len(minority_indices))
        
        # Générer des échantillons augmentés
        augmented_vectors = [vectors]
        augmented_labels = [labels]
        
        for _ in range(samples_needed):
            idx = np.random.choice(minority_indices)
            original_vector = vectors[idx]
            
            # Appliquer l'augmentation
            augmented_vector = self.apply_augmentation(original_vector)
            augmented_vectors.append([augmented_vector])
            augmented_labels.append([minority_label])
        
        final_vectors = np.concatenate(augmented_vectors)
        final_labels = np.concatenate(augmented_labels)
        
        pos_count = np.sum(final_labels == 1)
        neg_count = np.sum(final_labels == 0)
        print(f"Augmented: {pos_count} positive, {neg_count} negative")
        
        return final_vectors, final_labels

    def apply_augmentation(self, vector):
        """Applique des transformations d'augmentation modérées."""
        augmented = vector.copy()
        
        # Bruit gaussien léger
        noise = np.random.normal(0, 0.02, size=augmented.shape)
        augmented += noise
        
        # Scaling léger
        scale = np.random.uniform(0.98, 1.02)
        augmented *= scale
        
        return augmented

    def prepare_data(self):
        """Prépare les données pour l'entraînement."""
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            self.vectors, self.labels, 
            test_size=0.15, 
            stratify=self.labels, 
            random_state=42
        )
        
        # Division Wide/TabTransformer
        split_point = 50
        self.x_train_wide = X_train[:, :split_point]
        self.x_train_tab = X_train[:, split_point:]
        self.x_test_wide = X_test[:, :split_point]
        self.x_test_tab = X_test[:, split_point:]
        
        # Conversion en catégoriel
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        
        # Poids des classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        self.class_weight = {i: w for i, w in enumerate(class_weights)}

    def build_model(self):
        """Construit le modèle Wide + TabTransformer."""
        # Entrées
        input_wide = Input(shape=(self.x_train_wide.shape[1], self.x_train_wide.shape[2]))
        input_tab = Input(shape=(self.x_train_tab.shape[1], self.x_train_tab.shape[2]))
        
        # Normalisation
        wide_norm = LayerNormalization()(input_wide)
        tab_norm = LayerNormalization()(input_tab)
        
        # Partie TabTransformer
        tab_transformer = TabTransformer(
            num_heads=8,
            key_dim=64,
            ff_dim=256,
            num_layers=4,
            dropout_rate=self.dropout
        )(tab_norm)
        
        tab_flat = Flatten()(tab_transformer)
        tab_dense = Dense(128, activation='relu')(tab_flat)
        tab_dense = Dropout(self.dropout)(tab_dense)
        tab_dense = Dense(64, activation='relu')(tab_dense)
        tab_dense = Dropout(self.dropout)(tab_dense)
        
        # Partie Wide
        wide_flat = Flatten()(wide_norm)
        wide_dense = Dense(64, activation='relu')(wide_flat)
        wide_dense = Dropout(self.dropout)(wide_dense)
        wide_dense = Dense(32, activation='relu')(wide_dense)
        wide_dense = Dropout(self.dropout)(wide_dense)
        
        # Fusion
        merged = Concatenate()([wide_dense, tab_dense])
        
        # Couches finales
        final = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(merged)
        final = BatchNormalization()(final)
        final = Dropout(self.dropout)(final)
        
        final = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(final)
        final = BatchNormalization()(final)
        final = Dropout(self.dropout)(final)
        
        output = Dense(2, activation='softmax')(final)
        
        # Modèle
        model = Model(inputs=[input_wide, input_tab], outputs=output)
        
        # Compilation
        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

    def train(self):
        """Entraîne le modèle."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            [self.x_train_wide, self.x_train_tab],
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def test(self):
        """Évalue le modèle sur les données de test."""
        results = self.model.evaluate(
            [self.x_test_wide, self.x_test_tab], 
            self.y_test, 
            batch_size=self.batch_size, 
            verbose=0
        )
        
        print(f"\nAccuracy: {results[1]:.4f}")
        
        # Prédictions
        predictions = self.model.predict(
            [self.x_test_wide, self.x_test_tab], 
            batch_size=self.batch_size, 
            verbose=0
        )
        
        # Matrice de confusion
        y_true = np.argmax(self.y_test, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Métriques
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"False Positive Rate: {fp / (fp + tn):.4f}")
        print(f"False Negative Rate: {fn / (fn + tp):.4f}")

    def plot_training_curves(self, history):
        """Affiche les courbes d'entraînement."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Résumé
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")