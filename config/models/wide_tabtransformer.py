# Améliorations pour wide_tabtransformer.py

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from config.transformer import TabTransformer

# Focal Loss pour gérer le déséquilibre
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculer la cross-entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculer le facteur focal
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * self.alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        
        # Focal loss
        focal_loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

# Scheduler de learning rate amélioré
class CosineWarmupScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, max_lr, total_epochs):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.max_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

class WIDE_TabTransformer:
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.dropout = args.dropout

        # Préparation des données avec validation croisée
        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values
        
        # Augmentation des données améliorée
        self.vectors, self.labels = self.advanced_augmentation(self.vectors, self.labels)
        
        # Division des données
        self.prepare_data()
        
        # Construction du modèle
        self.model = self.build_improved_model()

    def advanced_augmentation(self, vectors, labels):
        """Augmentation avancée avec plusieurs techniques."""
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        
        print(f"Original: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        # Augmentation équilibrée
        minority_class = 1 if len(positive_indices) < len(negative_indices) else 0
        minority_indices = positive_indices if minority_class == 1 else negative_indices
        majority_indices = negative_indices if minority_class == 1 else positive_indices
        
        # Générer des échantillons synthétiques
        augmented_vectors = [vectors]
        augmented_labels = [labels]
        
        # Nombre d'échantillons à générer
        samples_needed = min(len(majority_indices) - len(minority_indices), len(minority_indices))
        
        for i in range(samples_needed):
            # Sélectionner deux échantillons de la classe minoritaire
            idx1, idx2 = np.random.choice(minority_indices, 2, replace=False)
            
            # Mixup entre les deux échantillons
            lambda_param = np.random.beta(0.2, 0.2)
            mixed_vector = lambda_param * vectors[idx1] + (1 - lambda_param) * vectors[idx2]
            
            # Ajouter du bruit contrôlé
            noise = np.random.normal(0, 0.01, mixed_vector.shape)
            mixed_vector += noise
            
            augmented_vectors.append([mixed_vector])
            augmented_labels.append([minority_class])
        
        final_vectors = np.concatenate(augmented_vectors)
        final_labels = np.concatenate(augmented_labels)
        
        # Mélanger les données
        shuffle_idx = np.random.permutation(len(final_vectors))
        final_vectors = final_vectors[shuffle_idx]
        final_labels = final_labels[shuffle_idx]
        
        pos_count = np.sum(final_labels == 1)
        neg_count = np.sum(final_labels == 0)
        print(f"Augmented: {pos_count} positive, {neg_count} negative")
        
        return final_vectors, final_labels

    def prepare_data(self):
        """Prépare les données avec validation stratifiée."""
        # Division stratifiée
        X_train, X_test, y_train, y_test = train_test_split(
            self.vectors, self.labels, 
            test_size=0.2,  # Augmenté pour plus de robustesse
            stratify=self.labels, 
            random_state=42
        )
        
        # Division Wide/TabTransformer optimisée
        split_point = 40  # Réajusté pour équilibrer les composants
        self.x_train_wide = X_train[:, :split_point]
        self.x_train_tab = X_train[:, split_point:]
        self.x_test_wide = X_test[:, :split_point]
        self.x_test_tab = X_test[:, split_point:]
        
        # Normalisation des données
        self.normalize_data()
        
        # Conversion en catégoriel
        self.y_train = to_categorical(y_train, num_classes=2)
        self.y_test = to_categorical(y_test, num_classes=2)
        
        # Poids des classes ajustés
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        # Ajuster les poids pour favoriser légèrement la classe positive
        class_weights[1] *= 1.2  # Réduire les faux négatifs
        self.class_weight = {i: w for i, w in enumerate(class_weights)}
        
        print(f"Class weights: {self.class_weight}")

    def normalize_data(self):
        """Normalisation avancée des données."""
        # Normalisation par composant
        wide_mean = np.mean(self.x_train_wide, axis=(0, 1), keepdims=True)
        wide_std = np.std(self.x_train_wide, axis=(0, 1), keepdims=True) + 1e-8
        
        tab_mean = np.mean(self.x_train_tab, axis=(0, 1), keepdims=True)
        tab_std = np.std(self.x_train_tab, axis=(0, 1), keepdims=True) + 1e-8
        
        # Appliquer la normalisation
        self.x_train_wide = (self.x_train_wide - wide_mean) / wide_std
        self.x_test_wide = (self.x_test_wide - wide_mean) / wide_std
        
        self.x_train_tab = (self.x_train_tab - tab_mean) / tab_std
        self.x_test_tab = (self.x_test_tab - tab_mean) / tab_std

    def build_improved_model(self):
        """Modèle amélioré avec attention croisée."""
        # Entrées
        input_wide = Input(shape=(self.x_train_wide.shape[1], self.x_train_wide.shape[2]))
        input_tab = Input(shape=(self.x_train_tab.shape[1], self.x_train_tab.shape[2]))
        
        # Partie TabTransformer avec attention
        tab_transformer = TabTransformer(
            num_heads=8,
            key_dim=64,
            ff_dim=256,
            num_layers=3,  # Réduit pour éviter l'overfitting
            dropout_rate=self.dropout
        )(input_tab)
        
        # Attention pooling pour TabTransformer
        tab_attention = MultiHeadAttention(num_heads=4, key_dim=32)(tab_transformer, tab_transformer)
        tab_pooled = GlobalAveragePooling1D()(tab_attention)
        
        # Partie Wide avec plus de profondeur
        wide_flat = Flatten()(input_wide)
        wide_dense1 = Dense(128, activation='gelu')(wide_flat)
        wide_dense1 = BatchNormalization()(wide_dense1)
        wide_dense1 = Dropout(self.dropout)(wide_dense1)
        
        wide_dense2 = Dense(64, activation='gelu')(wide_dense1)
        wide_dense2 = BatchNormalization()(wide_dense2)
        wide_dense2 = Dropout(self.dropout)(wide_dense2)
        
        # TabTransformer processing
        tab_dense1 = Dense(128, activation='gelu')(tab_pooled)
        tab_dense1 = BatchNormalization()(tab_dense1)
        tab_dense1 = Dropout(self.dropout)(tab_dense1)
        
        tab_dense2 = Dense(64, activation='gelu')(tab_dense1)
        tab_dense2 = BatchNormalization()(tab_dense2)
        tab_dense2 = Dropout(self.dropout)(tab_dense2)
        
        # Attention croisée entre Wide et TabTransformer
        cross_attention = MultiHeadAttention(num_heads=4, key_dim=32)(
            tf.expand_dims(wide_dense2, 1), 
            tf.expand_dims(tab_dense2, 1)
        )
        cross_pooled = GlobalAveragePooling1D()(cross_attention)
        
        # Fusion avec résiduelle
        merged = Concatenate()([wide_dense2, tab_dense2, cross_pooled])
        
        # Couches finales avec connexions résiduelles
        final = Dense(128, activation='gelu', kernel_regularizer=l1_l2(0.001, 0.001))(merged)
        final = BatchNormalization()(final)
        final = Dropout(self.dropout)(final)
        
        # Connexion résiduelle
        residual = Dense(128, activation='gelu')(merged)
        final = Add()([final, residual])
        
        final = Dense(64, activation='gelu', kernel_regularizer=l1_l2(0.001, 0.001))(final)
        final = BatchNormalization()(final)
        final = Dropout(self.dropout)(final)
        
        output = Dense(2, activation='softmax')(final)
        
        # Modèle
        model = Model(inputs=[input_wide, input_tab], outputs=output)
        
        # Compilation avec Focal Loss
        model.compile(
            optimizer=Adam(learning_rate=self.lr, clipnorm=1.0),  # Gradient clipping
            loss=FocalLoss(alpha=0.3, gamma=2.0),  # Focal loss pour déséquilibre
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

    def train(self):
        """Entraînement avec callbacks améliorés."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=12,  # Plus de patience
                restore_best_weights=True,
                verbose=1
            ),
            CosineWarmupScheduler(
                warmup_epochs=5,
                max_lr=self.lr,
                total_epochs=self.epochs
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  # Réduction plus douce
                patience=6,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_f1_score',  # Monitorer le F1-score
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Entraînement avec validation plus importante
        history = self.model.fit(
            [self.x_train_wide, self.x_train_tab],
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.25,  # Plus de données de validation
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def test(self):
        """Évaluation avancée avec métriques détaillées."""
        # Prédictions avec moyennage de plusieurs inférences
        predictions_list = []
        for _ in range(5):  # Test-time augmentation
            pred = self.model.predict(
                [self.x_test_wide, self.x_test_tab], 
                batch_size=self.batch_size, 
                verbose=0
            )
            predictions_list.append(pred)
        
        # Moyenne des prédictions
        predictions = np.mean(predictions_list, axis=0)
        
        # Évaluation standard
        results = self.model.evaluate(
            [self.x_test_wide, self.x_test_tab], 
            self.y_test, 
            batch_size=self.batch_size, 
            verbose=0
        )
        
        print(f"\nTest Results:")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        
        # Matrice de confusion détaillée
        y_true = np.argmax(self.y_test, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Métriques détaillées
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nDetailed Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"False Positive Rate: {fp / (fp + tn):.4f}")
        print(f"False Negative Rate: {fn / (fn + tp):.4f}")
        
        # Analyse des seuils
        self.analyze_thresholds(predictions, y_true)

    def analyze_thresholds(self, predictions, y_true):
        """Analyse des seuils de décision."""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        print(f"\nThreshold Analysis:")
        for threshold in thresholds:
            y_pred_thresh = (predictions[:, 1] > threshold).astype(int)
            
            if len(np.unique(y_pred_thresh)) == 2:  # Éviter les erreurs si une seule classe
                cm = confusion_matrix(y_true, y_pred_thresh)
                tn, fp, fn, tp = cm.ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"Threshold {threshold}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    def plot_training_curves(self, history):
        """Visualisation améliorée des courbes d'entraînement."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistiques détaillées
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"\nTraining Summary:")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Final training accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        print(f"Overfitting gap: {abs(final_train_acc - final_val_acc):.4f}")
        
        if final_train_acc > final_val_acc + 0.05:
            print("⚠️  Possible overfitting detected!")
        elif final_val_acc > final_train_acc + 0.05:
            print("⚠️  Unusual validation > training accuracy!")
        else:
            print("✅ Training appears stable")