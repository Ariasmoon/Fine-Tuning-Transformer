#!/usr/bin/env python3
# train_ssense.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import time
from classifier import SuperSenseClassifier

def load_data(file_path):

    columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS",
               "HEAD", "DEPREL", "DEPS", "MISC",
               "PARSEME:MWE", "FRSEMCOR:NOUN", "PARSEME:NE"]
    sentences = []
    sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                continue
            elif line.strip() == '':
                if sentence:
                    try:
                        sentences.append(pd.DataFrame(sentence, columns=columns))
                    except ValueError as e:
                        print(f"Erreur avec une phrase : {sentence}")
                        raise e
                    sentence = []
            else:
                parts = line.strip().split('\t')
                # Vérifie si on a le bon nombre de colonnes
                if len(parts) != len(columns):
                    print(f"Ligne mal formatée : {parts}")
                    continue
                sentence.append(parts)

    return sentences


def prepare_data(sentences, tokenizer, target_pos, tagvocab):

    data = []
    labels = []

    i = 0
    for sentence_df in sentences:
        i += 1

        tokens = sentence_df['FORM'].tolist()
        pos_tags = sentence_df['UPOS'].tolist()
        labels_col = sentence_df['FRSEMCOR:NOUN'].tolist()

        if i == 1:  #  pour vérifier la première phrase
            print("tokens  =", tokens)
            print("pos_tags=", pos_tags)
            print("labels  =", labels_col)

        tokenized = tokenizer(tokens, is_split_into_words=True,
                              return_tensors='pt', padding=True)
        word_ids = tokenized.word_ids(batch_index=0) # liste qui mappe sous-token -> index du mot, peut avoir plusieurs fois le même indice

        sentence_labels = []
        target_indices = []

        for idx, pos in enumerate(pos_tags): 
            if pos in target_pos: # Tri des UPOS, si ils sont dans {"NOUN", "PROPN", "NUM"}
                
                sentence_labels.append(tagvocab[labels_col[idx]]) # Renvoie le label, si aucun label n'est trouvé, alors renvoie "*" par défaut
                target_indices.append(idx) # Renvoie la position du mot target dans la phrase, que l'on met dans data pour le classifieur,

        data.append((tokenized, word_ids, target_indices))  # Données pour l'entrainement
        labels.append(sentence_labels) # Renvoie les vrais labels pour chaque mot

    return data, labels


def main():
    # Hyper-paramètres
    model_name = "camembert-base"   #  "camembert-base" pour camembert ou "distilbert-base-multilingual-cased" pour distilbert
    lr = 1e-4
    num_epochs = 10
    batch_size = 32  

    train_file = "c:/Users/thoma/Desktop/Fac/PSTALN/TP1/sequoia/sequoia-ud.parseme.frsemcor.simple.train"
    dev_file   = "c:/Users/thoma/Desktop/Fac/PSTALN/TP1/sequoia/sequoia-ud.parseme.frsemcor.simple.dev"
    save_path  = "c:/Users/thoma/Desktop/Fac/PSTALN/super_sense_finetuned_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Appareil utilisé :", device)

    train_sentences = load_data(train_file)
    dev_sentences = load_data(dev_file)

    tagvocab = defaultdict(lambda: len(tagvocab))
    tagvocab["*"]   # étiquette par défaut, si étiquette absente, alors on lui met *

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pretrained_model = AutoModel.from_pretrained(model_name)

    # (Dé)figer les couches DistilBERT
    for param in pretrained_model.parameters():
        param.requires_grad = False
    # figer toutes les couches sauf les 3 dernières:
    for param in pretrained_model.encoder.layer[-3:]:
        for p in param.parameters():
            p.requires_grad = True 

    # Prépare les données
    target_pos = {"NOUN", "PROPN", "NUM"}
    train_data, train_labels = prepare_data(train_sentences, tokenizer, target_pos, tagvocab)
    dev_data, dev_labels = prepare_data(dev_sentences, tokenizer, target_pos, tagvocab)

    num_classes = len(tagvocab)
    print(f"Nombre de classes super-sense: {num_classes}")

    classifier = SuperSenseClassifier(
        pretrained_model=pretrained_model,
        hidden_size=pretrained_model.config.hidden_size,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    print(f"Nombre d'exemples d'entraînement : {len(train_labels)}")

    train_dataset = list(zip(train_data, train_labels))

    classifier.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        random.shuffle(train_dataset)
        train_data_shuffled, train_labels_shuffled = zip(*train_dataset)

        # On itère en batch de taille 'batch_size'
        for i in range(0, len(train_data_shuffled), batch_size):
            batch_data = train_data_shuffled[i:i + batch_size]
            batch_labels = train_labels_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            batch_loss = 0.0
            count_in_batch = 0

            # On accumule les pertes phrase par phrase dans le batch
            for (tokens, word_ids, target_indices), labels_list in zip(batch_data, batch_labels):
                tokens = {k: v.to(device) for k, v in tokens.items()}
                logits = classifier(tokens, word_ids, target_indices)
                if logits is None:
                    continue

                labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)
                loss = criterion(logits, labels_tensor)
                batch_loss += loss
                count_in_batch += 1

            # On fait la moyenne des pertes dans le batch
            if count_in_batch > 0:
                batch_loss = batch_loss / count_in_batch

                # Rétropropagation
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

        # Calcul d'accuracy sur le dev
        dev_acc = evaluate_accuracy(classifier, dev_data, dev_labels, device)

        print(f"Epoch {epoch+1}/{num_epochs} - loss: {total_loss:.4f} - dev_acc: {dev_acc:.2f}")

    torch.save({
        'model_state_dict': classifier.state_dict(),
        'tagvocab': dict(tagvocab),
        'pretrained_model_name': model_name,
        'num_classes': num_classes
    }, save_path)
    print("Modèle sauvegardé dans", save_path)


def evaluate_accuracy(model, dev_data, dev_labels, device):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (tokens, word_ids, target_indices), labels_list in zip(dev_data, dev_labels):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(tokens, word_ids, target_indices)
            if logits is None:
                continue
            predictions = torch.argmax(logits, dim=1).cpu()
            for pred_label, gold_label in zip(predictions, labels_list):
                if pred_label.item() == gold_label:
                    correct += 1
            total += len(labels_list)
    model.train()
    return 100.0 * correct / max(1, total)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    temps_ecoule = end_time - start_time
    print(f"Temps d'exécution : {temps_ecoule:.4f} secondes")
