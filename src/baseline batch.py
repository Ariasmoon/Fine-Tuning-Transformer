import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import time
from torch.utils.data import DataLoader, TensorDataset

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
                        df = pd.DataFrame(sentence, columns=columns)
                        sentences.append(df)
                    except ValueError as e:
                        print(f"Erreur avec une phrase : {sentence}")
                        raise e
                    sentence = []
            else:
                parts = line.strip().split('\t')
                if len(parts) != len(columns):
                    print(f"Ligne mal formatée : {parts}")
                    continue
                sentence.append(parts)

    return sentences

def prepare_sentence_embeddings(sentences, tokenizer, model, target_pos, tagvocab):
    all_sentence_embeddings = []
    all_sentence_labels = []

    # On met le modèle en mode eval + on désactive le calcul du gradient
    model.eval()
    with torch.no_grad():
        for sentence_df in sentences:
            tokens = sentence_df["FORM"].tolist()
            pos_tags = sentence_df["UPOS"].tolist()
            labels_col = sentence_df["FRSEMCOR:NOUN"].tolist()

            # Tokenisation par mots
            tokenized = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            # Passage dans DistilBERT
            outputs = model(**tokenized)
            # Récupère la dernière couche 
            contextual_embeddings = outputs.last_hidden_state[0]  # shape: (seq_len, hidden_dim)

            word_ids = tokenized.word_ids(batch_index=0) # liste qui mappe sous-token -> index du mot, peut avoir plusieurs fois le même indice

            target_embeddings = []
            target_labels = []

            for idx, pos in enumerate(pos_tags):
                if pos in target_pos:
                    # Récupére la liste des positions des sous-tokens correspondant à ce mot
                    subtoken_positions = [i for i, w_id in enumerate(word_ids) if w_id == idx]
                    if not subtoken_positions:
                        continue
                    # Moyenne des sous-tokens
                    word_embedding = contextual_embeddings[subtoken_positions].mean(dim=0)
                    target_embeddings.append(word_embedding)

                    # Construit l'index de label
                    label_idx = tagvocab[labels_col[idx]]
                    target_labels.append(label_idx)

            if len(target_embeddings) == 0:
                all_sentence_embeddings.append(torch.empty(0, model.config.hidden_size))
                all_sentence_labels.append(torch.empty(0, dtype=torch.long))
            else:
                sent_emb_tensor = torch.stack(target_embeddings, dim=0)
                sent_label_tensor = torch.tensor(target_labels, dtype=torch.long)
                all_sentence_embeddings.append(sent_emb_tensor)
                all_sentence_labels.append(sent_label_tensor)

    return all_sentence_embeddings, all_sentence_labels

class SuperSenseClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SuperSenseClassifier, self).__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, token_embeddings):
        x = self.relu(self.layer(token_embeddings))
        logits = self.classifier(x)
        return logits

def evaluate_accuracy(model, dataloader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens_batch, labels_batch in dataloader:
            tokens_batch = tokens_batch.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(tokens_batch)  # (batch_size, num_classes)
            predictions = torch.argmax(logits, dim=1)  # (batch_size,)

            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

    model.train()
    accuracy = 100.0 * correct / max(total, 1)
    return accuracy

def main():
    train_file = "c:/Users/thoma/Desktop/Fac/PSTALN/TP1/sequoia/sequoia-ud.parseme.frsemcor.simple.train"
    dev_file   = "c:/Users/thoma/Desktop/Fac/PSTALN/TP1/sequoia/sequoia-ud.parseme.frsemcor.simple.dev"
    model_save_path = "c:/Users/thoma/Desktop/Fac/PSTALN/super_sense_finetuned_model.pth"

    model_name = "camembert-base" #  "camembert-base" pour camembert ou "distilbert-base-multilingual-cased" pour distilbert

    # Hyper-paramètres
    target_pos = {"NOUN", "PROPN", "NUM"}
    num_epochs = 10
    learning_rate = 1e-3
    hidden_dim_classifier = 128
    batch_size = 32  

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Appareil utilisé :", device)

    train_sentences = load_data(train_file)
    dev_sentences = load_data(dev_file)

    tagvocab = defaultdict(lambda: len(tagvocab))
    tagvocab["*"]  # étiquette par défaut, si étiquette absente, alors on lui met *

    for sentence_df in train_sentences:
        for label in sentence_df["FRSEMCOR:NOUN"]:
            _ = tagvocab[label]  # force la création d'un index

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    distilbert = AutoModel.from_pretrained(model_name)
    distilbert.eval()  # on s'assure qu'il est en mode eval
    for p in distilbert.parameters():
        p.requires_grad = False  # on fige DistilBERT

    train_sentence_embeddings, train_sentence_labels = prepare_sentence_embeddings(
        train_sentences, tokenizer, distilbert, target_pos, tagvocab
    )

    total_train_tokens = sum(len(lbls) for lbls in train_sentence_labels)

    dev_sentence_embeddings, dev_sentence_labels = prepare_sentence_embeddings(
        dev_sentences, tokenizer, distilbert, target_pos, tagvocab
    )
    total_dev_tokens = sum(len(lbls) for lbls in dev_sentence_labels)
    train_tokens_list = []
    train_labels_list = []
    for emb, lab in zip(train_sentence_embeddings, train_sentence_labels):
        if emb.size(0) > 0:
            train_tokens_list.append(emb)
            train_labels_list.append(lab)

    if len(train_tokens_list) > 0:
        train_tokens_tensor = torch.cat(train_tokens_list, dim=0)
        train_labels_tensor = torch.cat(train_labels_list, dim=0)
    else:
        train_tokens_tensor = torch.empty(0, distilbert.config.hidden_size)
        train_labels_tensor = torch.empty(0, dtype=torch.long)

    dev_tokens_list = []
    dev_labels_list = []
    for emb, lab in zip(dev_sentence_embeddings, dev_sentence_labels):
        if emb.size(0) > 0:
            dev_tokens_list.append(emb)
            dev_labels_list.append(lab)

    if len(dev_tokens_list) > 0:
        dev_tokens_tensor = torch.cat(dev_tokens_list, dim=0)
        dev_labels_tensor = torch.cat(dev_labels_list, dim=0)
    else:
        dev_tokens_tensor = torch.empty(0, distilbert.config.hidden_size)
        dev_labels_tensor = torch.empty(0, dtype=torch.long)

    train_dataset = TensorDataset(train_tokens_tensor, train_labels_tensor)
    dev_dataset = TensorDataset(dev_tokens_tensor, dev_labels_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(tagvocab)
    input_dim = distilbert.config.hidden_size
    classifier = SuperSenseClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim_classifier,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Boucle d'entraînement
    print("Début de l'entraînement...")
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.0

        for tokens_batch, labels_batch in train_dataloader:
            tokens_batch = tokens_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Forward
            logits = classifier(tokens_batch)  # (batch_size, num_classes)
            loss = criterion(logits, labels_batch)

            # Backprop + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Évaluation sur dev
        dev_acc = evaluate_accuracy(classifier, dev_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Dev Acc: {dev_acc:.2f}%")

    torch.save({
        'model_state_dict': classifier.state_dict(),
        'tagvocab': dict(tagvocab),
        'pretrained_model_name': model_name,
        'num_classes': num_classes
    }, model_save_path)
    print(f"Modèle sauvegardé dans : {model_save_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    temps_ecoule = end_time - start_time
    print(f"Temps d'exécution : {temps_ecoule:.4f} secondes")
