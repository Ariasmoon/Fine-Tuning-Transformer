# ssense_classifier.py
import torch
import torch.nn as nn

class SuperSenseClassifier(nn.Module):
    def __init__(self, pretrained_model, hidden_size, num_classes):
        """
        :param pretrained_model: le backbone (DistilBERT, CamemBERT, etc.)
        :param hidden_size: la dimension de hidden_state (pretrained_model.config.hidden_size)
        :param num_classes: le nombre total de classes super-sense (dont la classe "*")
        """
        super(SuperSenseClassifier, self).__init__()
        self.bert = pretrained_model  # modèle pré-entraîné
        self.classifier = nn.Sequential( #Pour passer dans l'ordre dans les différentes couches
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, tokens, word_ids, target_indices):
        """
        :param tokens: dict venant du tokenizer (input_ids, attention_mask, etc.), batch unique
        :param word_ids: liste renvoyée par tokenizer(..., return_tensors='pt', is_split_into_words=True).word_ids()
        :param target_indices: indices de mots (dans la phrase originale) qu'on veut classifier
        :return: logits (num_target_words x num_classes) ou None si pas de cibles
        """
        outputs = self.bert(**tokens)['last_hidden_state'][0] #Renvoie les embeddings du batch

        # Moyenne des sous-tokens pour chaque mot-cible
        embeddings = []

        for idx in target_indices:
            subtoken_indices = [i for i, wid in enumerate(word_ids) if wid == idx]
            if subtoken_indices:
                # On récupère l'embedding du mot avec le même indice, et on fait la moyenne
                word_embedding = torch.mean(outputs[subtoken_indices, :], dim=0)
                
                embeddings.append(word_embedding)

        if not embeddings: # Si pas de target indice
            return None  # Ne renvoie pas d'embedding, bonne idée ? #mettre avant 

        embeddings = torch.stack(embeddings)  # (num_targets, hidden_size)
        logits = self.classifier(embeddings)  # (num_targets, num_classes)
        return logits
