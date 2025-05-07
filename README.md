# Étiqueteur Super-Senses avec Transformer pré-entraîné

![Langage](https://img.shields.io/badge/langage-Python-blue.svg)

## Description

Ce projet consiste à développer un étiqueteur de super-senses pour le français, basé sur des architectures Transformer (DistilBERT et CamemBERT). L’objectif est de comparer deux approches :

Baseline : extraction statique d'embeddings à partir d'un Transformer gelé.

Fine-Tuning : réentraînement partiel ou complet du Transformer pour ajuster les poids.

Le projet s’appuie sur le corpus Sequoia enrichi via FR-SeMCOR et se concentre sur la classification en 24 super-senses (Person, Location, Institution, Feeling, etc.) pour les mots de type NOUN, PROPN ou NUM.
