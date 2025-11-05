# CBTOX: Context Based TOXicity detection with text augmentation
Western Sydney University CACE 2025 Capstone Project for Cyber Security and Behaviour Bachelor Students

# Contributors
(2025) Anita Chun, Collin Lu, Luke Armitage-Masi, Jasmine Flora, Uswah Rahman

# Western Sydney University

# Problem Statement
Despite growing research into online toxicity, accurate detection remains a significant challenge due to 
the dynamic and nuanced nature of digital communication. Sarcasm, abbreviations, and informal 
expressions frequently lead to misclassification, limiting the reliability of existing models (Walia, 2023). 
Traditional machine learning approaches, while valuable for establishing baselines, often fail to capture 
deeper contextual signals and the behavioural intent behind messages. Transformer-based models such as 
BERT and DeBERTa show promise but continue to face issues related to dataset imbalance, 
generalization across platforms, and explainability (Khapre et al., 2025). Moreover, most existing 
frameworks are trained on general social media datasets rather than gaming-specific data, which limits 
their effectiveness in real-time, competitive environments like Dota 2 (Swati Pandita et al., 2024). 
Therefore, there is a need for a domain-specific AI framework that integrates behavioural insights and 
contextual understanding to improve the detection and management of toxic behaviour in online gaming. 

# Related Works
- **Singh 2022**
  Sentiment Analysis of Dota 2 videogame chat in context of Cyber-bullying MSc Research Project Masters of Science in Data Analytics

- **Fesalbon et al.,2024**
   Fine-tuning Pre-trained Language Models to Detect In-game Trash Talks

- **Fortunatus et al., 2020**
  Combining textual features to detect cyberbullying in social media posts

- **Gandolfi & Ferdig, 2021**
   Sharing dark sides on game service platforms: Disruptive behaviors and toxicity in DOTA2 through a platform lens

- J.Angel Diaz-Garcia & Carvalho, 2925
   A Literature Review of Textual Cyber Abuse Detection Using Cutting‐Edge Natural Language Processing Techniques: Language Models and Large Language Models

# Methodology
This study uses an in-depth research approach through reviewing existing case studies working on 
cyberbullying detection, understanding relevant excerpts of the language model’s code in which the 
language model is constructed upon, modifying the model to detect toxicity in online video game 
environments. The methodology combines qualitative and computational research components in the 
form of databases containing chatlogs from Dota 2 that will be utilized to develop an AI system based on 
the public DeBERTa model.  
Through a process including a literature review, data collection, data preprocessing, model training, and 
evaluation, the study aims to design, test and assess the performance of a DeBERTa model trained for the 
identification of toxic behavior in gaming related environments.

Design Phases
- Phase 1: Data Collection and Processing
- Phase 2: Workflow Model
- Phase 3: Machine Learning Model Development
- Phase 4: Integration

# Results
