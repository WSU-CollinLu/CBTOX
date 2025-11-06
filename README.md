# CBTOX: Context Based TOXicity detection with text augmentation
![Logo of Western Sydney University](https://github.com/WSU-CollinLu/CBTOX/blob/main/github-images/WSU.png) 
## Western Sydney University CACE 2025 Capstone Project for Cyber Security and Behaviour Bachelor Students

# Contributors
(2025) Anita Chun, Collin Lu, Luke Armitage-Masi, Jasmine Flora, Uswah Rahman

# Western Sydney University
![Image of Western Sydney University's Cybersecurity Aid and Community Engagement program](https://github.com/WSU-CollinLu/CBTOX/blob/main/github-images/westernCACE.png)

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
- **Singh, 2022:**
  Sentiment Analysis of Dota 2 videogame chat in context of Cyber-bullying MSc Research Project Masters of Science in Data Analytics

- **Fesalbon et al.,2024:**
   Fine-tuning Pre-trained Language Models to Detect In-game Trash Talks

- **Fortunatus et al., 2020:**
  Combining textual features to detect cyberbullying in social media posts

- **Gandolfi & Ferdig, 2021:**
   Sharing dark sides on game service platforms: Disruptive behaviors and toxicity in DOTA2 through a platform lens

- **J.Angel Diaz-Garcia & Carvalho, 2025:**
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

## Design Phases
- Phase 1: Data Collection and Processing
- Phase 2: Workflow Model
- Phase 3: Machine Learning Model Development
- Phase 4: Integration

The DeBerta v2 model is trained with an initial dataset from SafeTalk then fine-tuned using a Dota-2 specific dataset from Hugging Face. We will compare the accuracy, precision and F1 scores of the results between the original dataset and text augmented dataset.

# Data Analysis
Since SMOTE is limited to numerical data, text augmentation techniques inspired by SMOTE principles 
were applied to handle class imbalance. These techniques included synonym replacement, word insertion, 
word order variation, and paraphrasing to expand minority toxic classes with linguistically valid 
variations. 
![Pie charts showing the initial data set message distribution and the augmented data set](https://github.com/WSU-CollinLu/CBTOX/blob/main/github-images/dataBalancing.png)

# Results
**Training Loss (Baseline vs Augmented)**
This figure shows that both models exhibit a steady 
decline in training loss over the five epochs. However, 
the augmented model maintains a consistently lower 
loss value across all epochs, demonstrating faster and 
more stable convergence due to exposure to diverse 
and balanced training samples. 
![Training Loss graph depicting the augmented model's decrease in training loss per epoch compared to baseline](https://github.com/WSU-CollinLu/CBTOX/blob/main/github-images/trainingLoss.png)

**Validation Loss (Baseline vs Augmented)**
The validation loss plot mirrors the training loss 
pattern, confirming that the augmented model 
generalizes better to unseen data. The narrower gap 
between training and validation losses indicates 
reduced overfitting and stronger model robustness. 
![Validation Loss graph depicting the augmented model's decrease in validation loss per epoch compared to baseline](https://github.com/WSU-CollinLu/CBTOX/blob/main/github-images/validationLoss.png)

**F1 Score Progression (Baseline vs Augmented)**
This figure highlights a consistent increase in F1 
scores across all epochs, with the augmented model 
outperforming the baseline in every iteration. By the 
fifth epoch, the augmented model achieved a notably 
higher F1 score, confirming balanced improvements 
in precision and recall across all toxicity levels. 
![F1 Score Graph depicting the augmented model's increase in performance per epoch compared to baseline](https://github.com/WSU-CollinLu/CBTOX/blob/main/github-images/f1Score.png)

# Conclusion
This study successfully developed and evaluated a transformer-based model, DeBERTa, for detecting 
toxic behaviour in Dota 2 gaming chat environments. Through a structured methodology encompassing 
data collection, preprocessing, model training, and evaluation, the research demonstrated how modern 
Natural Language Processing (NLP) techniques can address one of the most challenging aspects of online 
interaction—context-dependent toxicity. The integration of text augmentation techniques, adapted from 
SMOTE principles, proved to be a pivotal step in overcoming class imbalance and enhancing the model’s 
sensitivity to minority classes. This innovation enabled the model to recognize subtle linguistic cues, 
sarcasm, and coded toxicity more effectively than baseline approaches, marking a significant step forward 
in gaming moderation research.

Quantitative results revealed notable improvements in performance metrics, with accuracy increasing 
from 91.2% to 93.8% and recall for minority classes improving by over 20 percentage points following 
augmentation. Qualitative analyses further highlighted the model’s ability to generalize across phrasal 
variations and contextual nuances common in gaming discourse, such as “gg ez” or “feed harder.” The 
comparison with prior studies (e.g., Singh, 2022; Fesalbon et al., 2024; Lee et al., 2025) confirmed that 
DeBERTa’s disentangled attention and contextual reasoning outperform traditional lexicon-based and 
hybrid approaches, reinforcing the superiority of fine-tuned transformer models for toxicity detection. 
Moreover, the study’s findings align with Diaz-Garcia and Carvalho (2025), who reported that 
transformer-based systems outperform classical ML models in detecting complex, implicit forms of 
toxicity.

From a practical standpoint, this research contributes to the growing field of responsible AI in online 
moderation. The model’s architecture and real-time inference capability demonstrate clear potential for 
deployment across multiplayer games and digital platforms. However, limitations remain in dataset 
diversity, computational demands, and ethical considerations such as privacy and fairness. These highlight 
the importance of transparency, user consent, and appeal mechanisms when integrating AI moderation 
tools into real-world systems. Addressing these challenges through multilingual expansion, explainable 
AI modules, and hybrid human-AI moderation frameworks will be critical for future development.

In conclusion, this project not only validates the potential of transformer-based architectures like 
DeBERTa for gaming toxicity detection but also emphasizes the importance of linguistic and contextual
awareness in AI moderation systems. By combining data-driven augmentation with domain-specific 
fine-tuning, it lays a foundation for scalable, culturally adaptive, and ethically responsible approaches to 
combating toxicity in online environments. The outcomes of this study underscore the transformative role 
of AI in fostering safer, more inclusive digital spaces—an essential step toward enhancing user 
experience and community well-being in the evolving world of online gaming. 
