import pandas as pd
import re

def classify_toxicity(text):
    if pd.isna(text) or not text.strip():
        return 0
    
    text_lower = text.lower()
    
    # Non-toxic patterns
    neutral = ['gg', 'gl', 'hf', 'wp', 'ty', 'thanks', 'sorry', 'ok', 'np', 'yes', 'no']
    if text_lower.strip() in neutral:
        return 0
    
    # Toxic patterns
    toxic_patterns = [
        r'\b(fuck|shit|bitch|cunt|ass|dick|bastard|whore|faggot|nigga|retard|idiot|moron|stupid)\b',
        r'\b(noob|feeder|trash|useless|cancer|aids|kill yourself|kys|die|autism)\b',
        r'\b(stfu|shut up|fuck off|fuck you|suck|hate)\b',
    ]
    
    for pattern in toxic_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    return 0

# Read CSV
df = pd.read_csv('dota2_test(1).csv')

# Add toxicity column
df['toxicity'] = df['text'].apply(classify_toxicity)

# Save results
df.to_csv('dota2_labeled(1).csv', index=False)

# Print statistics
print(f"Total: {len(df)}")
print(f"Toxic [1]: {(df['toxicity'] == 1).sum()}")
print(f"Non-toxic [0]: {(df['toxicity'] == 0).sum()}")