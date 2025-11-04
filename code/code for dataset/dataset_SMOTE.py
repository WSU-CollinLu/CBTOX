import pandas as pd
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

# STEP 1: Load and Label Dataset


# Load dataset (already labeled in your case)
df = pd.read_csv('Dota2_toxic_text.csv')

# Clean data
df = df.dropna(subset=['message', 'target'])
df = df[df['message'].str.strip() != '']

print("Original Dataset:")
print(f"Total: {len(df)}")
print(df['target'].value_counts().sort_index())


# STEP 2: Text Augmentation (SMOTE-style for text)
def augment_text(text):
    """Simple text augmentation - creates synthetic variations"""
    words = text.split()
    if len(words) < 2:
        return text
    
    # Random technique
    technique = random.choice(['swap', 'insert', 'delete'])
    
    if technique == 'swap' and len(words) >= 2:
        # Swap two random words
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    elif technique == 'insert' and len(words) >= 1:
        # Insert a random word from text
        insert_word = random.choice(words)
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, insert_word)
    
    elif technique == 'delete' and len(words) > 2:
        # Delete a random word
        del_pos = random.randint(0, len(words) - 1)
        words.pop(del_pos)
    
    return ' '.join(words)

# Balance dataset
target_count = df['target'].value_counts().max()
balanced_data = []

for label in [0, 1, 2]:
    class_data = df[df['target'] == label].copy()
    current_count = len(class_data)
    needed = target_count - current_count
    
    # Add original samples
    balanced_data.append(class_data)
    
    # Add synthetic samples if needed
    if needed > 0:
        samples_to_augment = class_data.sample(n=needed, replace=True, random_state=42)
        synthetic_messages = [augment_text(text) for text in samples_to_augment['message']]
        
        synthetic_df = pd.DataFrame({
            'message': synthetic_messages,
            'target': label
        })
        balanced_data.append(synthetic_df)

df_balanced = pd.concat(balanced_data, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced Dataset:")
print(f"Total: {len(df_balanced)}")
print(df_balanced['target'].value_counts().sort_index())


# STEP 3: Split Dataset (Train, Test, Validation)
# First split: 70% train, 30% temp (for test + val)
train_df, temp_df = train_test_split(
    df_balanced, 
    test_size=0.3, 
    random_state=42, 
    stratify=df_balanced['target']
)

# Second split: Split temp into 50% test, 50% validation (15% + 15% of total)
test_df, val_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_df['target']
)

print("\nTrain Set (70%):")
print(f"Total: {len(train_df)}")
print(train_df['target'].value_counts().sort_index())

print("\nTest Set (15%):")
print(f"Total: {len(test_df)}")
print(test_df['target'].value_counts().sort_index())

print("\nValidation Set (15%):")
print(f"Total: {len(val_df)}")
print(val_df['target'].value_counts().sort_index())


# STEP 4: Save Files
# Save balanced dataset
df_balanced.to_csv('dota2_toxic_balanced.csv', index=False)
print("\n✓ Saved: dota2_toxic_balanced.csv")

# Save train/val splits
train_df.to_csv('dota2_toxic_train.csv', index=False)
val_df.to_csv('dota2_toxic_validation.csv', index=False)
print("✓ Saved: dota2_toxic_train.csv")
print("✓ Saved: dota2_toxic_validation.csv")


# Step 5: Pie Chart Visualization
# Get counts
before_counts = df['target'].value_counts().sort_index()
after_counts = df_balanced['target'].value_counts().sort_index()

# Define colors and labels
colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
labels = ['Non-toxic [0]', 'Toxic [1]', 'Highly Toxic [2]']

# Create pie charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Toxicity Classification: Before vs After Balancing', fontsize=16, fontweight='bold')

# Pie chart 1: Before
wedges1, texts1, autotexts1 = ax1.pie(
    before_counts.values, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%',
    startangle=90, 
    explode=(0.05, 0.05, 0.05),
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)
ax1.set_title(f'Before Balancing\n(Total: {len(df)} samples)', 
              fontsize=13, fontweight='bold', pad=20)

for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# Add legend with counts
legend_labels1 = [f'{labels[i]}: {before_counts.values[i]:,}' for i in range(len(labels))]
ax1.legend(legend_labels1, loc='upper left', bbox_to_anchor=(0, 0, 0.1, 1), fontsize=10)

# Pie chart 2: After
wedges2, texts2, autotexts2 = ax2.pie(
    after_counts.values, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%',
    startangle=90, 
    explode=(0.05, 0.05, 0.05),
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)
ax2.set_title(f'After Balancing\n(Total: {len(df_balanced)} samples)', 
              fontsize=13, fontweight='bold', pad=20)

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# Add legend with counts
legend_labels2 = [f'{labels[i]}: {after_counts.values[i]:,}' for i in range(len(labels))]
ax2.legend(legend_labels2, loc='upper left', bbox_to_anchor=(0, 0, 0.1, 1), fontsize=10)

plt.tight_layout()
plt.savefig('dota2_pie_charts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dota2_pie_charts.png")

print("\nReady for DeBERTa training!")