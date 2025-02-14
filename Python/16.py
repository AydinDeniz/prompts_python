
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(label)

# Load data
texts = ["I love this!", "This is bad.", "Amazing experience.", "Terrible service."]
labels = [1, 0, 1, 0]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
model.train()
for epoch in range(3):
    for batch in train_loader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds))

# Visualize attention weights
def visualize_attention(text, tokenizer, model):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1][0].mean(dim=0).cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    sns.heatmap(attention[:len(tokens), :len(tokens)], xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title("Attention Heatmap")
    plt.show()

visualize_attention("I love this!", tokenizer, model)
