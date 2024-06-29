import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words
from nltk.corpus import stopwords

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

def main():
    with open('intents1.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    stop_words = set(stopwords.words('english'))
    print("Stop_words", stop_words)

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        print(f"Processing intent with tag: {tag}")
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            # Filter out stop words before stemming
            w = [word for word in w if word.lower() not in stop_words]
            # print("Tokenized Pattern:", w)
            all_words.extend(w)
            xy.append((w, tag))
            # print("XY:", xy)

    ignore_words = ['?', '!', '.', ',', '--', '-', '(', ')', '/', '`']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    print("All words after removing ignore words:", all_words)
    print()

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)
        
        # Print the entire bag of words in human-readable format
        bag_readable = [(all_words[idx], val) for idx, val in enumerate(bag)]
        print(f"Bag of words for pattern_sentence '{pattern_sentence}': {bag_readable}")
        print()
        print(f"Length of bag of words: {len(bag)}")
        print()

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("Data prepared for training")

    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.005
    num_epochs = 500

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")

    for epoch in range(num_epochs):
        for i, (words, labels) in enumerate(train_loader):
            words = words.to(device).float()
            labels = labels.to(device).long()

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'Training complete. File saved to {FILE}')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # This line is necessary in Windows
    main()