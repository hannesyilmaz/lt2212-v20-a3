import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
# Whatever other imports you need

# You can implement classes and helper functions here too.


input_dir = "enron_sample"

def process(file_name):
    file_text = open(file_name).readlines()
    first_line_number = None
    last_line_number = None
    for line in file_text:
        if "X-FileName:" in line:
            first_line_number = file_text.index(line) + 1
        if "Original Message" in line:
            last_line_number = file_text.index(line)
    if not last_line_number:
        last_line_number = len(file_text) + 1
    file_content = " ".join(line.strip() for line in file_text[first_line_number : last_line_number])
    return file_content
    
def read_documents(input_dir):
    complete_files = {}
    author_contents_complete = []
    authors = []
    for folder in os.listdir(input_dir):
        filepath = os.path.join(input_dir,folder)
        if os.path.isdir(filepath):
            folder_text = []
            for file_name in os.listdir(filepath):
                file_content = process(os.path.join(filepath,file_name))
                folder_text.append(file_content)
                author_contents_complete.append(file_content)
                authors.append(folder)
            complete_files[folder] = folder_text
    print(author_contents_complete)
    print(authors)
    return complete_files, author_contents_complete, authors
# read_documents(input_dir)

def build_table(dims, testsize, author_contents_complete, authors):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(author_contents_complete)
#     pca = PCA(n_components=dims)
    svd = TruncatedSVD(n_components=dims)
    X_transformed = svd.fit_transform(X)
    columns = ["dim"+str(i) for i in range(1, dims + 1)]
    table = pd.DataFrame.from_records(X_transformed, columns=columns)
    table['author'] = authors
    testrows = int(testsize * len(table))
    print(testrows)
    table.loc[:testrows, 'data_source'] = "train"
    table.loc[testrows:, 'data_source'] = "test"
    table = shuffle(table)
    print(table.head())
    print(table.tail())
    return table
    
    
build_table(10, 0.2,author_contents_complete, authors)


class Network(nn.Module):
    def __init__(self, input_size, hidden_layer):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_size, hidden_layer)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 1)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

    
def design_nn_model(table, input_size, hidden_layer, batch_size):
    net = Network(input_size, hidden_layer)
    trainloader = torch.utils.data.DataLoader(table, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N_EPOCHS = 5
    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(table) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_func(table[table['data_source'] == "train"])
        valid_loss, valid_acc = test(table[table['data_source'] == "test"])
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))

    # implement everything you need here
    table = pd.read_csv(args.featurefile)
    
    design_nn_model(table)
    
    
