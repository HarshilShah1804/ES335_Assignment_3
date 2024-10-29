import streamlit as st
import torch
import torch.nn as nn
import csv
import re
import os

# Model definition
class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, vocab_size)
        self.activation = torch.relu if activation == "relu" else torch.tanh

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))
        x = self.lin4(x)
        return x

# Token mappings
def load_tokens(file_path):
    stoi = {}
    itos = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            token_id = int(row[1])  # Convert ID to integer
            token = row[0]           # Token remains as a string
            stoi[token] = token_id    # Map token to ID
            itos[token_id] = token     # Map ID to token

    return stoi, itos

# Load token mappings
path = 'dataset'
stoi, itos = load_tokens(os.path.join(path, 'tokens_holmes.csv'))

# Function to load models from .pt files with CPU mapping
def load_model(block_size, vocab_size, emb_dim, hidden_size, activation, model_path):
    model_path = os.path.join(path, model_path)
    model = NextChar(block_size, vocab_size, emb_dim, hidden_size, activation)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dict
    model.eval()  # Set the model to evaluation mode
    return model

# Load your actual models from .pt files
models = {
    "holmes_5_64_relu": load_model(5, len(itos), 64, 1024, "relu", "holmes_5_64_relu.pt"),
    "holmes_5_64_tanh": load_model(5, len(itos), 64, 1024, "tanh", "holmes_5_64_tanh.pt"),
    "holmes_5_128_tanh": load_model(5, len(itos), 128, 1024, "tanh", "holmes_5_128_tanh.pt"),
    "holmes_10_64_relu": load_model(10, len(itos), 64, 1024, "relu", "holmes_10_64_relu.pt"),
    "holmes_10_64_tanh": load_model(10, len(itos), 64, 1024, "tanh", "holmes_10_64_tanh.pt"),
    "holmes_10_128_relu": load_model(10, len(itos), 128, 1024, "relu", "holmes_10_128_relu.pt"),
    # "holmes_10_128_tanh": load_model(10, len(itos), 128, 1024, "tanh", "holmes_10_128_tanh.pt"),
    "holmes_15_64_relu": load_model(15, len(itos), 64, 1024, "relu", "holmes_15_64_relu.pt"),
    "holmes_15_64_tanh": load_model(15, len(itos), 64, 1024, "tanh", "holmes_15_64_tanh.pt"),
    "holmes_15_128_relu": load_model(15, len(itos), 128, 1024, "relu", "holmes_15_128_relu.pt"),
}

# Function to tokenize user input
def tokenize_code(data):
    pattern = r"(\b\w+\b|\d+|[^\w\s]|\s+)"
    tokens = re.findall(pattern, data)
    return tokens

block_size_array = [5, 5, 5, 10, 10, 10, 15, 15, 15] 
# Function to generate text from the selected model
def generate_text(prompt, model, itos, stoi, block_size, max_len=10):
    context = [0] * block_size
    generated_text = prompt

    for ch in tokenize_code(prompt):
        if ch not in stoi:
            context = context[1:] + [1]  # Assume 1 is the token for unknown characters
            continue
        ix = stoi[ch]
        context = context[1:] + [ix]

    for i in range(max_len):
        x = torch.tensor(context).view(1, -1)
        with torch.no_grad():  # Disable gradient calculation
            y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '' and generated_text != '':
            break
        generated_text += ch
        context = context[1:] + [ix]
    
    return generated_text

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for model selection
with st.sidebar:
    st.title("Choose Your Model")
    selected_model_name = st.selectbox("Select a model:", models.keys())
    selected_model = models[selected_model_name]
    
    # st.title("History")
    # if st.button("Show History"):
    #     for entry in st.session_state.history:
    #         st.write(entry)

# Main chat interface
st.title("Chat-like Text Generation App")

# # Display chat history
# for message in st.session_state.messages:
#     role, content = message
#     if role == "user":
#         st.markdown(f"**You:** {content}")
#     else:
#         st.markdown(f"**{selected_model_name}:** {content}")

# Input field for user message
user_input = st.text_input("Your message:", key="user_input")

# Handle message submission
if st.button("Generate Response"):  
    if user_input:
        # Store the user's message
        st.session_state.messages.append(("user", user_input))
        
        # Determine the block_size based on the selected model
        block_size_index = list(models.keys()).index(selected_model_name)
        block_size = block_size_array[block_size_index]
        
        # Generate response using the selected model
        response = generate_text(user_input, selected_model, itos, stoi, block_size, 25)  # Use the selected model
        st.session_state.messages.append((selected_model_name, str(response)))  # Append the generated response

        # Clear the input field
        # st.session_state.user_input = ""

        # # Rerun the app to display updated chat
        # st.experimental_rerun()
        st.write(response)
    else:
        st.warning("Please enter a message.")


