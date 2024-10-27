import streamlit as st

# Mock functions simulating your models
def load_model1():
    return lambda text: f"Model 1 response to '{text}'"

def load_model2():
    return lambda text: f"Model 2 response to '{text}'"

def load_model3():
    return lambda text: f"Model 3 response to '{text}'"

# Load models into a dictionary for easy selection
models = {
    "Model 1": load_model1(),
    "Model 2": load_model2(),
    "Model 3": load_model3(),
}

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for model selection
with st.sidebar:
    st.title("Choose Your Model")
    selected_model_name = st.selectbox("Select a model:", models.keys())
    selected_model = models[selected_model_name]

# Main chat interface
st.title("Enter a message")

# Display chat history
for message in st.session_state.messages:
    role, content = message
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**{selected_model_name}:** {content}")

# Input field for user message
user_input = st.text_input("Your message:", key="user_input")

# Handle message submission
if st.button("Send"):
    if user_input:
        # Store the user's message
        st.session_state.messages.append(("user", user_input))
        
        # Generate response using the selected model
        response = selected_model(user_input)
        st.session_state.messages.append((selected_model_name, response))

        # Clear the input field
        st.session_state.user_input = ""

        # Rerun the app to display updated chat
        st.experimental_rerun()
    else:
        st.warning("Please enter a message.")

