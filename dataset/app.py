import streamlit as st

# Assume you have three models already trained and available
# Replace these with your actual model loading logic
def load_model1():
    return lambda text: f"Model 1 response to '{text}'"

def load_model2():
    return lambda text: f"Model 2 response to '{text}'"

def load_model3():
    return lambda text: f"Model 3 response to '{text}'"

# Load models into a dictionary for easy access
models = {
    "Model 1": load_model1(),
    "Model 2": load_model2(),
    "Model 3": load_model3(),
}

# Title of the app
st.title("Text Generation using Multiple Models")

# Take user input
user_input = st.text_input("Enter your text:")

# Model selection dropdown
model_name = st.selectbox("Select a model to generate text:", models.keys())

# Button to generate text
if st.button("Generate Text"):
    if user_input:
        # Get the selected model and generate text
        selected_model = models[model_name]
        generated_text = selected_model(user_input)
        
        # Display the generated text
        st.write(f"**Generated Text:** {generated_text}")
    else:
        st.warning("Please enter some text before generating.")

