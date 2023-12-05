import streamlit as st
import joblib

# Function to load the trained model
def load_model():
    return joblib.load('spam_ham_model.joblib')

# Function to make predictions
def predict_spam_ham(message, model):
    return model.predict([message])[0]

def main():
    st.title("SMS Spam/Ham Classifier")

    # Load the model
    model = load_model()

    # Get user input
    user_input = st.text_area("Enter the SMS message:")

    # Make predictions when the user clicks the button
    if st.button("Predict"):
        if user_input:
            prediction = predict_spam_ham(user_input, model)
            st.success(f"The SMS is {prediction}")
        else:
            st.warning("Please enter a message.")

if __name__ == "__main__":
    main()
