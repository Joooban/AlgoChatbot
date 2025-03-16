import Backend as BE

def main():
    # Streamlit page setup
    BE.st.set_page_config(page_title="FactUBot", page_icon="🤖")
    BE.st.title("FactUBot: AI-Powered Fake News Detector 🤖")

    dataset_path = "WELFake_Dataset.csv"  # Path to dataset
    BE.handle_conversation(dataset_path)  # Call the backend function

# Run the application
if __name__ == "__main__":
    main()
