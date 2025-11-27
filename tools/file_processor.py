import os

def save_uploaded_file(uploaded_file, directory="data/temp_uploads"):
    """Saves the uploaded Streamlit file object to a local directory."""
    
    # Ensure the upload directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the full file path
    file_path = os.path.join(directory, uploaded_file.name)
    
    # Write the file content to the local disk
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None