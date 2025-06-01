import streamlit as st
from planner import Planner
from plannerAddData import DataUpload
from memory.memory import Memory
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pymongo import MongoClient
from dotenv import load_dotenv
import tempfile
from datetime import datetime
import shutil

# Constants
METADATA_FILE = "./memory/memory_metadata.json"
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "mongodb")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
load_dotenv()

# Authentication (existing code)
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def authenticate(auth_code):
    correct_code = os.getenv("AUTH_CODE", "1234")
    if auth_code == correct_code:
        st.session_state["authenticated"] = True
    else:
        st.error("\ud83d\udea8 Incorrect authentication code")

def login_page():
    st.title("🔐 Login")
    auth_code = st.text_input("Enter Code", type="password")
    st.button("Login", on_click=authenticate, args=(auth_code,))

class AppMain:
    def __init__(self):
        with open("session_id.txt", "r") as file:
            self.session_id_app = file.read().split(",")
        with open("app_selected.txt", "r") as file:
            self.selection_apps = file.read().split(",")

        if "messages" not in st.session_state:
            st.session_state.messages = []

    def data_upload_page(self):
        st.header("📤 Literature Upload")

        # Allow the user to upload multiple .txt files
        uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)
        new_app = st.text_input("Create new application (Thyroid tumor is already added)")
        
        # Add new app to selection apps if not already present
        if new_app not in self.selection_apps and new_app:
            self.selection_apps.append(new_app)

        # Store applications list to file
            with open("app_selected.txt", "w") as file:
                str_app = ",".join(self.selection_apps)
                str_app = str_app.replace(",,", ",")
                file.write(str_app)

        # Select an application from the list
        self.app_selected = st.selectbox("Select Application (Thyorid for Thyroid cancer)", self.selection_apps)

        # Button to initiate data upload
        if st.button("Upload Data"):
            # Check if files were uploaded
            if not uploaded_files:
                st.warning("Please upload one or more .txt or .pdf files!")
                return

            # Create a folder in the current directory to store the uploaded files
            upload_folder = "uploaded_files"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            try:
                # Save uploaded files to the upload folder
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(upload_folder, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                # Get the absolute path for the upload folder
                full_upload_folder_path = os.path.abspath(upload_folder)
                # Pass the upload folder path to the planner for further processing
                try:
                    with st.spinner("⏳ Processing files..."):
                        # Pass the folder path containing the uploaded files
                        #planner = Planner()  # Assuming your Planner class handles folder processing
                        data=DataUpload()
                        print("new application:", self.app_selected, full_upload_folder_path)
                        data.sequence(dataPath=full_upload_folder_path , new_app=self.app_selected)  # Provide the folder path instead of individual files
                    st.success("✅ Data uploaded and processed successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"🚨 Error during processing: {str(e)}")
                    return

            except Exception as e:
                st.error(f"🚨 Error during file upload: {str(e)}")
                return
            finally:
                # Delete the upload folder after processing
                try:
                    # Delete the folder and all its contents
                    shutil.rmtree(upload_folder)
                    st.info(f"Temporary folder {upload_folder} has been deleted.")
                except Exception as e:
                    st.error(f"🚨 Error deleting folder: {str(e)}")

    def process_query_page(self):
        st.header("💬 Chat/Actions")
        application_name = st.selectbox("Application name (Thyorid for Thyroid cancer)", self.selection_apps)
        pdf_file = st.file_uploader("Patient context like pathology or radiology report", type=["pdf"])
        output_format = st.selectbox("Format you would like to see the output", ["Text", "JSON", "Table", "Python Code"])

        if st.button("Start a new session (Each patient can have own session)"):
            curr_date_time = datetime.now()
            date_time_ = f"{application_name}_{str(curr_date_time)}".replace(" ", "_")
            self.session_id_app.append(date_time_)
            with open("session_id.txt", "w") as file:
                file.write(",".join(self.session_id_app))

        session_id = st.selectbox("Select a session", self.session_id_app)
        query = st.chat_input("Enter your question")

        memory = Memory(application_name)

        def trigger_feedback(index, feedback_value, query, answer, sessionid):
            memory.update_feedback(query, answer, sessionid, feedback_value)
            st.session_state[f"feedback_done_{index}"] = True

        # If user sent a new message
        if query:
            try:
                pdf_path = ""
                if pdf_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                        tmp_pdf.write(pdf_file.getvalue())
                        pdf_path = tmp_pdf.name

                planner_input = {
                    "query": query,
                    "context": pdf_path,
                    "output_expectation": output_format,
                    "sessionid": session_id
                }

                planner = Planner()
                result = planner.sequence(planner_input, application_name)

                memory.add_conversation(query=query, answer=result, sessionid=session_id)

                st.session_state.messages.append({
                    "role": "user", 
                    "content": query,
                    "sessionid": session_id
                })
                st.session_state.messages.append({
                    "role": "bot", 
                    "content": result,
                    "sessionid": session_id,
                    "query": query,
                    "answer": result
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Always display all messages
        for i, msg in enumerate(st.session_state.messages):
            if msg["sessionid"] != session_id:
                continue  # Skip messages not related to this session

            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    st.chat_message("assistant").write(msg["content"])
                if not st.session_state.get(f"feedback_done_{i}"):
                    with col2:
                        st.button("👍", key=f"positive_{i}",
                                on_click=trigger_feedback,
                                args=(i, "positive", msg["query"], msg["answer"], msg["sessionid"]))
                    with col3:
                        st.button("👎", key=f"negative_{i}",
                                on_click=trigger_feedback,
                                args=(i, "negative", msg["query"], msg["answer"], msg["sessionid"]))
                else:
                    st.markdown("✅ Your feedback is recorded")

    def feedback_page(self):
        st.header("📝 Feedback on chat/actions")

        application_name = st.selectbox("Select Application (Thyroid for Thyroid cancer)", self.selection_apps)
        session_id = st.selectbox("Select Session (Each patient can have own session)", self.session_id_app)
        feedback_text = st.text_area("Enter your feedback")

        if st.button("Submit Feedback"):
            if not session_id or not feedback_text or not application_name:
                st.warning("Please fill all fields")
                return

            try:
                if DATABASE_TYPE == "mongodb":
                    collection = self.get_feedback_storage(application_name)
                    if collection is None:
                        return

                    result = collection.update_one(
                        {"sessionid": session_id},
                        {"$set": {
                            "feedback": feedback_text,
                            "timestamp": datetime.now().isoformat()
                        }},
                        upsert=True
                    )

                    if result.modified_count > 0 or result.upserted_id:
                        st.success("Feedback submitted to MongoDB!")

                else:
                    feedback_store = self.get_feedback_storage(application_name)
                    if feedback_store is None:
                        return

                    feedback_doc = Document(
                        page_content=feedback_text,
                        metadata={
                            "sessionid": session_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

                    feedback_store.add_documents([feedback_doc])

                    feedback_path = os.path.join(
                        os.getenv("FAISS_STORAGE_PATH", "./faiss_store"),
                        application_name,
                        "feedback"
                    )
                    feedback_store.save_local(feedback_path)

                    st.success("Feedback stored in FAISS!")

            except Exception as e:
                st.error(f"Error processing feedback: {str(e)}")

def main():
    if not st.session_state["authenticated"]:
        login_page()
    else:
        st.title("Thyroid Cancer Agent")
        st.sidebar.title("Navigation Options")
        page = st.sidebar.radio("", 
                               ["Literature Upload", "Chat/Actions", "Feedback"],
                               index=1)

        app = AppMain()
        if page == "Literature Upload":
            app.data_upload_page()
        elif page == "Chat/Actions":
            app.process_query_page()
        elif page == "Feedback":
            app.feedback_page()

if __name__ == '__main__':
    main()
