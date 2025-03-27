import streamlit as st
from planner import Planner

from plannerAddData import DataUpload
from memory.memory import Memory
from utils.google_llm_services import GoogleLLM
import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import tempfile
from datetime import datetime
import shutil  # For removing directories
# Initialize core components

METADATA_FILE = "./memory/memory_metadata.json"
FEEDBACK_COLLECTION_NAME = "session_feedback"
load_dotenv()


class AppMain():
    def __init__(self):
        #self.app_selected = "Thyroid"
        #self.session_id = "default"
        with open("session_id.txt", "r") as file:
            self.session_id_app = file.read().split(",")
        with open("app_selected.txt", "r") as file:
            self.selection_apps = file.read().split(",")

    def get_feedback_collection(self):
        client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
        return client["lungtumor_db"][FEEDBACK_COLLECTION_NAME]



    def data_upload_page(self):
        st.header("📤 Data Upload")

        # Allow the user to upload multiple .txt files
        uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)
        new_app = st.text_input("Create new application")
        
        # Add new app to selection apps if not already present
        if new_app not in self.selection_apps and new_app:
            self.selection_apps.append(new_app)

        # Store applications list to file
            with open("app_selected.txt", "w") as file:
                str_app = ",".join(self.selection_apps)
                str_app = str_app.replace(",,", ",")
                file.write(str_app)

        # Select an application from the list
        self.app_selected = st.selectbox("Select Application", self.selection_apps)

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
                        print("new application:", self.app_selected)
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
        st.header("🔍 Process Query")
        query = st.text_input("Enter your query")
        pdf_file = st.file_uploader("Upload PDF context", type=["pdf"])
        output_format = st.selectbox("Select output format", ["JSON", "Text", "Table"])
        application_name=st.text_input("Provide the application name")
        # session_id = st.text_input("Session ID")
        # Check if the application name exists in the MongoDB database
        if application_name:
            # Initialize MongoDB client and check if the database exists
            try:
                client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
                # List all database names
                existing_databases = client.list_database_names()

                # Check if the application_name exists in the list of databases
                if application_name not in existing_databases:
                    st.error(f"Invalid application name: '{application_name}'. Please provide a correct application name.")
                    return  # Stop execution if the application name is invalid
            except Exception as e:
                st.error(f"Error connecting to MongoDB: {str(e)}")
                return
        
        if st.button("Add new session"):
            
            curr_date_time = datetime.now()
            date_time_ = (application_name + "_" + str(curr_date_time)).replace(" ", "_")
            self.session_id_app.append(date_time_)

            with open("session_id.txt", "w") as file:
                str_app = ",".join(self.session_id_app)
                str_app = str_app.replace(",,", ",").replace(", ", ",").replace(",/n", ",")
                file.write(str_app)                    

        self.session_id = st.selectbox("Select session", self.session_id_app)

        if st.button("Process Query"):
            if not query:
                st.error("Please enter a query")
            else:
                pdf_path = ""
                if pdf_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                        tmp_pdf.write(pdf_file.getvalue())
                        pdf_path = tmp_pdf.name

                try:
                    planner_input = {
                        "query": query,
                        "context": pdf_path,
                        "output_expectation": output_format,
                        "sessionid": self.session_id
                    }
                    
                    planner = Planner()
                    result = planner.sequence(planner_input,application_name)
                    
                    memory = Memory(application_name)
                    memory.add_conversation(query, result, sessionid=self.session_id)
                    
                    st.subheader("Result")
                    st.write(str(result))
                    # st.json(result) if output_format == "JSON" else st.write(str(result))
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

        

    def get_feedback_collection(self, application_name):
        """
        Get the feedback collection from MongoDB. If it doesn't exist, create it.
        """
        try:
            client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
            db = client[application_name]
            
            # Check if feedback collection exists, if not create it
            if "feedback" not in db.list_collection_names():
                # Create feedback collection if it doesn't exist
                db.create_collection("feedback")
                st.info("Created 'feedback' collection in the application.")
            
            # Return the feedback collection
            collection = db.get_collection("feedback")
            return collection
        except Exception as e:
            st.error(f"Error fetching feedback collection: {str(e)}")
            return None

    def feedback_page(self):
        st.header("📝 Feedback")

        # Provide both session and application name input
        application_name = st.selectbox("Select application", self.selection_apps)
        session_id = st.selectbox("Select session", self.session_id_app)
        
        # Provide feedback text input
        feedback_text = st.text_area("Your Feedback")

        if st.button("Submit Feedback"):
            # Check if all fields are filled
            if not session_id or not feedback_text or not application_name:
                st.warning("Please fill all the fields (Application, Session ID, and Feedback)")
                return

            try:
                # Step 1: Check if the application (database) exists
                client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
                existing_databases = client.list_database_names()

                # Check if the selected application (database) exists
                if application_name not in existing_databases:
                    st.error(f"No application found with the name: '{application_name}'. Please provide a correct application name.")
                    return  # Stop further execution if the application is not found

                # Step 2: Get or create the 'feedback' collection
                collection = self.get_feedback_collection(application_name)
                if collection is None:
                    st.error(f"Feedback collection could not be fetched or created for application '{application_name}'.")
                    return  # Stop further execution if the collection couldn't be fetched/created

                
                # Step 3: Insert or update the feedback document in the MongoDB collection
                result = collection.update_one(
                    {"sessionid": session_id},
                    {"$set": {"feedback": feedback_text}},
                    upsert=True
                )

                # Check if the operation was successful
                if result.modified_count > 0 or result.upserted_id:
                    st.success("Feedback submitted successfully!")
                else:
                    st.error("Failed to submit feedback")

            except Exception as e:
                st.error(f"Error processing feedback: {str(e)}")

def main():
    st.title("AI Assistant Platform")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data Upload", "Process Query", "Feedback"],
        index=1  # Default to Process Query
    )

    obj = AppMain()

    # Display the selected page
    if page == "Data Upload":
        obj.data_upload_page()
    elif page == "Process Query":
        obj.process_query_page()
    elif page == "Feedback":
        obj.feedback_page()

if __name__ == '__main__':
    main()