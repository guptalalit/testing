from flask import Flask, request, jsonify
# from tasks.task import Task
from planner import Planner
from memory.memory import Memory
import plannerAddData
from utils.google_llm_services import GoogleLLM
import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize memory
memory = Memory()
METADATA_FILE = "./memory/memory_metadata.json"

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as file:
            return json.load(file)
    return []

def save_metadata(data):
    with open(METADATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    data = request.get_json()
    data_path = data.get("data_path")
    
    plannerAddData.Planner().sequence(data_path)
    return jsonify({"Status": "Data Uploaded"})


@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get("query")
    pdf_path = data.get("pdf_path", "")
    output_expectation = data.get("output_format")
    sessionid = data.get("sessionid")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Process the query and PDF path
    # task = Task()
    # result = task.input(query, pdf_path)
    #pdf_text = GoogleLLM().getResults(pdf_path)
    pdf_text=''
    planner_input = {
        "query": query,
        "context": pdf_path,
        "output_expectation": output_expectation
    }
    sessionid = "abcd1234"

    # Pass the corrected query to the planner
    planner = Planner()
    planner_result = planner.sequence({"query": planner_input["query"], "context": planner_input["context"], "output_expectation": planner_input["output_expectation"], "sessionid": sessionid})

    # Store the conversation in memory
    memory.add_conversation(query, planner_result,sessionid=sessionid)
    #memory.save_memory()

    return jsonify({"planner_result": planner_result})

# Add to MongoDB Configuration
FEEDBACK_COLLECTION_NAME = "session_feedback"  # New collection for feedback

@app.route('/add_feedback', methods=['POST'])
def add_feedback():
    load_dotenv()
    data = request.get_json()
    sessionid = data.get("sessionid")
    feedback = data.get("feedback")
    
    if not sessionid or not feedback:
        return jsonify({"error": "sessionid and feedback are required"}), 400
    try:
        # Access feedback collection
        feedback_collection = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))["newdb"][FEEDBACK_COLLECTION_NAME]
        
        # Update or insert feedback for the session
        result = feedback_collection.update_one(
            {"sessionid": sessionid},
            {"$set": {"feedback": feedback}},
            upsert=True
        )
        
        if result.modified_count > 0 or result.upserted_id:
            return jsonify({"message": "Feedback added successfully"})
        return jsonify({"error": "Failed to save feedback"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)



