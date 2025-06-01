import re
import os
import easyocr
import numpy as np
#from google import genai
from pdf2image import convert_from_path
from dotenv import load_dotenv
import google.generativeai as genai



# Get the root directory of the project (AGENTS2)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define the path to the .env file in the root directory
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Load the .env file
load_dotenv(ENV_PATH)

# Ensure environment variables are loaded correctly
api_key = os.getenv('GEMINI_API_KEY')
model_name = "gemini-1.5-flash"

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")

if not model_name:
    raise ValueError("GOOGLE_LLM_MODEL is not set. Please check your .env file.")

# Assign values to os.environ
#s.environ["GOOGLE_API_KEY"] = api_key
#genai.configure(api_key=api_key)
# Ensure GOOGLE_API_KEY is correctly set
genai.configure(api_key=api_key)
genconfig = genai.GenerationConfig(temperature=0)

def getLines(data):
    lines = []
    current_line = []
    previous_y = data[0][0][0][1]

    for item in data:
        word = item[1]
        y_coord = item[0][0][1]

        if abs(y_coord - previous_y) <= 20:  # Adjust the threshold as needed
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            previous_y = y_coord

    if current_line:
        lines.append(" ".join(current_line))

    for i in range(len(lines)):
        lines[i] = re.sub(r'[^A-Za-z0-9 /.-]+', "", lines[i])
        lines[i] = re.sub(r'\s+', " ", lines[i])

    return "\n".join(lines)

class GoogleLLM:
    def getResults(self, fileName):

        outputPath = os.path.abspath(fileName)

        # Ensure GOOGLE_API_KEY is correctly set
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        genconfig = genai.GenerationConfig(temperature=0)

        model = genai.GenerativeModel(model_name=f"models/{model_name}")

        reader = easyocr.Reader(['en'])

        if ".jpg" in fileName or ".bmp" in fileName or ".jpeg" in fileName:
            images = [fileName]
        if ".pdf" in fileName:
            images = convert_from_path(outputPath)
        context_text = ""
        print("images: ", len(images))
        for i in range(len(images)):  
            print(i)
            # try:
            #     if ".pdf" in fileName:
            #         image_path = f'page{i}.jpg'
            #         images[i].save(image_path, 'JPEG')                
            #         # Upload and process using Gemini
            #         sample_file = genai.upload_file(path=image_path)
            #         os.remove(image_path)
            #     else:
            #         sample_file = genai.upload_file(path=images[i])
            #     response = model.generate_content(
            #         ["OCR the image, if image has table or flowchart, put them in json like format to make it readable by LLM", sample_file], 
            #         generation_config=genconfig
            #     )
            #     context_text += response.text
            #     print("Gemini OCR response.text ===> ", response.text)

            # except Exception as e:
            #print("EasyOCR fallback, error:", e)
            imgArray = np.array(images[i])
            result = reader.readtext(imgArray)
            if len(result) == 0:
                continue
            else:
                try:
                    resp_text = getLines(result)
                except:
                    continue
            #print("EasyOCR response ==> ", resp_text)
            context_text += resp_text + " "

        return context_text
