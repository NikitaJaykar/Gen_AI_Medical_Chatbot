from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json

load_dotenv()

# Load patient data
with open("Patient_data.json", "r") as file:
    patient_data = json.load(file)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

# 👉 Step 1: Take user query
user_query = input("Enter your medical query: ")

# 👉 Step 2: Convert patient data to string (important for prompt)
patient_info = json.dumps(patient_data, indent=2)

# 👉 Step 3: Create prompt template
prompt = f"""
You are an intelligent medical assistant.

Below is the patient data:
{patient_info}

User question:
{user_query}

Instructions:
- Analyze patient condition, vitals, lab values, and medications.
- Give personalized medical advice.
- Suggest possible precautions, lifestyle changes, or risks.
- Keep answer simple and clear.
- Do NOT give dangerous or unsafe advice.

Answer:
"""
print(prompt)
# 👉 Step 4: Call LLM
result = llm.invoke(prompt)

# 👉 Step 5: Print response
print("\n🩺 Medical Assistant Response:\n")
print(result.content)