from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json

load_dotenv()
# retrive data from vector database
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "file_storage", "faiss_index")


def medical_chat(query:str):
    # Load same embedding model (VERY IMPORTANT)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load saved FAISS index
    vector_db = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
    )
    
    results = vector_db.similarity_search(query, k=3)


    # Load patient data
    with open("Patient_data.json", "r") as file:
        patient_data = json.load(file)

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview"
    )

    # Step 2: Convert patient data to string (important for prompt)
    patient_info = json.dumps(patient_data, indent=2)
    retriever_data = "\n".join([doc.page_content for doc in results])

    # Step 3: Create prompt template
    prompt = f"""
    You are an intelligent medical assistant.

    Below is the patient data:
    {patient_info}

    Medical Context:
    {retriever_data}

    User question:
    {query}

    Instructions:
    - Analyze patient condition, vitals, lab values, and medications.
    - Give personalized medical advice.
    - Suggest possible precautions, lifestyle changes, or risks.
    - Keep answer simple and clear.
    - Do NOT give dangerous or unsafe advice.

    Answer:
    """
    print(prompt)
    #Step 4: Call LLM
    result = llm.invoke(prompt)

    #Step 5: Print response

    # Extract only text
    data = result.content
    texts = [item["text"] for item in data]

    # Join all text (if multiple entries)
    final_text = "\n\n".join(texts)
    
    return final_text
