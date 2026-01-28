import pdfplumber
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

PDF = "assets/cv_teste.pdf"
OUTPUT = "output/cv_en.txt"

def extract_text(cv):
    text = ""
    with pdfplumber.open(cv) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()



prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a professional resume translator.

Rules:
- Translate ONLY the text
- Do NOT add, remove or rewrite content
- Keep professional resume tone
- Preserve numbers, dates, technologies and company names

Translate the following text to English:

{text}
"""
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


chain = prompt | llm

def translate_service():
    og_txt = extract_text(PDF)

    response = chain.invoke({"text": og_txt})
    translated_text = response.content

    os.makedirs("output", exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(translated_text)

if __name__ == "__main__":
    translate_service()

