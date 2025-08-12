import os
import google.generativeai as genai

a = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
b = os.getenv("GEMINI_API_KEY")

print(a)
print(b)
