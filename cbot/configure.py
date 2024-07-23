import google.generativeai as genai

import os

def configure():
  os.getenv("GOOGLE_API_KEY")
  genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))