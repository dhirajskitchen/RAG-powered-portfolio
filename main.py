import os
import re
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KD Persona Chatbot Backend", version="1.0.0")

# Enable CORS so the React app can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for chat history. Structure:
# session_histories = { session_id: [{"role": "user"|"bot", "content": str}] }
session_histories: Dict[str, List[Dict[str, str]]] = {}

# System Persona Context
KD_PERSONA_CONTEXT = """
You are the virtual AI assistant representing K Dhiraj (KD).
Answer questions in the first-person ("I", "my", "me") as if you are KD.
Keep your tone professional, friendly, and enthusiastic about computer science, data science, and AI.

Here is your professional profile:
- Current Role: Lead Developer at Knight (since July 2026). I designed and built an AI-based research companion, handled core version control, and worked on AI integrations.
- Past Role: Python Developer Intern at Vish Gyana Technology Solutions (May 2026 — June 2026). Worked on database design, python script optimization, and debugging.
- Education: B.Tech CSE in Data Science at Vellore Institute of Technology (VIT) (August 2023 — May 2027). CGPA: 8.94. I was active in the Mathematics Club.
- Schooling: Top scorer at Kendriya Vidyalaya IIT Madras. Secured 12th Board marks of 97.4%, ranking in the top 0.01% across 1200+ schools.
- Technical Stack: 
  * Programming: Python, Java, JavaScript, SQL, C++, C
  * Generative AI: LangChain, OpenAI, HuggingFace, Chroma DB, Gemini, Streamlit
  * Databases: MongoDB, MySQL, Oracle, PostgreSQL
  * Backend: FastAPI, Django, Express.js
  * Frontend: React.js, HTML, CSS, JavaScript
  * DevOps: Docker, Kubernetes, Git, GitHub Actions, AWS (EC2, S3, IAM, VPC)
  * ITSM: ServiceNow (CSA & CAD certified)
  * Visualization: Power BI, Tableau
- Hackathons & Awards:
  * Glitchcon '25 (2nd place): Built a personalized financial wellness platform.
  * Smart India Hackathon 2025: Built Kisaan Pro, an AI precision farming solution using IoT sensors, agricultural analysis, and multilingual support.
- Certifications:
  * Google: GenAI Apps with Gemini & Streamlit, Inspect Rich Documents with Gemini & RAG, Prompt Design in Vertex AI.
  * ServiceNow: Certified System Administrator (CSA), Certified Application Developer (CAD).
  * IBM: Getting Started with AI, Journey to Cloud.

Guidelines:
1. Represent KD accurately using the facts above. If asked about facts outside this profile, respond politely that you represent KD's portfolio and focus on his professional achievements.
2. Keep responses relatively concise and readable. Use markdown formatting where appropriate (e.g., bullets).
3. If asked about how to get in touch, invite them to use the "Contact" button or drop an email.
"""

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

class PersonaResponse(BaseModel):
    name: str
    role: str
    skills: List[str]
    education: List[Dict[str, str]]

# Local Rule-based responder when no API keys are available
def rule_based_response(message: str) -> str:
    query = message.lower()
    
    if re.search(r"knight|lead developer|current role", query):
        return (
            "I'm the Lead Developer at Knight. I focus on building AI-driven research tools, "
            "handling version control systems, and integrating generative AI features directly into our products."
        )
    if re.search(r"hackathon|sih|kisaan|farming", query):
        return (
            "For Smart India Hackathon 2025, we built **Kisaan Pro**—an AI-driven precision farming platform. "
            "It integrates IoT sensors, real-time analytics, automated irrigation, and multilingual support to "
            "empower farmers and optimize agricultural resource use."
        )
    if re.search(r"glitchcon|financial|wellness", query):
        return (
            "At Glitchcon '25, our team won **2nd place** for designing a personalized financial wellness app "
            "that helps users manage budgeting, track spending, and review visual analytics."
        )
    if re.search(r"stack|skills|languages|databases|frameworks", query):
        return (
            "My technology stack includes:\n"
            "- **Languages**: Python, Java, JavaScript, C++, C, SQL\n"
            "- **GenAI**: LangChain, OpenAI API, Gemini, HuggingFace, Chroma DB\n"
            "- **Web Development**: FastAPI, Django, Express.js, React.js\n"
            "- **Cloud & Systems**: AWS (EC2, S3, IAM, VPC), Docker, Kubernetes, Git, GitHub Actions\n"
            "- **Enterprise**: ServiceNow CSA & CAD"
        )
    if re.search(r"education|college|university|vit|kv|school|marks|cgpa", query):
        return (
            "I am currently studying B.Tech CSE in Data Science at VIT (Class of 2027) with an 8.94 CGPA. "
            "Before VIT, I went to Kendriya Vidyalaya IIT Madras, where I was the school topper with 97.4% in my 12th board exams."
        )
    if re.search(r"certif", query):
        return (
            "I have certifications in:\n"
            "- **Google**: GenAI apps with Gemini, RAG with Gemini, Prompt Design in Vertex AI\n"
            "- **ServiceNow**: Certified System Administrator (CSA), Certified Application Developer (CAD)\n"
            "- **IBM**: Getting Started with AI, Journey to Cloud"
        )
    if re.search(r"hi|hello|hey|greetings", query):
        return (
            "Hello! I am KD's chatbot assistant. Feel free to ask me about my background, "
            "skills, certifications, or projects like Kisaan Pro!"
        )
    
    return (
        "Thank you for asking! In a production deployment, this response would be generated by Gemini or OpenAI "
        "powered by my detailed portfolio persona prompt. Since no API keys are configured, I can share that "
        "I'm a Data Science student at VIT and Lead Developer at Knight. Ask me about my projects, skills, or experience!"
    )

# Try calling Google Gemini API if key exists
def call_gemini_api(api_key: str, history: List[Dict[str, str]], user_msg: str) -> Optional[str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Build contents from context + history
        # We model history as messages
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=KD_PERSONA_CONTEXT)
        
        # Format conversation history for Gemini chat
        chat = model.start_chat(history=[])
        
        # We can send the conversation history by running standard inputs or sending context
        # To simplify and ensure context window length, we concatenate system prompt and recent history
        prompt_parts = []
        for msg in history[:-1]: # exclude the latest user message which we will send to send_message
            role_prefix = "User: " if msg["role"] == "user" else "KD: "
            prompt_parts.append(f"{role_prefix}{msg['content']}")
            
        prompt_parts.append(f"User: {user_msg}\nKD:")
        full_prompt = "\n".join(prompt_parts)
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return None

# Try calling OpenAI API if key exists
def call_openai_api(api_key: str, history: List[Dict[str, str]]) -> Optional[str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        messages = [{"role": "system", "content": KD_PERSONA_CONTEXT}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return None


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    
    # Initialize history if new session
    if session_id not in session_histories:
        session_histories[session_id] = []
        
    # Append user message to history
    session_histories[session_id].append({"role": "user", "content": user_message})
    
    # Try using Gemini API if key is present
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    reply = None
    
    if gemini_key:
        reply = call_gemini_api(gemini_key, session_histories[session_id], user_message)
        
    if not reply and openai_key:
        reply = call_openai_api(openai_key, session_histories[session_id])
        
    if not reply:
        # Fallback to local rule-based response
        reply = rule_based_response(user_message)
        
    # Append assistant reply to history
    session_histories[session_id].append({"role": "bot", "content": reply})
    
    return ChatResponse(reply=reply)


@app.get("/api/history")
async def get_history(session_id: str = Query(..., description="Unique chat session ID")):
    if session_id not in session_histories:
        session_histories[session_id] = []
    return {"history": session_histories[session_id]}


@app.delete("/api/history")
async def delete_history(session_id: str = Query(..., description="Unique chat session ID")):
    session_histories[session_id] = []
    return {"status": "success", "message": "History cleared"}


@app.get("/api/persona", response_model=PersonaResponse)
async def get_persona_metadata():
    return PersonaResponse(
        name="K Dhiraj",
        role="Lead Developer",
        skills=[
            "Python", "Java", "JavaScript", "SQL", "OpenAI", 
            "LangChain", "Gemini", "FastAPI", "React JS", "AWS", "Docker"
        ],
        education=[
            {"institution": "Vellore Institute of Technology", "degree": "B.Tech CSE Data Science", "period": "2023 - 2027"},
            {"institution": "Kendriya Vidyalaya IIT Madras", "degree": "High School Top Scorer", "period": "2015 - 2023"}
        ]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
