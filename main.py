# type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from agents import Agent, Runner, set_tracing_disabled
from config import model
import httpx
import re

set_tracing_disabled(True)

# Request and Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class SimpleRAG:
    """Lightweight RAG system without heavy dependencies"""
    
    def __init__(self, url: str):
        self.url = url
        self.content = ""
        
    async def fetch_content(self) -> bool:
        """Fetch content from website"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.url)
                response.raise_for_status()
                
                # Simple HTML cleaning
                text = response.text
                # Remove script and style tags
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)
                
                self.content = text.strip()
                return True
                
        except Exception as e:
            print(f"Error fetching content: {e}")
            return False
    
    def get_relevant_text(self, query: str, max_chars: int = 2000) -> str:
        """Simple keyword-based content extraction"""
        if not self.content:
            return ""
        
        # Extract keywords from query
        keywords = query.lower().split()
        keywords = [k for k in keywords if len(k) > 3]  # Filter short words
        
        if not keywords:
            return self.content[:max_chars]
        
        # Find sections containing keywords
        sentences = self.content.split('.')
        scored_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Sort by score and get top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        relevant = '. '.join([s[1] for s in scored_sentences[:5]])
        
        return relevant[:max_chars] if relevant else self.content[:max_chars]

# Global instances
rag_system = None
chat_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown handler"""
    global rag_system, chat_agent
    
    print("🚀 Starting up...")
    
    # Initialize simple RAG
    rag_system = SimpleRAG("https://phys-five.vercel.app/")
    await rag_system.fetch_content()
    print("✓ Content loaded")
    
    # Initialize agent
    chat_agent = Agent(
        name="RAG Chat Agent",
        instructions="""You are a helpful AI assistant specializing in Physical AI and Humanoid Robotics.
        
Use the provided context from the textbook when relevant. Be concise and accurate.
Topics: ROS 2, robotics simulation, URDF, Gazebo, NVIDIA Isaac Sim, VLA models, and humanoid robots.""",
        model=model,
    )
    
    print("✅ Ready!")
    yield
    print("👋 Shutting down...")

# Initialize FastAPI
app = FastAPI(title="RAG Chat API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response"""
    if not rag_system or not chat_agent:
        raise HTTPException(status_code=503, detail="System not ready")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Get relevant context
        context = rag_system.get_relevant_text(request.message)
        
        # Build prompt
        if context:
            enhanced_input = f"""Context from the textbook:

{context}

---

Question: {request.message}

Answer based on the context when relevant."""
        else:
            enhanced_input = request.message
        
        # Get AI response
        final_answer = await Runner.run(chat_agent, enhanced_input)
        
        return ChatResponse(response=final_answer.final_output)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chat API is running",
        "usage": "POST /chat with JSON body: {\"message\": \"your question\"}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
