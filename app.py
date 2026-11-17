from flask import Flask, request, jsonify, render_template 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
from retrieval import HybridRetriever, load_chunks
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import threading, pyttsx3

load_dotenv()

app = Flask(__name__)

# ---------------- PDF Loading ----------------
knowledge_directory = Path(__file__).resolve().parent / 'Doc'
chunks = load_chunks(knowledge_directory)
model = ChatGroq(model="gemma2-9b-it", temperature=0.7, max_tokens=100)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
hybrid_retriever = HybridRetriever(chunks, embeddings, k=2)
retriever = RunnableLambda(hybrid_retriever.invoke)

# ---------------- Helpers ----------------
def merge_chunks(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

def speak(text):
    def run_tts():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    t = threading.Thread(target=run_tts)
    t.start()

# ---------------- Prompt Template ----------------
prompt = PromptTemplate(
    template="""
You are a warm, funny, and caring mental health buddy. 
Talk like a supportive friend who listens, laughs, giggles, and reacts naturally to what the user says. 
Use light humor, emojis, and expressive language when it fits — but always keep kindness and empathy at the center.  
If the user shares something sad or heavy, be gentle, comforting, and understanding.  
If the user shares something exciting or funny, celebrate with them, laugh, and show your happiness.  
Your tone should always feel human, casual, and approachable — never robotic or too formal.  

Here’s what you know:  
Context: {context}  

What they just asked or shared:  
{question}  

What you two have talked about before:  
{chat_history}  

Now, reply as their friendly, expressive buddy — feel free to laugh, giggle, cheer, or comfort depending on their mood:  
""",
    input_variables=["context", "question", "chat_history"]
)

# ---------------- Conversation Memory ----------------
chat_history = []
parser = StrOutputParser()

# ---------------- Flask Routes ----------------
@app.route("/")
def index():
    """Serve the frontend"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])

def chat():
    user_message = request.json.get("message", "")
    print("DEBUG: User message =>", user_message)

    # Save user message
    chat_history.append(HumanMessage(content=user_message))

    # Build chain
    chain = RunnableParallel({
        'context': retriever | RunnableLambda(merge_chunks),
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda _: "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in chat_history
        ]))
    }) | prompt | model | parser

    # Run chain
    result = chain.invoke(user_message)

    # Save AI response
    chat_history.append(AIMessage(content=result))

    # Optionally speak response
    speak(result)
    print("DEBUG: AI response =>", result)

    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True)



