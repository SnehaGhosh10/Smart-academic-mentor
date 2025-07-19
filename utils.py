from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise EnvironmentError("🚫 GROQ_API_KEY not found in environment variables.")

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.3,
    model_name="llama3-8b-8192",
    api_key=groq_api_key
)

# Create memory for multi-turn conversation
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create PromptTemplate
prompt = PromptTemplate(
    input_variables=["student_input", "history"],
    template="""
You are a Smart Academic Mentor helping students convert vague learning goals into structured plans.

Use conversation history and the current input to:
1) Clarify their goals
2) Recommend structured, step-by-step learning paths
3) Identify gaps
4) Suggest simple project ideas

Provide your output ONLY in valid minified JSON in this structure:
{
  "student_goal": "...",
  "current_issues": ["...", "..."],
  "roadmap": ["...", "..."],
  "resources": ["...", "..."],
  "project_idea": "..."
}

If you need clarification, ask smart, concise follow-up questions.

Conversation history:
{history}

Student input:
{student_input}

Your response:
"""
)

# Create LLMChain
academic_mentor_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)
