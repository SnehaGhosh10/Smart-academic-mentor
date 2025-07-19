from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
import re

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise EnvironmentError("🚫 GROQ_API_KEY not found in environment variables.")

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    api_key=groq_api_key
)

# Prompt template
template = """
You are a Smart Academic Mentor AI.

Task:
Given a student's learning goal, confusion, or vague request, you will:
1. Clarify and identify their goal.
2. Identify current issues they face.
3. Recommend a structured, step-by-step roadmap in clear weekly steps.
4. Suggest practical resources (courses, playlists, sheets).
5. Suggest one practical mini-project idea.

🎯 Instructions:
- Return ONLY valid, minified JSON using EXACTLY this structure:
{"student_goal": "string", "current_issues": ["issue1", "issue2"], "roadmap": ["step1", "step2"], "resources": ["resource1", "resource2"], "project_idea": "string"}

- Do not include any explanations, apologies, or markdown.
- Return only the JSON directly.

STUDENT INPUT:
{student_input}
"""

prompt = PromptTemplate(
    input_variables=["student_input"],
    template=template
)

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# LLM Chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Analysis function
def analyze_learning_goal(student_input):
    try:
        response = chain.invoke({"student_input": student_input})
        if isinstance(response, dict) and "text" in response:
            response_text = response["text"]
        else:
            response_text = str(response)

        # Debug: print raw output
        print("RAW OUTPUT START")
        print(response_text)
        print("RAW OUTPUT END")

        # Strict regex to extract JSON
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            cleaned_json = json_match.group(0)
            parsed_json = json.loads(cleaned_json)
            return parsed_json

        # If extraction fails
        return {
            "error": "Could not parse LLM response as JSON.",
            "raw_output": response_text
        }

    except Exception as e:
        return {"error": str(e)}
