from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
    temperature=0.0,
    model_name="llama3-8b-8192",
    api_key=groq_api_key
)

# Strict structured prompt
template = """
You are a Smart Academic Mentor AI.

Your job:
- Take the student's vague learning goal or confusion.
- Clarify the goal.
- Identify current issues.
- Create a clear week-wise roadmap (3-8 steps).
- Suggest 2-4 free or structured resources.
- Give 1 project idea aligned with the goal.

⚠️ Output ONLY valid minified JSON in EXACTLY this structure:
{"student_goal":"string","current_issues":["issue1","issue2"],"roadmap":["step1","step2"],"resources":["resource1","resource2"],"project_idea":"string"}

Do NOT add explanations, markdown, commentary, or extra text.

STUDENT_INPUT:
{student_input}

⚠️ You will be penalized if you do not return JSON exactly as requested.
"""

# Create PromptTemplate
prompt = PromptTemplate(
    input_variables=["student_input"],
    template=template
)

# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Analysis function
def analyze_learning_goal(student_input):
    try:
        response = chain.invoke({"student_input": student_input})

        # Extract raw text
        if isinstance(response, dict) and "text" in response:
            response_text = response["text"]
        else:
            response_text = str(response)

        # Remove any accidental code fences
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Debug print (remove in production if needed)
        print("LLM Raw Output:", response_text)

        # Extract first JSON object using regex
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            parsed_json = json.loads(json_text)

            # Validate keys to ensure frontend safety
            expected_keys = {"student_goal", "current_issues", "roadmap", "resources", "project_idea"}
            if not expected_keys.issubset(parsed_json.keys()):
                return {
                    "error": "Parsed JSON missing expected keys.",
                    "raw_output": response_text
                }

            return parsed_json

        else:
            return {
                "error": "Could not find valid JSON in AI output.",
                "raw_output": response_text
            }

    except Exception as e:
        return {
            "error": f"Error during analysis: {str(e)}"
        }
