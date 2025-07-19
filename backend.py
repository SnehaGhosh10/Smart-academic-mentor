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

# Strict prompt
 template = """
You are an expert Smart Academic Mentor AI. Your task is to help students convert vague learning goals or confusion into a clear, structured learning plan.

Start by carefully reading and extracting key information:

STUDENT INPUT:
{student_input}

Now perform the following in order:
1. Identify and rephrase the student's primary learning goal clearly.
2. Identify current issues or blockers the student is facing regarding this goal.
3. Create a practical, step-by-step roadmap to achieve the learning goal.
4. Suggest 2-3 high-quality free or affordable learning resources aligned with the roadmap.
5. Suggest one actionable project idea related to the goal to help solidify the student's learning.

Return ONLY valid JSON with:
- student_goal (string)
- current_issues (list)
- roadmap (list)
- resources (list)
- project_idea (string)

Return raw JSON without markdown, code fences, explanations, or greetings.
"""


prompt = PromptTemplate(
    input_variables=["student_input"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def extract_json(text):
    """
    Attempts to robustly extract JSON from messy LLM output.
    """
    try:
        # Attempt direct load
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove code fences and backticks
    text = text.replace("```json", "").replace("```", "").strip()

    # Use regex to find the first JSON object
    json_match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None

def analyze_learning_goal(student_input):
    try:
        response = chain.invoke({"student_input": student_input})

        # Get raw text
        response_text = response.get("text", "") if isinstance(response, dict) else str(response)

        print("DEBUG Raw LLM Output:", response_text)  # keep for debugging

        parsed_json = extract_json(response_text)
        if parsed_json:
            return parsed_json

        return {
            "error": "Could not parse valid JSON from AI output.",
            "raw_output": response_text
        }

    except Exception as e:
        return {
            "error": f"Exception during analysis: {str(e)}"
        }
