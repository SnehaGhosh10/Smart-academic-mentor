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

# Strong structured prompt
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

# Prompt template
prompt = PromptTemplate(
    input_variables=["student_input"],
    template=template
)

# LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Utility to extract clean JSON
def extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            parsed_json = json.loads(json_text)
            return parsed_json
        else:
            return {
                "error": "⚠️ Could not find valid JSON in AI output.",
                "raw_output": text
            }
    except Exception as e:
        return {
            "error": f"⚠️ Error while parsing JSON: {e}",
            "raw_output": text
        }

# Analysis function
def analyze_learning_goal(student_input):
    try:
        response = chain.invoke({"student_input": student_input})
        response_text = response.get("text", str(response)).strip()
        print("LLM Raw Output:", response_text)
        return extract_json(response_text)
    except Exception as e:
        return {"error": f"⚠️ Error during LLM analysis: {str(e)}"}

