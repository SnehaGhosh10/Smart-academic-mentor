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
    temperature=0.0,  # Ensure deterministic structured output
    model_name="llama3-8b-8192",
    api_key=groq_api_key
)

# Strong structured prompt
template = """
You are a Smart Academic Mentor AI.

You help students convert vague learning goals into a structured plan.

Return ONLY valid, minified JSON in this exact structure:
{"student_goal":"string","current_issues":["issue1","issue2"],"roadmap":["step1","step2"],"resources":["resource1","resource2"],"project_idea":"string"}

⚠️ Do not add any explanations, markdown, code fences, or additional commentary. Output only the JSON.

STUDENT_INPUT:
{student_input}
"""

# Prompt template
prompt = PromptTemplate(
    input_variables=["student_input"],
    template=template
)

# LLM chain
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

        # Optional: Remove accidental code fences
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Debug log (optional)
        print("LLM Raw Output:", response_text)

        # Use regex to extract first JSON object
        json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            parsed_json = json.loads(json_text)
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
