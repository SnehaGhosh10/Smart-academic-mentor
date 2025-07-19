from utils import academic_mentor_chain
import json
import re
import ast

def get_academic_mentorship(student_input):
    try:
        response = academic_mentor_chain.invoke({"student_input": student_input})
    except Exception as e:
        return {"error": f"LLM invocation error: {str(e)}", "raw_output": ""}

    response_text = response.get("text") if isinstance(response, dict) else str(response)

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            cleaned_json = json_match.group(0)
            parsed_json = json.loads(cleaned_json)
            return parsed_json
        except Exception as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_output": response_text
            }

    # Fallback using ast.literal_eval
    try:
        parsed = ast.literal_eval(response_text.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "error": "Could not parse response as valid JSON.",
        "raw_output": response_text
    }
