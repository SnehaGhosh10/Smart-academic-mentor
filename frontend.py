import streamlit as st
from backend import analyze_learning_goal

# Streamlit config
st.set_page_config(page_title="Smart Academic Mentor", layout="centered")
st.title("🎓 Smart Academic Mentor – From Confused to Clarity")
st.markdown("Type your **learning goal or confusion** below to receive a structured learning roadmap powered by GenAI.")

# Input box
student_input = st.text_area("🖊️ Your Learning Goal, Doubt, or Area of Confusion", height=150)

if st.button("🚀 Get Mentorship Plan"):
    if student_input.strip():
        with st.spinner("🤖 Generating your personalized mentorship roadmap..."):
            result = analyze_learning_goal(student_input)
        
        if "error" in result:
            st.error("⚠️ Could not parse the AI output properly.")
            st.text_area("🔍 Raw AI Output for Debugging", result.get("raw_output", "No output available."), height=300)
        else:
            st.success("✅ Roadmap Generated!")

            st.markdown("## 🎯 Your Goal")
            st.markdown(f"**{result['student_goal']}**")

            st.markdown("## 🚧 Current Issues")
            for issue in result.get("current_issues", []):
                st.markdown(f"- {issue}")

            st.markdown("## 🗺️ Recommended Roadmap")
            for idx, step in enumerate(result.get("roadmap", []), start=1):
                st.markdown(f"{idx}. {step}")

            st.markdown("## 📚 Resources")
            for resource in result.get("resources", []):
                st.markdown(f"- {resource}")

            st.markdown("## 💡 Project Idea")
            st.info(result.get("project_idea", "No project idea provided."))
    else:
        st.warning("⚠️ Please enter your learning goal or confusion before requesting a roadmap.")

st.markdown("---")
st.caption("🚀 Built with ❤️ using Streamlit + Groq LLM to turn confusion into clarity for students.")
