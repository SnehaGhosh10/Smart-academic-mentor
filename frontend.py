import streamlit as st
from backend import get_academic_mentorship

st.set_page_config(page_title="🎓 Smart Academic Mentor", layout="centered")

st.title("🎓 Smart Academic Mentor")
st.markdown("Type your **learning goal or confusion** below to receive a structured learning roadmap powered by GenAI.")

student_input = st.text_area("✏️ Your Learning Goal, Doubt, or Area of Confusion", height=150)

if st.button("🚀 Get Mentorship Plan"):
    if student_input.strip():
        with st.spinner("🤖 Thinking and generating your personalized learning roadmap..."):
            result = get_academic_mentorship(student_input)
        
        if "error" in result:
            st.error("⚠️ Could not parse the AI output properly.")
            st.text_area(
                "🔍 Raw AI Output for Debugging",
                result.get("raw_output", "No output available."),
                height=300
            )
        else:
            st.success("✅ Here is your personalized learning plan!")

            st.markdown("### 🎯 Goal")
            st.write(result.get("student_goal", "Not specified."))

            st.markdown("### 🚧 Current Issues")
            if result.get("current_issues"):
                for issue in result["current_issues"]:
                    st.markdown(f"- {issue}")
            else:
                st.markdown("_No major issues identified._")

            st.markdown("### 🗺️ Roadmap")
            if result.get("roadmap"):
                for step in result["roadmap"]:
                    st.markdown(f"✅ {step}")
            else:
                st.markdown("_No roadmap generated._")

            st.markdown("### 📚 Recommended Resources")
            if result.get("resources"):
                for resource in result["resources"]:
                    st.markdown(f"📌 {resource}")
            else:
                st.markdown("_No resources suggested._")

            st.markdown("### 💡 Project Idea")
            st.info(result.get("project_idea", "No project idea provided."))
    else:
        st.warning("⚠️ Please enter your learning goal or confusion before generating a plan.")

# Footer
st.markdown("---")
st.caption("🚀 Built with ❤️ using Streamlit + Groq LLM for personalized academic clarity and mentorship.")
