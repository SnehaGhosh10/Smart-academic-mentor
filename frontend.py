import streamlit as st
from backend import analyze_learning_goal

# Streamlit page configuration
st.set_page_config(page_title="Smart Academic Mentor", layout="centered")
st.title("🎓 Smart Academic Mentor – From Confused to Clarity")
st.markdown(
    "Type your **learning goal or confusion** below to receive a structured learning roadmap powered by GenAI."
)

# Input box for student goal/confusion
student_input = st.text_area(
    "🖊️ Your Learning Goal, Doubt, or Area of Confusion",
    placeholder="e.g., I want to learn DSA for product-based companies but don't know where to start.",
    height=150,
)

# Action button
if st.button("🚀 Get Mentorship Plan"):
    if student_input.strip():
        with st.spinner("🤖 Generating your personalized mentorship roadmap..."):
            result = analyze_learning_goal(student_input)

        if "error" in result:
            st.error("⚠️ Could not parse the AI output properly.")
            with st.expander("🔍 Raw AI Output for Debugging"):
                st.code(result.get("raw_output", "No output available."), language="json")
        else:
            st.success("✅ Roadmap Generated!")

            # Display structured outputs clearly
            st.markdown("## 🎯 Your Goal")
            st.info(result.get("student_goal", "Not provided."))

            st.markdown("## 🚧 Current Issues")
            issues = result.get("current_issues", [])
            if issues:
                for issue in issues:
                    st.markdown(f"- {issue}")
            else:
                st.markdown("✅ No major issues identified.")

            st.markdown("## 🗺️ Recommended Roadmap")
            roadmap = result.get("roadmap", [])
            if roadmap:
                for idx, step in enumerate(roadmap, start=1):
                    st.markdown(f"{idx}. {step}")
            else:
                st.markdown("⚠️ No roadmap steps provided.")

            st.markdown("## 📚 Resources")
            resources = result.get("resources", [])
            if resources:
                for resource in resources:
                    st.markdown(f"- {resource}")
            else:
                st.markdown("⚠️ No resources suggested.")

            st.markdown("## 💡 Project Idea")
            st.info(result.get("project_idea", "No project idea provided."))

    else:
        st.warning("⚠️ Please enter your learning goal or confusion before requesting a roadmap.")

# Footer
st.markdown("---")
st.caption("🚀 Built with ❤️ using Streamlit + Groq LLM to turn confusion into clarity for students.")
