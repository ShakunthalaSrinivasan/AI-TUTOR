import streamlit as st
import re
from datetime import datetime
import json, os
import pandas as pd
import matplotlib.pyplot as plt

def question_mode(retriever, model):
    st.subheader("Ask a Question")

    topic_options = [
    "Biodiversity and Conservation",
    "Sexual Reproduction in Plants",
    "Human Health and Disease"]

    selected_topic = st.selectbox("Select a topic:", topic_options)
    user_query = st.text_input("Ask a question based on the selected topic:")
    
    if st.button("Get Answer"):
        if not user_query or not user_query.strip():
            st.warning("Please enter a valid question.")
            return

        with st.spinner("Processing..."):
            # Get documents
            result = retriever.get_relevant_documents(user_query)
            context = "\n\n".join([doc.page_content[:500] for doc in result])

            # Create prompt
            full_prompt = f"""Answer briefly using the following context. If irrelevant, say: OUT OF CONTEXT.

Context:
{context}

Question: {user_query}"""

            # Call Gemini model
            try:
                response = model.generate_content(full_prompt)
                st.success("Answer:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

def generate_mcqs(context, model, num_qs=2):
    prompt = f"""
    Rules:
    DO CREATE {num_qs} MCQs based only on the following context.
    1. Start each question with Q1, Q2, Q3,...
    2. After each question wait for the user answer.
    3. For each question, give exactly 4 options labeled a-d.
    4. FORMAT should be the same. DO NOT add any extra text.
    5. Do not reveal the correct answer.
    6. Make questions directly based on the context.

    Context:
    {context}
    """
    response = model.generate_content(prompt, generation_config={"temperature": 0})
    return response.text

def check_answer(question, user_answer, model):
    prompt = f"""
You are an exam checker. A student answered this MCQ:

{question}

Student's answer: {user_answer}

Tell if it's correct or not and reveal the correct answer with one line explanation.
Tell only correct or incorrect.

"""
    response = model.generate_content(prompt)
    return response.text.strip()

def view_quiz_history(file_path="quiz_log.json"):
    import streamlit as st
    from datetime import datetime

    if not os.path.exists(file_path):
        st.warning("No quiz data yet.")
        return

    with open(file_path, "r") as f:
        logs = json.load(f)

    st.subheader("Quiz History")
    for i, entry in enumerate(logs, 1):
        date = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
        st.write(f"{i}. [{date}] Topic: {entry['topic']} - Score: {entry['score']}/{entry['total']}")

def update_topicwise_performance(topic, score, total, file_path="topics.json"):
    stats = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            stats = json.load(f)
    if topic not in stats:
        stats[topic] = {"correct": 0, "total": 0}
    stats[topic]["correct"] += score
    stats[topic]["total"] += total
    with open(file_path, "w") as f:
        json.dump(stats, f, indent=2)

def save_quiz_score(topic, questions, score, total, file_path="quiz_log.json"):
    quiz_data = {
        "topic": topic,
        "score": score,
        "total": total,
        "timestamp": datetime.now().isoformat(),
        "questions": questions
    }
    logs = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            logs = json.load(f)
    logs.append(quiz_data)
    with open(file_path, "w") as f:
        json.dump(logs, f, indent=2)

def quiz_mode(retriever, model):
    st.subheader("Quiz Generator")

    topic_options = [
    "Biodiversity and Conservation",
    "Sexual Reproduction in Plants",
    "Human Health and Disease"]

    topic = st.selectbox("Select a topic to generate quiz:", topic_options)
    num_qs = st.slider("Number of Questions", 1, 10, 3)

    if st.button("Start Quiz") and topic:
        with st.spinner("Generating quiz..."):
            docs = retriever.get_relevant_documents(topic)
            context = "\n\n".join(doc.page_content[:500] for doc in docs[:1])
            mcq_text = generate_mcqs(context, model, num_qs)

            questions = re.split(r"\n(?=Q\d+\.)", mcq_text)
            questions = [q.strip() for q in questions if q.strip()]
            
            st.session_state['quiz'] = {
                'topic': topic,
                'questions': questions,
                'index': 0,
                'score': 0,
                'num_qs': num_qs
            }

    # Show next question
    if 'quiz' in st.session_state:
        qdata = st.session_state['quiz']
        index = qdata['index']

        if index < len(qdata['questions']):
            question_text = qdata['questions'][index]
            st.markdown(f"### Q{index + 1}")
            st.text(question_text)

            options = re.findall(r"[a-d]\)\s.*", question_text)
            user_ans = st.radio("Choose your answer:", options, key=f"q{index}")

            if st.button("Submit", key=f"submit{index}"):
                # Extract a/b/c/d from selected option
                selected = user_ans[0].lower() if user_ans else ''
                feedback = check_answer(question_text, selected, model)
                st.write(feedback)

                if feedback.strip().lower().startswith("correct"):
                    qdata['score'] += 1

                qdata['index'] += 1  # Move to next question
                st.experimental_rerun()

        else:
            st.success(f"Quiz Completed! Score: {qdata['score']}/{qdata['num_qs']}")
            save_quiz_score(qdata['topic'], qdata['questions'], qdata['score'], qdata['num_qs'])
            update_topicwise_performance(qdata['topic'], qdata['score'], qdata['num_qs'])
            del st.session_state['quiz']


def plot_score(file_path="quiz_log.json"):
    if not os.path.exists(file_path):
        st.warning("No quiz history yet.")
        return

    with open(file_path, "r") as f:
        logs = json.load(f)

    data = []
    for entry in logs:
        data.append({
            "Date": datetime.fromisoformat(entry["timestamp"]),
            "Topic": entry["topic"],
            "Score": entry["score"],
            "Total": entry["total"]
        })

    df = pd.DataFrame(data)
    df["Accuracy"] = (df["Score"] / df["Total"]) * 100
    df.sort_values("Date", inplace=True)

    if df.empty:
        st.info("No quiz records to show yet.")
        return

    # Line Chart - Topic-wise trend
    st.subheader("Topic-wise Accuracy Over Time")

    plt.clf()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for topic in df["Topic"].unique():
        topic_df = df[df["Topic"] == topic]
        ax1.plot(topic_df["Date"], topic_df["Accuracy"], marker="o", label=topic)

    ax1.set_title("Performance Trend by Topic")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.grid(True)
    fig1.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig1)

    # Bar Chart - Latest accuracy per topic
    st.subheader("Latest Accuracy Per Topic")

    latest = df.sort_values("Date").groupby("Topic").tail(1)

    plt.clf()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(latest["Topic"], latest["Accuracy"], color="skyblue")

    for i, row in enumerate(latest.itertuples()):
        ax2.text(i, row.Accuracy + 2, f"{row.Accuracy:.1f}%", ha="center")

    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Most Recent Quiz Accuracy")
    ax2.grid(True, axis="y")
    plt.tight_layout()
    st.pyplot(fig2)





