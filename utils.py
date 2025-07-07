import streamlit as st
import re
from datetime import datetime
import json, os
import pandas as pd
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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

def generate_mcqs(context, model, num_qs):
    prompt = f"""You are a NEET Biology tutor. Based on the context below, generate exactly {num_qs} multiple choice questions.

Each question should be in this format:
Q1. <question text>
a) <option a>
b) <option b>
c) <option c>
d) <option d>

DO NOT REVEAL THE ANSWER BEFORE THE USER ANSWERS THE QUESTION.


Context:
\"\"\"
{context}
\"\"\"
Only provide the questions and answer key as described above. Do not skip any numbers.
"""

    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text


def check_answer(question, user_answer, model):
    prompt = f"""
You are an exam checker. A student answered this MCQ:

{question}

Student's answer: {user_answer}

Tell if it's correct or not and also reveal the correct answer like this:
"The correct answer is b) ..." (Start this line with 'The correct answer is').

Provide one line explanation. Avoid repeating question or options.

"""
    response = model.generate_content(prompt)
    response_text=response.text.strip()
    match = re.search(r"[Cc]orrect.*?([a-dA-D])\)", response_text)
    correct_letter = match.group(1).lower() if match else "?"

    return response_text, correct_letter

def view_my_results():
    st.subheader("View My Quiz Results")

    username = st.text_input("Enter your name to view your results:")

    if not username:
        st.info("Please enter your name above to load your results.")
        return

    try:
        client = get_gsheet_client()
        sheet = client.open("quiz_scores").sheet1
        data = sheet.get_all_records()

        if not data:
            st.info("No quiz results found yet.")
            return

        df = pd.DataFrame(data)

        # Filter for current user
        df_user = df[df["User Name"].str.lower() == username.strip().lower()]

        if df_user.empty:
            st.warning("No results found for this name.")
            return

        st.success(f"Showing results for: {username}")
        st.dataframe(df_user)

        # Download as CSV
        csv = df_user.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"{username}_quiz_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Failed to load results: {e}")


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

def get_gsheet_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)
    return gspread.authorize(creds)

def save_detailed_quiz_to_gsheet(username,topic, questions, selected_answers, correct_answers, correctness):
    client = get_gsheet_client()
    sheet = client.open("quiz_scores").sheet1

    for i in range(len(questions)):
        question_text = questions[i].splitlines()[0]
        selected = selected_answers[i]
        correct = correct_answers[i]
        result = "Correct" if correctness[i] else "Incorrect"

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            username,
            topic,
            f"Q{i+1}",
            question_text,
            correct,
            selected,
            result,
        ]
        sheet.append_row(row)

def quiz_mode(retriever, model):
    st.subheader("Quiz Mode")

    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {
            "started": False,
            "username": "",
            "topic": "",
            "questions": [],
            "index": 0,
            "score": 0,
            "total": 0,
            "selected_answers": [],
            "correct_answers": [],
            "correctness": [],
            "saved_to_sheet": False
        }

    state = st.session_state.quiz_state

    topic_options = [
        "Biodiversity and Conservation",
        "Sexual Reproduction in Plants",
        "Human Health and Disease"
    ]

    if not state["started"]:
        username = st.text_input("Enter your name:")
        topic = st.selectbox("Choose a topic:", topic_options)
        num_qs = st.slider("Select number of questions", 1, 10, 2)

        if st.button("Start Quiz"):
            if not username.strip():
                st.warning("Please enter your name to start the quiz.")
                return

            with st.spinner("Generating quiz..."):
                docs = retriever.get_relevant_documents(topic)
                context = "\n\n".join(doc.page_content for doc in docs[:5])
                mcq_text = generate_mcqs(context, model, num_qs)

                questions = re.split(r"\n(?=Q\d+\.)", mcq_text)
                questions = [q.strip() for q in questions if q.strip()]

                state.update({
                    "started": True,
                    "username": username.strip(),
                    "topic": topic,
                    "questions": questions,
                    "index": 0,
                    "score": 0,
                    "total": len(questions),
                    "selected_answers": [],
                    "correct_answers": [],
                    "correctness": [],
                    "saved_to_sheet": False
                })
            st.rerun()

    else:
        questions = state["questions"]
        index = state["index"]

        if index < len(questions):
            q = questions[index]
            q_lines = q.splitlines()

            st.markdown(f"### Question {index + 1} of {state['total']}")
            st.markdown(f"**{q_lines[0]}**")

            options = [line for line in q_lines[1:] if re.match(r"[A-Da-d]\)", line)]
            selected = st.radio("Choose your answer:", options, key=f"q{index}_opt")

            if f"submitted_{index}" not in st.session_state:
                st.session_state[f"submitted_{index}"] = False

            if not st.session_state[f"submitted_{index}"]:
                if st.button("Submit", key=f"submit_{index}"):
                    selected_letter = selected[0].lower() if selected else ""
                    feedback, correct_answer = check_answer(q, selected_letter, model)
                    is_correct = feedback.strip().lower().startswith("correct")

                    st.session_state[f"feedback_{index}"] = feedback
                    st.session_state[f"correct_answer_{index}"] = correct_answer
                    st.session_state[f"selected_letter_{index}"] = selected_letter
                    st.session_state[f"is_correct_{index}"] = is_correct
                    st.session_state[f"submitted_{index}"] = True
                    st.rerun()

            else:
                st.markdown(f"**Feedback:** {st.session_state.get(f'feedback_{index}', '')}")

                if st.button("Next", key=f"next_{index}"):
                    selected_letter = st.session_state.get(f"selected_letter_{index}", "")
                    correct_answer = st.session_state.get(f"correct_answer_{index}", "?")
                    is_correct = st.session_state.get(f"is_correct_{index}", False)

                    state["selected_answers"].append(selected_letter)
                    state["correct_answers"].append(correct_answer)
                    state["correctness"].append(is_correct)

                    if is_correct:
                        state["score"] += 1

                    state["index"] += 1
                    st.rerun()

        else:
            st.success(f"Quiz Completed! Your Score: {state['score']} / {state['total']}")

            if not state["saved_to_sheet"]:
                save_detailed_quiz_to_gsheet(
                    state["username"],
                    state["topic"],
                    state["questions"],
                    state["selected_answers"],
                    state["correct_answers"],
                    state["correctness"]
                )
                update_topicwise_performance(state["topic"], state["score"], state["total"])
                state["saved_to_sheet"] = True

            if st.button("Restart Quiz"):
                keys_to_clear = [key for key in st.session_state if key.startswith("feedback_") or key.startswith("submitted_") or key.startswith("selected_letter_") or key.startswith("is_correct_") or key.startswith("correct_answer_") or key.startswith("q")]
                for key in keys_to_clear:
                    del st.session_state[key]

                st.session_state.quiz_state = {
                    "started": False,
                    "username": "",
                    "topic": "",
                    "questions": [],
                    "index": 0,
                    "score": 0,
                    "total": 0,
                    "selected_answers": [],
                    "correct_answers": [],
                    "correctness": [],
                    "saved_to_sheet": False
                }
                st.rerun()

def plot_score():
    st.subheader("Performance Trend (Topic-wise Over Time)")

    try:
        client = get_gsheet_client()
        sheet = client.open("quiz_scores").sheet1
        records = sheet.get_all_records()

        if not records:
            st.info("No quiz data available.")
            return

        df = pd.DataFrame(records)

        required_cols = {"User Name", "Date", "Topic", "Result"}
        if not required_cols.issubset(df.columns):
            st.error("Missing required columns in Google Sheet.")
            return

        usernames = sorted(df["User Name"].dropna().unique())
        selected_user = st.selectbox("Select user:", usernames)

        df = df[df["User Name"] == selected_user].copy()

        # Round to minute to group by quiz attempt
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.floor("min")
        df["Result"] = df["Result"].str.lower().str.strip()
        df["Correct"] = df["Result"].apply(lambda x: 1 if x == "correct" else 0)

        grouped = df.groupby(["Date", "Topic"]).agg(
            total_qs=pd.NamedAgg(column="Result", aggfunc="count"),
            correct=pd.NamedAgg(column="Correct", aggfunc="sum")
        ).reset_index()

        grouped["Accuracy"] = (grouped["correct"] / grouped["total_qs"]) * 100

        if grouped.empty:
            st.info("No quiz records to plot.")
            return

        st.markdown(f"### Accuracy Trend for *{selected_user}*")

        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 5))

        for topic in grouped["Topic"].unique():
            topic_df = grouped[grouped["Topic"] == topic]
            ax.plot(topic_df["Date"], topic_df["Accuracy"], marker="o", label=topic)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Topic-wise Accuracy Over Time")
        ax.set_ylim(0, 110)
        ax.legend(title="Topic")
        ax.grid(True)
        fig.autofmt_xdate()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to load score trend: {e}")




