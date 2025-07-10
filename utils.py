import streamlit as st
import re
import random
from datetime import datetime
import pytz
import json, os
import pandas as pd
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import threading

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
            context = "\n\n".join([doc.page_content for doc in result])

            # Create prompt
            full_prompt = f"""Answer briefly using the following context. If irrelevant, say: OUT OF CONTEXT.

Context:
{context}

Question: {user_query}"""

            # Call Gemini model
            try:
                response = model.generate_content(full_prompt)
                st.success(f"Answer: {response.text.strip()}")
            except Exception as e:
                st.error(f"Error: {e}")

def generate_mcqs(context, model, num_qs):
    variation_tag = random.choice([
        "Ensure these questions are phrased differently from earlier ones.",
        "Try not to repeat phrasing from previous sets.",
        "This set should sound slightly different.",
        "Rephrase questions uniquely while preserving accuracy.",
        "Avoid repeating earlier structure exactly."
    ])
    prompt = f"""You are a NEET Biology tutor. Based on the context below, generate exactly {num_qs} multiple choice questions.
    
    Each question should be in this format:
    Q1. <question text>
    a) <option a>
    b) <option b>
    c) <option c>
    d) <option d>
    
    DO NOT REVEAL THE ANSWER BEFORE THE USER ANSWERS THE QUESTION.
    
    {variation_tag}
    
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
You are an expert NEET Biology examiner. Evaluate the student's answer to the following MCQ.

{question}

Student's answer: {user_answer}

Instructions:
1. First line must be: Correct or Incorrect.
2. Second line: The correct answer is X) ...
3. Third line: A one-line explanation why it's correct.

Each on a new line. No merging.
Don't repeat the question or options.
"""

    response = model.generate_content(prompt)
    response_text = response.text.strip()

    # Extract correct letter
    match = re.search(r"[Tt]he correct answer is ([a-dA-D])\)", response_text)
    correct_letter = match.group(1).lower() if match else "?"

    # Determine correctness
    is_correct = user_answer.lower() == correct_letter

    # Extract explanation (if it's not neatly split, fallback to last sentence)
    lines = response_text.splitlines()
    if len(lines) >= 3:
        explanation = lines[2].strip()
    else:
        # fallback: extract last sentence or guess explanation
        parts = re.split(r"\. |\n", response_text)
        explanation = parts[-1].strip() if parts else ""

    return response_text, correct_letter, is_correct, explanation


def view_my_results():
    st.subheader("Review My Quiz Results")

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

        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        df.rename(columns={
            "Q#": "Question No",
            "Your Answer": "Selected Answer",
            "Time": "Time Taken (s)",
            "Date": "Timestamp"  
        }, inplace=True)

        # Filter for current user
        df_user = df[df["User Name"].str.lower() == username.strip().lower()]

        if df_user.empty:
            st.warning("No results found for this name.")
            return
        df_user["Timestamp"] = pd.to_datetime(df_user["Timestamp"])
        df_user = df_user.sort_values("Timestamp", ascending=False)

        # Group by each quiz attempt: Timestamp + Topic
        grouped = df_user.groupby([df_user["Timestamp"].dt.strftime('%Y-%m-%d %H:%M'), "Topic"])

        for (attempt_time, topic), group in grouped:
            with st.container():
                st.markdown(f"### Attempt on {attempt_time} — Topic: *{topic}*")

            for _, row in group.iterrows():   
                question_text = row['Question'].split('. ', 1)[-1]
                with st.expander(f"{row['Question No']} - {question_text}"):
                    st.markdown(f"**Options:** {row.get('Options', 'N/A')}")
                    st.markdown(f"**Your Answer:** {row['Selected Answer']}")
                    st.markdown(f"**Correct Answer:** {row['Correct Answer']}")
                    st.markdown(f"**Result:** {row['Result']}")
                    st.markdown(f"**Explanation:** {row.get('Explanation', 'N/A')}")
                    st.markdown(f"**Time Taken:** {row.get('Time Taken (s)', 'N/A')} sec")


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

def get_gsheet_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)
    return gspread.authorize(creds)


def save_detailed_quiz_to_gsheet(username, topic, questions, selected_answers, correct_answers, correctness, timings, all_options, explanations):
    client = get_gsheet_client()
    sheet = client.open("quiz_scores").sheet1

    ist = pytz.timezone("Asia/Kolkata")

    for i in range(len(questions)):
        question_text = questions[i].splitlines()[0]  # first line is Q text
        selected = selected_answers[i]
        correct = correct_answers[i]
        result = "Correct" if correctness[i] else "Incorrect"
        time_taken = timings[i] if i < len(timings) else ""

        options_str = ", ".join(all_options[i])  # list to string
        explanation = explanations[i] if i < len(explanations) else ""

        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

        row = [
            timestamp,
            username,
            topic,
            f"Q{i+1}",
            question_text,
            options_str,
            correct,
            selected,
            result,
            explanation,
            time_taken
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
            "saved_to_sheet": False,
            "timings": [],
            "options_list": [],
            "explanations": []
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
                random.shuffle(docs)
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
                    "saved_to_sheet": False,
                    "timings": [],
                    "options_list": [],
                    "explanations": []
                })
                st.session_state.quiz_start_time = datetime.now()
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

                    # Updated check_answer returns explanation too
                    feedback, correct_answer, is_correct, explanation = check_answer(q, selected_letter, model)

                    # Save timing
                    state["timings"].append((datetime.now() - st.session_state.quiz_start_time).total_seconds())

                    # Save feedback
                    st.session_state[f"feedback_{index}"] = feedback
                    st.session_state[f"correct_answer_{index}"] = correct_answer
                    st.session_state[f"selected_letter_{index}"] = selected_letter
                    st.session_state[f"is_correct_{index}"] = is_correct
                    st.session_state[f"submitted_{index}"] = True

                    # Save options and explanation
                    state["options_list"].append(options)
                    state["explanations"].append(explanation)

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
            total_time = (datetime.now() - st.session_state.quiz_start_time).total_seconds()
            st.info(f"Total time taken: **{total_time:.2f} seconds**")

            if not state["saved_to_sheet"]:
                save_detailed_quiz_to_gsheet(
                    username=state["username"],
                    topic=state["topic"],
                    questions=state["questions"],
                    selected_answers=state["selected_answers"],
                    correct_answers=state["correct_answers"],
                    correctness=state["correctness"],
                    timings=state["timings"],
                    all_options=state["options_list"],
                    explanations=state["explanations"]
                )
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
                    "saved_to_sheet": False,
                    "timings": [],
                    "options_list": [],
                    "explanations": []
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

        required_cols = {"User Name", "Timestamp", "Topic", "Result"}
        if not required_cols.issubset(df.columns):
            st.error("Missing required columns in Google Sheet.")
            return

        usernames = sorted(df["User Name"].dropna().unique())
        selected_user = st.selectbox("Select user:", usernames)

        df = df[df["User Name"] == selected_user].copy()

        # Convert timestamp and prepare for grouping
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.floor("min")
        df["Result"] = df["Result"].str.lower().str.strip()
        df["Correct"] = df["Result"].apply(lambda x: 1 if x == "correct" else 0)

        grouped = df.groupby(["Timestamp", "Topic"]).agg(
            total_qs=pd.NamedAgg(column="Result", aggfunc="count"),
            correct=pd.NamedAgg(column="Correct", aggfunc="sum")
        ).reset_index()

        grouped["Accuracy"] = (grouped["correct"] / grouped["total_qs"]) * 100

        if grouped.empty:
            st.info("No quiz records to plot.")
            return

        st.markdown(f"### Accuracy Trend for *{selected_user}*")

        fig, ax = plt.subplots(figsize=(10, 5))

        for topic in grouped["Topic"].unique():
            topic_df = grouped[grouped["Topic"] == topic]
            ax.plot(topic_df["Timestamp"], topic_df["Accuracy"], marker="o", label=topic)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Topic-wise Accuracy Over Time")
        ax.set_ylim(-10, 110)
        ax.legend(title="Topic")
        ax.grid(True)
        fig.autofmt_xdate()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to load score trend: {e}")

def leaderboard():
    st.subheader("Leaderboard by Topic")

    try:
        client = get_gsheet_client()
        sheet = client.open("quiz_scores").sheet1
        df = pd.DataFrame(sheet.get_all_records())

        # Update column check to match your sheet
        required_cols = {"User Name", "Timestamp", "Topic", "Result"}
        if not required_cols.issubset(df.columns):
            st.error("Missing required columns in Google Sheet.")
            return

        # Preprocess
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Result"] = df["Result"].str.lower().str.strip()
        df["Correct"] = df["Result"].apply(lambda x: 1 if x == "correct" else 0)

        # Topic selection
        topic_options = sorted(df["Topic"].dropna().unique())
        selected_topic = st.selectbox("Select Topic", topic_options)

        df_topic = df[df["Topic"] == selected_topic]

        # Handle empty topic case
        if df_topic.empty:
            st.info(f"No quiz attempts found for topic '{selected_topic}'.")
            return

        # Find latest attempt per user
        latest_attempts = (
            df_topic.groupby("User Name")["Timestamp"]
            .max()
            .reset_index()
            .rename(columns={"Timestamp": "Latest_Timestamp"})
        )

        merged = pd.merge(df_topic, latest_attempts, on="User Name", how="inner")

        # Compare only date to avoid microsecond mismatches
        merged["Timestamp_only"] = merged["Timestamp"].dt.date
        merged["Latest_Timestamp_only"] = merged["Latest_Timestamp"].dt.date

        latest_df = merged[merged["Timestamp_only"] == merged["Latest_Timestamp_only"]]

        # Accuracy per user
        summary = (
            latest_df.groupby("User Name")
            .agg(
                Accuracy=("Correct", lambda x: round((x.sum() / len(x)) * 100)),
                Total_Questions=("Correct", "count"),
                Latest_Attempt=("Timestamp", "first"),
            )
            .reset_index()
            .sort_values("Accuracy", ascending=False)
            .reset_index(drop=True)
        )

        summary.insert(0, "Rank", range(1, len(summary) + 1))

        # Show top 3 performers
        st.markdown("###Top 3 Performers")
        st.dataframe(summary.head(3), use_container_width=True)

        # Expandable full leaderboard
        with st.expander("See Full Leaderboard"):
            st.dataframe(summary, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load leaderboard: {e}")



