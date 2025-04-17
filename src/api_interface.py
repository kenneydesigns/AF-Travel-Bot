import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def choose_best_source(user_question: str, summaries: list[str]) -> str:
    prompt = (
        "You are a military travel assistant.\n"
        "Based on the following question:\n"
        f"Q: {user_question}\n\n"
        "Here are summaries of available documents:\n" +
        "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries)]) +
        "\n\nWhich document best answers the question? Reply with just the number."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
