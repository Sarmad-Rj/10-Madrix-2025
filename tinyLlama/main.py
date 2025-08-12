from rag import build_index, retrieve
from model import load_model

def rag_answer(query, docs, embedder, index, generator):
    context = retrieve(query, docs, embedder, index, k=1)[0]

    system_instructions = (
        "You are Madrix, the official assistant for Sarmad Rj's portfolio site.\n"
        "Always speak in third person about Sarmad Rj.\n"
        "Only use the provided context to answer in one short sentence.\n"
    )

    prompt = f"""{system_instructions}
Context: {context}
Question: {query}
Answer:"""

    # Generate a bit more text than needed
    result = generator(prompt, max_new_tokens=50)[0]['generated_text']

    # Get only the part after "Answer:" and stop if "Question:" appears
    answer = result.split("Answer:")[-1].strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()

    return answer



def main():
    docs = [
        "Sarmad Rj is studying at COMSATS University in Pakistan.",
        "Sarmad Rj is skilled in Python, Java, and web development.",
        "Sarmad Rj created a school management system using Java Swing and Oracle SQL."
    ]
    embedder, index = build_index(docs)
    generator = load_model()

    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["quit", "exit"]:
            break
        print(rag_answer(query, docs, embedder, index, generator))

if __name__ == "__main__":
    main()
