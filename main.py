from rag import build_index, retrieve
from model import load_model

def rag_answer(query, docs, embedder, index, generator):
    context = retrieve(query, docs, embedder, index, k=1)[0]
    prompt = f"""You are an assistant on Sarmad Rj's portfolio site.
    Always answer in third person about Sarmad Rj.
    Only answer the question concisely based on the given context.
    Do not explain your reasoning or repeat the question.

    Context: {context}
    Question: {query}
    Answer:"""

    result = generator(prompt)[0]['generated_text']
    return result.split("Answer:")[-1].strip()

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
