import os
from utils import ingest, answer_query

def main():
    persist_dir = "./chroma_db"

    # Ingest only once
    if not os.path.exists(persist_dir):
        print("ChromaDB not found. Running ingestion...")
        ingest(speech_path="speech.txt", persist_dir=persist_dir)
    else:
        print("ChromaDB already exists. Skipping ingestion.")

    print("\nRAG system ready! Ask any question about the speech.")
    print("Type 'exit' to quit.\n")

    # Main question loop
    while True:
        question = input("Your question: ")

        # Exit condition
        if question.lower().strip() in ["exit", "quit", "q"]:
            print("\nExiting... Goodbye!")
            break

        # Process the question
        answer = answer_query(question=question, persist_dir=persist_dir)

        print("\n=== ANSWER ===")
        print(answer)
        print("\n----------------------------\n")


if __name__ == "__main__":
    main()
