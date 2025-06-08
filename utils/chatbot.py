from openai import OpenAI

def query_bot(query, index, text_chunks, model):
    embedding = model.embed_query(query)
    D, I = index.search(np.array([embedding]), k=3)
    retrieved_text = "\n".join([text_chunks[i] for i in I[0]])
    
    prompt = f"""You are a chatbot that only uses the following context:
{retrieved_text}

User: {query}
Bot:"""

    response = OpenAI().ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
