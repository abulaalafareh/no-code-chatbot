from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_character_text_splitter(data, chunk_size, chunk_overlap):
    print("chunk size is", chunk_size)
    text_splitter=RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    documents = text_splitter.split_documents(data)

    return documents