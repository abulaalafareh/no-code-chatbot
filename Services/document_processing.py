from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Load_and_Chunk:
    
    def pdf_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            separators=['\n\n'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        ))
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents

    def csv_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = UnstructuredExcelLoader(file_path=file_path)
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            separators=['\n\n'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        ))
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents

    def excel_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = CSVLoader(file_path=file_path)
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            separators=['\n\n'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        ))
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents

    def doc_and_docx_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = Docx2txtLoader(file_path=file_path)
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            separators=['\n\n'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        ))
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents

    def txt_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = TextLoader(file_path=file_path)
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            separators=['\n\n'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        ))
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents