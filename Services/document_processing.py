from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Services.parsers import recursive_character_text_splitter
class Load_and_Chunk:
    
    def pdf_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = PyPDFLoader(file_path=file_path)
        data = loader.load()
        documents = recursive_character_text_splitter(data, chunk_size, chunk_overlap)
        print("len of docs",len(documents))
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
        print("len of docs",len(documents))
            
        return documents

    def csv_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = UnstructuredExcelLoader(file_path=file_path)
        data = loader.load()
        documents = recursive_character_text_splitter(data, chunk_size, chunk_overlap)

        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents

    def excel_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = CSVLoader(file_path=file_path)
        data = loader.load()
        documents = recursive_character_text_splitter(data, chunk_size, chunk_overlap)

        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents

    def doc_and_docx_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = Docx2txtLoader(file_path=file_path)
        data = loader.load()
        documents = recursive_character_text_splitter(data, chunk_size, chunk_overlap)
        
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents

    def txt_loader(file_path, chunk_size=300, chunk_overlap=20):
        loader = TextLoader(file_path=file_path)
        data = loader.load()
        documents = recursive_character_text_splitter(data, chunk_size, chunk_overlap)
        
        for doc in documents:
            doc.metadata= { **doc.metadata, "file_name": file_path.split("/")[-1], "chunk_size" : chunk_size }
            
        return documents