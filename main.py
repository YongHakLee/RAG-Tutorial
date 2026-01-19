import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def main():
    # ==========================================
    # 1. PDF 문서 로드 (Document Loading)
    # ==========================================
    print("1. PDF 문서를 로드 중...")
    file_path = (
        "data/10-2024-0099646_출원서.pdf"  # 여기에 실제 PDF 파일 경로 입력
    )
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"   - 총 {len(docs)} 페이지를 로드함.")

    # ==========================================
    # 2. 텍스트 분할 (Text Splitting)
    # ==========================================
    print("2. 텍스트를 작은 조각(Chunk)으로 분할 중...")
    # chunk_size: 한 조각의 크기 (1000자)
    # chunk_overlap: 조각 간 겹치는 구간 (200자) - 문맥 단절 방지
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"   - 총 {len(splits)} 개의 조각으로 분할됨.")

    # ==========================================
    # 3. 임베딩 모델 설정 및 벡터 DB 저장
    # ==========================================
    print("3. 벡터 데이터베이스 생성 중 (시간이 조금 걸릴 수 있음)...")
    # 한국어 성능이 좋은 임베딩 모델 사용 (HuggingFace)
    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={"device": "cuda"},  # GPU가 있다면 'cuda'로 변경
        encode_kwargs={"normalize_embeddings": True},
    )

    # ChromaDB에 문서 벡터 저장
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name="korean_rag_db",  # DB 이름
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )  # 유사한 문서 3개 검색

    # ==========================================
    # 4. Ollama 모델 설정 (Ministral-3:8B)
    # ==========================================
    print("4. Ollama 모델 연결 중...")
    # 사용자가 Ollama에 pull 받은 모델 이름과 정확히 일치해야 함
    llm = ChatOllama(
        model="ministral-3:8b",
        temperature=0,  # 0으로 설정하여 사실에 기반한 답변 유도
    )

    # ==========================================
    # 5. 프롬프트 및 체인(Chain) 구성
    # ==========================================
    # RAG를 위한 프롬프트 템플릿
    template = """
    당신은 주어진 문서를 기반으로 질문에 답변하는 유능한 AI 어시스턴트입니다.
    아래의 [참고 문서]만을 사용하여 질문에 답변하세요.
    문서에 없는 내용은 지어내지 말고 "문서에서 정보를 찾을 수 없습니다"라고 말하세요.
    답변은 반드시 한국어로 작성하세요.

    [참고 문서]:
    {context}

    질문: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LangChain 파이프라인 연결 (Retriever -> Format -> Prompt -> LLM -> Output)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ==========================================
    # 6. 실행 및 테스트
    # ==========================================
    print("--- 설정 완료! 질문을 시작합니다. ---")

    query = "이 문서의 핵심 내용이 무엇인가요?"  # 원하는 질문으로 변경 가능
    print(f"질문: {query}")

    response = rag_chain.invoke(query)
    print("\n답변:")
    print(response)


if __name__ == "__main__":
    main()
