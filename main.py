import os
import glob
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers.json import JsonOutputParser

from langgraph.graph import StateGraph, END

import pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    print("Error: OPENAI_API_KEY and PINECONE_API_KEY Not Found")
    exit()

llm = ChatOpenAI(
            model="usf1-mini",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.us.inc/usf/v1/hiring", 
            default_headers={
                "x-api-key": os.environ.get("OPENAI_API_KEY")
            }
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = None

PINECONE_INDEX_NAME = "transformationrag"

class GraphState(TypedDict):
    original_text: str
    user_goal: str
    style_analysis: dict
    retrieved_guides: str
    plan: List[str]
    transformed_text: str
    qc_result: dict
    retrieved_fact_check_guide: str
    
class StyleAnalysis(BaseModel):
    tone: str = Field(description="e.g., formal, casual, academic")
    complexity: str = Field(description="e.g., high, medium, low")
    format: str = Field(description="e.g., legal contract, email, scientific paper")
    vocabulary: str = Field(description="e.g., legal jargon, technical, everyday language")

class TransformationPlan(BaseModel):
    plan: List[str] = Field(description="A step-by-step plan to transform the content.")

class QualityControlResult(BaseModel):
    decision: str = Field(description="Either 'approve' or 'revise'.")
    feedback: str = Field(description="Detailed feedback for revision if the decision is 'revise'. Empty if 'approve'.")
    final_score: int = Field(description="A quality score from 1-10 on how well the output meets the goal.", ge=1, le=10)


def setup_rag_knowledge_base():
    global vectorstore
    knowledge_base_dir = "knowledge_base"
    if not os.path.exists(knowledge_base_dir):
        print(f"Error: The directory '{knowledge_base_dir}' was not found.")
        exit()

    filepaths = glob.glob(os.path.join(knowledge_base_dir, "*.txt"))
    
    if not filepaths:
        print(f"Error: No .txt files found in the '{knowledge_base_dir}' directory.")
        exit()
    
    texts_to_upload = []
    metadatas_to_upload = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts_to_upload.append(f.read())
        doc_id = os.path.splitext(os.path.basename(filepath))[0]
        metadatas_to_upload.append({"doc_id": doc_id})

    pc = pinecone.Pinecone()
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"\nCreating new Pinecone index: '{PINECONE_INDEX_NAME}'")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric='cosine',
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    print("\nUploading documents to Pinecone")
    vectorstore = PineconeVectorStore.from_texts(
        texts=texts_to_upload,
        metadatas=metadatas_to_upload,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    return vectorstore


def style_analysis_agent(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        "Analyze the writing style of the following text. Provide your analysis in JSON format.\n\n"
        "Text:\n\n{text}\n"
    )
    parser = JsonOutputParser(pydantic_object=StyleAnalysis)
    chain = prompt | llm | parser
    analysis = chain.invoke({"text": state['original_text']})
    
    return {"style_analysis": analysis}

def rag_style_guide_retrieval_agent(state: GraphState):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    query = f"Find a style guide for transforming text into a {state['user_goal']}"
    retrieved_docs = retriever.invoke(query)
    retrieved_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    return {"retrieved_guides": retrieved_content}

def transformation_planning_agent(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        "You are a master content strategist. Your task is to create a step-by-step plan to transform a text from one style to another.\n\n"
        "Original Text Style Analysis:\n{style_analysis}\n\n"
        "User's Goal: {user_goal}\n\n"
        "Relevant Style Guide (for reference):\n{style_guide}\n\n"
        "Based on all this information, create a clear, actionable, step-by-step plan in JSON format to rewrite the text. "
        "The plan should be a list of instructions."
        "Example Plan: ['1. Replace legal jargon with simple terms.', '2. Convert passive voice to active voice.']"
    )
    parser = JsonOutputParser(pydantic_object=TransformationPlan)
    chain = prompt | llm | parser
    plan_result = chain.invoke({
        "style_analysis": state['style_analysis'],
        "user_goal": state['user_goal'],
        "style_guide": state['retrieved_guides']
    })
    
    return {"plan": plan_result['plan']}

def content_conversion_agent(state: GraphState):
    prompt_template = (
        "You are an expert writer. Your task is to rewrite the 'Original Text' following the 'Transformation Plan' precisely.\n\n"
        "Original Text:\n\n{original_text}\n\n\n"
        "Transformation Plan:\n\n{plan}\n\n\n"
    )
    
    if state.get("qc_result") and state["qc_result"].get("feedback"):
        prompt_template += (
            "This is a revision. The previous attempt was rejected. You MUST address the following feedback:\n"
            "Quality Control Feedback:\n\n{qc_feedback}\n\n\n"
        )

    prompt_template += "Produce only the final, rewritten text. Do not add any commentary."
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    transformed_text = chain.invoke({
        "original_text": state['original_text'],
        "plan": "\n".join(state['plan']),
        "qc_feedback": state.get("qc_result", {}).get("feedback", "")
    }).content
    
    return {"transformed_text": transformed_text}

def rag_fact_check_retrieval_agent(state: GraphState):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    retrieved_docs = retriever.invoke("fact-checking protocol")
    retrieved_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    return {"retrieved_fact_check_guide": retrieved_content}

def quality_control_agent(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        "You are a meticulous Quality Control inspector. Your job is to review a content transformation based on multiple criteria.\n"
        "Compare the 'Original Text' with the 'Transformed Text' to ensure the transformation was successful and factually accurate.\n\n"
        "1. User's Goal: {user_goal}\n"
        "2. Transformation Plan:\n{plan}\n"
        "3. Fact-Checking Guide:\n{fact_checking_guide}\n\n"
        "Original Text:\n\n{original_text}\n\n\n"
        "Transformed Text:\n\n{transformed_text}\n\n\n"
        "Perform the following checks:\n"
        "- Goal Adherence: Does the transformed text meet the user's goal ('{user_goal}')?\n"
        "- Plan Execution: Was the transformation plan followed?\n"
        "- Factual Accuracy: Are all names, dates, numbers, and key facts from the original preserved correctly? (Refer to the Fact-Checking Guide).\n"
        "- Quality: Is the final output well-written and free of errors?\n\n"
        "Based on your assessment, provide a JSON output with your 'decision' ('approve' or 'revise'), 'feedback' for what to fix (if revising), and a final quality 'score' (1-10)."
    )
    parser = JsonOutputParser(pydantic_object=QualityControlResult)
    chain = prompt | llm | parser
    
    qc_result = chain.invoke({
        "original_text": state['original_text'],
        "transformed_text": state['transformed_text'],
        "user_goal": state['user_goal'],
        "plan": "\n".join(state['plan']),
        "fact_checking_guide": state['retrieved_fact_check_guide']
    })
    return {"qc_result": qc_result}

def decide_next_step(state: GraphState):
    if state['qc_result']['decision'] == 'revise':
        return "revise"
    else:
        return "approve"


workflow = StateGraph(GraphState)

workflow.add_node("style_analyzer", style_analysis_agent)
workflow.add_node("style_guide_retriever", rag_style_guide_retrieval_agent)
workflow.add_node("planner", transformation_planning_agent)
workflow.add_node("converter", content_conversion_agent)
workflow.add_node("fact_check_retriever", rag_fact_check_retrieval_agent)
workflow.add_node("quality_controller", quality_control_agent)

workflow.set_entry_point("style_analyzer")
workflow.add_edge("style_analyzer", "style_guide_retriever")
workflow.add_edge("style_guide_retriever", "planner")
workflow.add_edge("planner", "converter")
workflow.add_edge("converter", "fact_check_retriever")
workflow.add_edge("fact_check_retriever", "quality_controller")

workflow.add_conditional_edges(
    "quality_controller",
    decide_next_step,
    {
        "revise": "converter",
        "approve": END 
    }
)

app = workflow.compile()


if __name__ == "__main__":
    vectorstore = setup_rag_knowledge_base()

    original_content = (
        "Hereinafter, the party of the first part, referred to as 'the Licensor', does hereby grant and convey unto the party of the second part, "
        "referred to as 'the Licensee', a non-exclusive, non-transferable license to utilize the aforementioned intellectual property, specifically identified in "
        "Exhibit A (the 'Property'), for commercial purposes, commencing on the effective date of this agreement, which is August 15, 2024. The Licensee's "
        "utilization of the Property shall be strictly confined to the territorial boundaries of North America. Any usage outside of said territory without "
        "prior written consent from the Licensor shall be deemed a material breach of this covenant."
    )
    transformation_goal = "a simple and friendly blog post paragraph"

    print("\nExecuting the Agentic Workflow.")
    inputs = {"original_text": original_content, "user_goal": transformation_goal}

    final_state = app.invoke(inputs)
    
    final_transformed_text = final_state['transformed_text']
    final_qc_score = final_state['qc_result']['final_score']

    print(f"Final Approved Text:\n{final_transformed_text}")
    print(f"Final Quality Score: {final_qc_score}/10")