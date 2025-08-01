﻿🤖 LangGraph Content Transformation Agent
This project is a sophisticated agent built with LangGraph that automates the process of rewriting text from one style to another. It follows a multi-step workflow including style analysis, knowledge retrieval (RAG), planning, content generation, and a quality control loop to ensure the final output meets the user's goal.


✨ Features
    Style Analysis: Automatically analyzes the tone, complexity, format, and vocabulary of the source text.
    RAG for Guidance: Retrieves relevant style guides from a knowledge base (Pinecone) to inform the transformation.
    Strategic Planning: Generates a step-by-step plan for rewriting the content.
    Content Conversion: Rewrites the text according to the generated plan.
    Quality Control Loop: A dedicated QC agent reviews the transformed text against the original, the plan, and a fact-checking protocol. If the quality is not sufficient, it sends the text back for revision with specific feedback.


🚀 Setup and Installation
Follow these steps carefully to get the agent running on your local machine.

1. Prerequisites
Python 3.8 or newer.
An OpenAI API Key.
A Pinecone API Key and environment name from your Pinecone project dashboard.

2. Clone the Repository
If your project is in a git repository, clone it. Otherwise, simply ensure you have the project files in a dedicated folder.
git clone <your-repository-url>
cd <your-repository-name>



3. Create a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


4. Install Dependencies
Create a file named requirements.txt in your project folder and add the following lines:
langchain
langchain-openai
langgraph
python-dotenv
pinecone-client
langchain-pinecone
pydantic


Now, install these packages using pip:
pip install -r requirements.txt



5. Configure Environment Variables
Create a file named .env in the root of your project directory. This file will securely store your API keys. Do not commit this file to version control. 🔒
Open the .env file and add your keys:
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="your-pinecone-api-key"

⚠️ Important Note on OpenAI Base URL
The code uses a custom base_url for the OpenAI API (https://api.us.inc/usf/v1/hiring). If you are using the standard OpenAI API, you may need to remove the base_url and default_headers arguments from the ChatOpenAI constructor in your script.


6. Create the Knowledge Base
The agent relies on a local "knowledge base" of text files for its RAG capabilities.
Create a folder named knowledge_base in the root of your project directory.
Inside the knowledge_base folder, create at least two .txt files. The agent is specifically designed to look for a style guide and a fact-checking protocol.
Example: style-guide-blog-post.txt
Style Guide for a Simple and Friendly Blog Post:
- Use a conversational and approachable tone.
- Write in the active voice. Avoid passive constructions.
- Break down complex sentences into shorter, simpler ones.
- Replace jargon and formal language with everyday words.
- Use contractions like "you're" or "it's" to sound more natural.
- Start with a hook to grab the reader's attention.

Example: fact-checking-protocol.txt
Fact-Checking Protocol:
1. Verify all proper nouns (names of people, companies, places).
2. Double-check all numbers, statistics, and monetary values.
3. Confirm all dates and times are transcribed correctly.
4. Ensure that the core meaning and key facts of the original text are preserved without alteration.


▶️ How to Run the Agent
Once you have completed all the setup steps, you can run the agent by executing the Python script from your terminal.
Make sure your virtual environment is activated.
Run the script:
python your_script_name.py
Use code with caution.

(Replace your_script_name.py with the actual name of your Python file.)
The script will then:
Set up the Pinecone index and upload the documents from your knowledge_base folder.
Execute the agentic workflow with the sample content defined in the script.
Print the final, approved transformed text and its quality score to the console.


🛠️ Customization
To transform your own text, simply edit the variables at the bottom of the script inside the if __name__ == "__main__": block:
if __name__ == "__main__":
    vectorstore = setup_rag_knowledge_base()

    # --- EDIT THESE VALUES ---
    original_content = (
        "Put your own source text here. It can be long and complex."
    )
    transformation_goal = "describe your desired output style, e.g., a professional email"
    # -------------------------

    print("\nExecuting the Agentic Workflow.")
    inputs = {"original_text": original_content, "user_goal": transformation_goal}
    # ... rest of the script

KEY ENHANCEMENTS IN THE CODE
    Add a Revision Limit: To prevent infinite loops, I would add a revision_count to the GraphState. The decide_next_step function would then check this count and force an exit to a failure state after 2-3 revisions, preventing wasted computation if the agent gets stuck.
    Implement a RAG Re-ranker: The initial RAG retrieval is good, but the most relevant document isn't always the top result. I would add a re-ranking step (e.g., using CohereRerank or a cross-encoder model) after rag_style_guide_retrieval_agent to ensure the planner gets the most pertinent style guide possible.
    Add Observability: I would integrate the graph with an observability tool like LangSmith. This would provide invaluable traces to debug why a specific revision was needed, inspect the intermediate inputs/outputs of each agent, and analyze performance over time.
    
