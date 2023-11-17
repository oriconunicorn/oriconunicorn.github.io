# load core modules
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
# load agents and tools modules
import pandas as pd
from azure.storage.filedatalake import DataLakeServiceClient
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain

# initialize pinecone client and connect to pinecone index
pinecone.init(
        api_key="5fc9a612-0250-4bfa-a0b5-4dfe60e95542",  
        environment="gcp-starter"  
) 

index_name = 'fd'
index = pinecone.Index(index_name) # connect to pinecone index

# initialize embeddings object; for use with user query/input
embed = OpenAIEmbeddings(
                model = 'text-embedding-ada-002',
                openai_api_key="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",
            )

# initialize langchain vectorstore(pinecone) object
text_field = 'text' # key of dict that stores the text metadata in the index
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

llm = ChatOpenAI(    
    openai_api_key="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", 
    model_name="gpt-3.5-turbo", 
    temperature=0.0
    )

# initialize vectorstore retriever object
timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

df = pd.read_csv("feedback1.csv") # load employee_data.csv as dataframe
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts
user = 'Jane Smith' # set user
df_columns = df.columns.to_list() # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name = "Feedback Data",
        func=timekeeping_policy.run,
        description="""
        Useful for when you need to answer questions about user feedbacks.

        <user>: What is the review of the product?
        <assistant>: I need to check the feedback data to answer this question.
        <assistant>: Action: Feedback Data
        <assistant>: Action Input: Feedback Data - Review
        ...
        """
    ),
    Tool(
        name = "Feedback",
        func=python.run,
        description = f"""
        Useful for when you need to answer questions about feedback data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many stars do I rate for this product?
        <assistant>: df[df['name'] == '{user}']['Rating']
        <assistant>: You give n stars for rating.              
        """
    ),
    Tool(
        name = "Calculator",
        func=calculator.run,
        description = f"""
        Useful when you need to do math operations or arithmetic.
          """
    ),
    Tool(
        name = "Recommendation",
        func=python.run,
        description = f"""
        Summarize the labels of the feedback data stored in pandas dataframe 'df' and give recommendations on how to prioritize the labels, with user name attached.
        Based on the recommendation report, help the user to make decision on which labels to focus for product iteration.
        Run python pandas operations on 'df' and create a new colomun to store your answer.
        Provide a list of labels after summarizing the contents.
        'df' has the following columns: {df_columns}
        
        <user>: What kind of labels are there based on labels?
        <assistant>: df[df['name'] == '{user}']['Label']
        <assistant>: Labels are summarized as following: [].              
        """
    )
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


# initialize the LLM agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs
                         )
# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response