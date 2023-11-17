## Autonomous HR Chatbot built using ChatGPT, LangChain, Pinecone and Streamlit


---
## Instructions
---

I made this prototype using Azure deployments as my company is an Azure customer.  
I created a backend file called `hr_agent_backend_local.py` for those that does not want to use Azure.  
This is does not use any Azure components - the API is from platform.openai.com, the csv file is stored locally(i.e. on your own computer)

### How to use this repo

1. Install python 3.10. [Windows](https://www.tomshardware.com/how-to/install-python-on-windows-10-and-11#:~:text=1.,and%20download%20the%20Windows%20installer.&text=2.,is%20added%20to%20your%20path.), [Mac](https://www.codingforentrepreneurs.com/guides/install-python-on-macos/) 
2. Clone the repo to a local directory.
3. Navigate to the local directory and run this command in your terminal to install all prerequisite modules - `pip install -r requirements.txt`
4. Input your own API keys in the `hr_agent_backend_local.py` file (or `hr_agent_backend_azure.py` if you want to use the azure version; just uncomment it in the frontend.py file)
5. Run `streamlit run hr_agent_frontent.py` in your terminal

### Storing Embeddings in Pinecone

1. Create a Pinecone account in [pinecone.io](pinecone.io) - there is a free tier.  Take note of the Pinecone API and environment values.
2. Run the notebook 'store_embeddings_in_pinecone.ipynb'. Replace the Pinecone and OpenAI API keys (for  the embedding model) with your own.


---
  

## Video Demo 

![feedback_chatbot](https://github.com/oriconunicorn/feedback_bot/assets/89826444/25aefe44-b452-4b25-8c24-7e6d72f17ab2)


https://github.com/oriconunicorn/feedback_bot/assets/89826444/2a85eaf8-3232-4056-bc61-32c247819a48



https://github.com/oriconunicorn/feedback_bot/assets/89826444/340db9dd-aeb4-4257-9f24-2c9569d989ec


