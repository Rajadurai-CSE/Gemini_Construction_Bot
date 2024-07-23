from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from configure import configure
configure()

def get_conversational_chain():

    # model = genai.GenerativeModel(model='gemini-1.5-flash')

   

    model = ChatGoogleGenerativeAI(model ='gemini-1.5-flash',temperature=0.5,convert_system_message_to_human=True)
    # model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system',' You are a construction expert. Your work is to assist contractors, architects , general people who are interested in buying home . Your task will be from providing insights on budgets of new projects, aiding architects in designing to guiding people to buy new home . Provide the audience with relavant information from the given context. You will also be given images which you can analyze and answer. If you feel you are not able to fully understand the context or image, then just say i am not able to answer and provide some alternatives. Be precise, clear to the audience. If there is not context given or image given, then try to answer the question with your general knowledge related to construction Industry, be a friendly answering casual questions but not to questions that are unappropriate to construction industry'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human','\n{input}')
            
            ]
    )
    
    memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=3
    )

    chain = LLMChain(
    llm=model,
    prompt=prompt_template,
    memory=memory,
    verbose=True
    )
    # prompt = prompt_template.format(context=context, question=question)
    # chat = model.generate_content(prompt)
    # # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # conversation_with_summary = ConversationChain(
    # llm=model,
    # prompt=prompt,
    # memory=memory,
    # verbose=True
    # )
    return chain

#Image Generator Function to Generate Image


