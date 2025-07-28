from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch

class feedbackLabel(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='give the sentiment label')


load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser() #simple parser
pydantic_parser = PydanticOutputParser(pydantic_object=feedbackLabel) #pydantic parser

prompt = PromptTemplate(
    template='based on the given feedback,  classify the text as positive or negative \n {feedback_text} \n {format_instructions}',
    input_variables=['feedback_text'],
    partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)


classifier_chain = prompt | model | pydantic_parser

classification_label = classifier_chain.invoke({'I recevived the HP laptop in damage condiiton, it is not working properly'}).sentiment


postive_prompt = PromptTemplate(
    template='Write a appropriate response to this positvie feedback \n {feedback_text}',
    input_variables=['feedback_text']
)

negative_prompt = PromptTemplate(
    template='Write a appropriate response to this negative feedback \n {feedback_text}',
    input_variables=['feedback_text']
)

response_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', postive_prompt | model | parser),
    (lambda x: x.sentiment == 'negative', negative_prompt | model | parser),
    RunnableLambda(lambda x: f'No response needed for {x.sentiment} feedback')
)

final_chain = classifier_chain | response_chain

#print(final_chain.invoke({'feedback_text': 'I recevived the HP laptop in damage condiiton, it is not working properly'}))
print(final_chain.invoke({'feedback_text': 'I got the my cloth very nicly packed, it is very good quality and I am happy with the product'}))