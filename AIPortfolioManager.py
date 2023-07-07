import streamlit as st, pandas as pd, datetime as dt
import plotly.express as px

import streamlit_authenticator as stauth
import pickle

import openai
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI
import os

from pydantic import ValidationError
from streamlit_chat import message

st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="✅",
    layout="wide",
)
hide_default_format = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_default_format, unsafe_allow_html=True)


def show_api_key_missing():
    """
    Displays a message if the user has not entered an API key
    """
    st.markdown(
        """
        <div style='text-align: center;'>
            <h5>Enter your <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API key</a> to start chatting</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_knowledgebase(df, agent_path, temperature):
    # csv_memory = ConversationBufferMemory()

    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=temperature),
        df,
        verbose=False,
    )

    # agent_file = open(agent_path, 'ab')
    # pickle.dump(agent, agent_file)
    return agent


def load_model(agent_path):
    agent_file = open(agent_path, "rb")
    agent = pickle.load(agent_file)
    return agent


def save_output(output,df):
    try:
        companies = df['Stock Name'].tolist()
        output = eval(output)
        if all(x in companies for x in output):
          df = pd.DataFrame(output, columns=["companies"])
          df.to_csv("chat_df.csv")
          return True
        else:
            return False
    except Exception as e:
        return False


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


def load_api_key():
    """
    Loads the OpenAI API key from the .env file or
    from the user's input and returns it
    """
    if not hasattr(st.session_state, "api_key"):
        st.session_state.api_key = None
    # you can define your API key in .env directly
    if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
        user_api_key = os.environ["OPENAI_API_KEY"]
        st.success("API key loaded from .env", icon="🚀")
    else:
        if st.session_state.api_key is not None:
            user_api_key = st.session_state.api_key
            st.success("API key loaded from previous input", icon="🚀")
        else:
            user_api_key = st.text_input(
                label="#### Your OpenAI API key 👇",
                placeholder="sk-...",
                type="password",
            )
            if user_api_key:
                st.session_state.api_key = user_api_key

    return user_api_key


def chatbot():
    user_api_key = load_api_key()
    os.environ["OPENAI_API_KEY"] = user_api_key

    if not user_api_key:
        show_api_key_missing()

    def get_agent(df):
        # csv_file = 'finance.csv'
        agent_path = "agent_knowledge_base"
        temperature = 0.0
        agent = create_knowledgebase(df, agent_path, temperature)
        # agent = load_model(agent_path)
        return agent

    def save_output_to_csv(df, output):
        try:
            selected_rows = df[df["Stock Name"].isin(eval(output))]
            pd.DataFrame(selected_rows).to_csv("prompt_df.csv")
            return True
        except Exception as e:
            print("Exception occured", e)
            return False

    def get_answer(input_question, action):
        
        if action=='reset':
            df = pd.read_csv("finance.csv")
        else:
            if os.path.exists("prompt_df.csv"):
                df = pd.read_csv("prompt_df.csv")
            else:
                df = pd.read_csv("finance.csv")
                
        agent = get_agent(df)
        try:
          output_return = agent.run(input_question)
          output_stock_filter = agent.run(input_question + ". convert to python list")  
        except Exception as e:
            output_return='I didnt get you. Can you please come again.'
            output_stock_filter="['error']"
        
        print("output is ", output_stock_filter)
        retval = save_output(output_stock_filter,df)
        return df, output_return, output_stock_filter,retval

    def get_text():
        input_text = st.text_input("Please enter your query.", key="input")
        return input_text

    # with right_column:
    st.write("Chatbot")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    user_input = get_text()

    if user_input:
        action = ""
        if st.button("Reset"):
            if os.path.exists("prompt_df.csv"):
                os.remove("prompt_df.csv")
            action = "reset"
        
        df, output, output_stock_filter,filter_flag = get_answer(user_input, action)
        if filter_flag==True:
          save_output_to_csv(df, output_stock_filter)
          st.session_state.past.append(user_input)
          st.session_state.generated.append(output)
        else:
           st.session_state.past.append(user_input)
           st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def get_portfolio_df(input_df, input_companies, start_date=None, end_date=None):
    # filter input DataFrame to only include selected companies
    included_df = input_df[input_df["stock_name"].isin(list(input_companies))]
    if start_date and end_date:
        # filter data between start and end date
        included_df["Date"] = pd.to_datetime(
            included_df["Date"], format="%d/%m/%Y"
        ).dt.strftime("%Y-%m-%d")
        included_df = included_df[
            (included_df["Date"] >= str(start_date))
            & (included_df["Date"] <= str(end_date))
        ]

    # group by date and stock symbol, and sum the total close values
    grouped_df = included_df.groupby(["Date"]).sum().reset_index()
    grouped_df = grouped_df.sort_values(by="Date")
    grouped_df = grouped_df[["Date", "Adj Close"]]
    num_stocks = 10000 / grouped_df["Adj Close"].to_list()[0]
    grouped_df["Adj Close"] = grouped_df["Adj Close"] * num_stocks
    output_df = grouped_df.sort_values(by="Date")
    return output_df


def dashboard():
    df = pd.read_csv("finance.csv")
    chat_df = pd.DataFrame()
    chat_df_companies = []
    if os.path.exists("chat_df.csv"):
        chat_df = pd.read_csv("chat_df.csv")
    if chat_df.any().any() and len(chat_df.companies):
        chat_df_companies = list(chat_df.companies)

    dropdown = df["Stock Name"]

    st.title("Dashboard")

    print("default is ", chat_df_companies)
    option = st.multiselect(
        "Please select stocks",
        dropdown,
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        default=chat_df_companies,
    )

    col_1, col_2 = st.columns(2)
    with col_1:
        start_date = st.date_input(
            "Start date", dt.date.today() - dt.timedelta(days=395)
        )
    with col_2:
        end_date = st.date_input("End date", dt.date.today())
    stocks_df = pd.read_csv("stocks.csv")

    company_final = option
    if option:
        output_df = get_portfolio_df(stocks_df, company_final, start_date, end_date)
        print(output_df)
        nifty = get_portfolio_df(stocks_df, ["HDFC Bank Ltd."], start_date, end_date)
        output_df = output_df.rename(columns={"Adj Close": "Portfolio Adj Close"})
        data = pd.concat(
            [output_df["Portfolio Adj Close"], nifty["Adj Close"], output_df["Date"]],
            axis=1,
        )
        # Portfolio Adj Close,Adj Close

        fig = px.line(data, x="Date", y=["Portfolio Adj Close", "Adj Close"])
        st.plotly_chart(fig, use_container_width=True)

        # Creating Price Movements table
        st.header("Price Movements")

        included_df = df[df["Stock Name"].isin(option)]
        st.write(included_df)
    else:
        st.write("Please select companies")


col_1, col_2 = st.columns((1, 2))
with col_1:
    chatbot()
with col_2:
    dashboard()
