import os
import streamlit as st
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. DATABASE CONFIGURATION (Use your free Neon/Aiven URL here)
DB_CONNECTION_STRING = st.secrets.get("DB_URL", "postgresql://user:pass@ep-free-db.neon.tech/neondb")

# Mock User Credentials Database (Replace with a production OAuth/User DB later)
USER_DATABASE = {
    "agent_sarah": "florida2026",
    "agent_john": "miamihomes"
}

SYSTEM_PERSONA = """You are a licensed Florida Real Estate AI Assistant. Your job is to help users navigate the Florida housing market. You must adhere strictly to the rules of the Florida Department of Business and Professional Regulation (DBPR) and the Florida Real Estate Commission (FREC).

CRITICAL INSTRUCTIONS:
1. LEGAL BOUNDARIES: Always include a brief standard disclaimer stating you are an AI, not a human attorney or licensed human broker. Never draft binding contracts without human broker oversight.
2. FAIR HOUSING MANDATE: You are strictly forbidden from describing neighborhood demographics, race, religion, schools, or commenting on whether an area is "safe" or "unsafe" / "good" or "bad". These topics constitute illegal steering under the Federal Fair Housing Act. If asked about these, politely refuse to answer.
3. FLORIDA SPECIFICS: Base your contract answers on the standard FAR-BAR As-Is Contract and Florida's Johnson v. Davis seller disclosure requirements.
4. OUT OF BOUNDS: If a query requires specialized tax or legal advice, instruct the user to consult a licensed Florida professional."""

# Initialize Vector Base (RAG)
@st.cache_resource
def initialize_vector_store():
    florida_legal_documents = [
        "Florida Statutes Chapter 475: Regulates real estate brokers, sales associates, and schools. Real estate licenses must be renewed every two years through the DBPR.",
        "FAR-BAR As-Is Contract Inspection Period: Standard provision allows the buyer 15 days (unless modified) to conduct inspections and cancel the contract for any reason, obtaining a full refund of their earnest money deposit.",
        "Johnson v. Davis (Florida Supreme Court): The seller has an absolute duty to disclose all known latent material defects that materially affect the value of residential property and are not readily observable to the buyer."
    ]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_texts(texts=florida_legal_documents, embedding=embeddings, collection_name="fl_law")

# 2. LIVE RESO WEB API INTEGRATION ENGINES
def fetch_live_reso_mls_data(listing_id: str) -> str:
    """Connects directly to an MLS endpoint using standard OData RESO Web API protocols."""
    if not listing_id:
        return "No explicit listing or property requested."
        
    # Replace with your actual real estate board's OData URL (Stellar MLS, Miami Realtors, etc.)
    reso_endpoint = "https://mlsgrid.com" 
    access_token = st.secrets.get("RESO_API_TOKEN", "MOCK_TOKEN")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    # Query parameters conforming to standard RESO Web API specifications
    params = {
        "$filter": f"ListingId eq '{listing_id}'",
        "$select": "ListPrice,BedroomsTotal,BathroomsFull,ModificationTimestamp,PropertySubtype,ListAgentFullName"
    }
    
    try:
        # Short timeout to keep the conversational agent responsive
        response = requests.get(reso_endpoint, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("value"):
                listing = data["value"][0]
                return f"RESO MLS Verified Info: Price ${listing.get('ListPrice')}, Beds: {listing.get('BedroomsTotal')}, Baths: {listing.get('BathroomsFull')}. Status fetched via live Web API."
        return f"[RESO API Mode] Listing ID '{listing_id}' not found in active regional database records."
    except Exception:
        # Fallback simulation if tokens are not configured yet
        return f"[Offline Simulation] MLS ID '{listing_id}': Active Listing. Price $550,000. 3 Bed, 2 Bath located in Miami-Dade County."

# Assemble LangChain flow with database persistence
def build_agent_executor():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PERSONA),
        ("system", "Context from Florida Real Estate Law:\n{context}\n\nLive Property Data Matrix:\n{live_data}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    core_chain = (
        {
            "context": lambda x: "\n\n".join(doc.page_content for doc in retriever.invoke(x["question"])),
            "live_data": lambda x: fetch_live_reso_mls_data(x["property_context"]),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"]
        }

        | prompt_template | llm | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        core_chain,
        get_session_history=lambda session_id: PostgresChatMessageHistory(
            connection_string=DB_CONNECTION_STRING, session_id=session_id
        ),
        input_messages_key="question", history_messages_key="history"
    )

# 3. STREAMLIT MULTI-USER LOGIN SYSTEM
st.set_page_config(page_title="Florida AI Agent Portal", page_icon="🏠")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""

if not st.session_state["authenticated"]:
    st.title("🔒 Real Estate Agent Login Portal")
    username_input = st.text_input("Username / Agent ID")
    password_input = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username_input in USER_DATABASE and USER_DATABASE[username_input] == password_input:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username_input
            st.success(f"Logged in as {username_input}!")
            st.rerun()
        else:
            st.error("Invalid Username or Password.")
    st.stop()

# --- MAIN APP REGION (Visible Only to Authenticated Users) ---
st.title(f"🏠 Florida Real Estate Assistant Portal")
st.caption(f"Secure Session Active: {st.session_state['username']}")

with st.sidebar:
    st.header("Search Parameters")
    target_mls_id = st.text_input("Target MLS Listing ID", value="A1234567")
    if st.button("Log Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()

agent_executor = build_agent_executor()

# Establish localized database message history for logged-in user identity
db_history = PostgresChatMessageHistory(
    connection_string=DB_CONNECTION_STRING,
    session_id=st.session_state["username"]
)

# Render past data pulled securely from the SQL Cloud Server
for message in db_history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if user_query := st.chat_input("Ask a question regarding current property details or state laws..."):
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Processing transaction history rules..."):
            response = agent_executor.invoke(
                {"question": user_query, "property_context": target_mls_id},
                config={"configurable": {"session_id": st.session_state["username"]}}
            )
            st.markdown(response)
