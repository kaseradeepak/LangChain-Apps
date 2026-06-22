from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings

# 1. HR Policy documents.
HR_DOCUMENTS = [
    Document(
        page_content=(
            "Before Day 1, every new employee must complete identity verification, "
            "bank account details, NDA signing, emergency contact update, and tax declaration. "
            "Remote employees must also confirm their delivery address for IT equipment."
        ),
        metadata={"source": "HR-POL-001", "section": "Pre-Joining Checklist"},
    ),
    Document(
        page_content=(
            "Full-time employees receive 24 paid leaves per calendar year. "
            "Leave requests must be submitted through the HR portal. "
            "During the first 90 days, planned leaves require manager approval."
        ),
        metadata={"source": "HR-POL-002", "section": "Leave Policy"},
    ),
    Document(
        page_content=(
            "All new employees must complete Security Awareness, Code of Conduct, "
            "Data Privacy, and Anti-Harassment training within the first 7 working days."
        ),
        metadata={"source": "HR-POL-003", "section": "Mandatory Training"},
    ),
    Document(
        page_content=(
            "Health insurance enrollment must be completed within 15 days of joining. "
            "Employees can add spouse and up to two children as dependents. "
            "Parents require additional premium contribution."
        ),
        metadata={"source": "HR-POL-004", "section": "Benefits and Insurance"},
    ),
    Document(
        page_content=(
            "IT equipment for remote employees includes laptop, charger, headset, "
            "VPN access, email access, and collaboration tool access. "
            "Standard SLA for equipment delivery is 5 business days after address confirmation."
        ),
        metadata={"source": "IT-SOP-001", "section": "Remote Employee Setup"},
    ),
]

# 2. Employees database
EMPLOYEES: Dict[str, Dict] = {
    "E101": {
        "name": "Aarav Sharma",
        "role": "Backend Engineer",
        "location": "Bangalore",
        "work_mode": "remote",
        "joining_date": "2026-07-01",
        "manager": "Meera Iyer",
    },
    "E102": {
        "name": "Neha Verma",
        "role": "Product Manager",
        "location": "Delhi",
        "work_mode": "hybrid",
        "joining_date": "2026-07-03",
        "manager": "Rahul Khanna",
    },
}

# 3. Embedding Model
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(HR_DOCUMENTS)

retriever = vector_store.as_retriever(search_kwargs = {"k" : 3}) # top-3 matching documents

# 4. Tool for searching HR policies.
@tool
def search_hr_policy(query:  str):
    """
    Search HR policy documents and return the relevant policy documents for the user query.
    Use this tool whern the user asks about onboarding, leave policy, benefits, insurance, trainings etc.
    """

    docs = retriever.invoke(query)

    if not docs:
        return f"No relevant policy document found for the query: {query}"
    
    results = []
    for doc in docs:
        source = doc.metadata.get('source', 'UNKNOWN')
        section = doc.metadata.get('section', 'UNKNOWN')
        doc_content = doc.page_content
        results.append(
            f"{source} | {section} | {doc_content}"
        )
        
    return results

@tool
def get_employee_profile(employee_id: str):
    """
    Tool to get the employee information for the given employee id.
    Use this tool to fetch employee information from database.
    """

    employee = EMPLOYEES.get(employee_id)

    if not employee:
        return {
            "found" : False,
            "message" : f"Employee with id: {employee_id} not found."
        }
    
    return {
        "found" : True,
        "employee_id" : employee_id,
        **employee
    }

@tool
def get_training_recommendations(role: str):
    """
    Get the mandatory training recommendations for a new employee based on their role.
    """

    common_trainings = [
        "Security",
        "Code of Conduct",
        "Data Privacy",
        "Anti-Harassment Training"
    ]

    role_specific_trainings = {
        "Backend Engineer" : ['Python', 'FastAPI', 'Agentic AI'],
        "Product Manager" : ['Client Interaction', 'Client Data Handling']
    }

    return {
        "mandatory trainings" : common_trainings,
        "role specific trainings" : role_specific_trainings.get(role)
    }