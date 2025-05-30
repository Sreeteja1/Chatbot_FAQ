from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load a semantic model for better accuracy
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your FAQ database
faq_data = [ # Corrected line!
    {"question": "How do I reset my password?", "answer": "Click 'Forgot Password' on the login screen and follow the instructions."},
    {"question": "How can I raise a support ticket?", "answer": "Login to the portal, go to the Support section, and click 'New Ticket'."},
    {"question": "What is the expected turnaround time for support?", "answer": "Turnaround time depends on issue severity, but critical issues are addressed within 2 hours."},
    {"question": "How can I check the status of my ticket?", "answer": "Go to the 'My Tickets' section in your dashboard to view status updates."},
    {"question": "Can I update a submitted ticket?", "answer": "Yes, you can reply to the email thread or update the ticket via your portal."},
    {"question": "What does 'Go-Live' mean?", "answer": "Go-Live indicates the partner's services are active and at least one successful transaction has been completed."},
    {"question": "How is ticket urgency determined?", "answer": "Our AI system categorizes tickets as High, Medium, or Low based on keywords and context."},
    {"question": "I am not receiving OTPs. What should I do?", "answer": "Please check your registered number/email, or raise a ticket if the issue persists."},
    {"question": "How do I whitelist an IP address?", "answer": "Raise a support ticket with the IP address and purpose, and our team will verify and whitelist it."},
    {"question": "I am getting a '403 Forbidden' error in API. Why?", "answer": "This usually means your IP is not whitelisted or authentication failed."},
    {"question": "What do I do if the API is returning errors?", "answer": "Check the API documentation, logs, and ensure parameters are correct. Raise a ticket if needed."},
    {"question": "Where can I find API documentation?", "answer": "You can find all API docs under the 'Documentation' section of your developer dashboard."},
    {"question": "What is the UAT environment?", "answer": "UAT (User Acceptance Testing) is the test environment for validating integrations before going live."},
    {"question": "Can I use UAT credentials in live environment?", "answer": "No, UAT and Live environments use different credentials for security reasons."},
    {"question": "I am facing issues with dashboard login.", "answer": "Try resetting your password. If still locked out, raise a support ticket."},
    {"question": "How do I integrate your payment gateway?", "answer": "Follow the steps in our API documentation and test in UAT before going live."},
    {"question": "What if my transaction failed?", "answer": "Check the response code. If funds are debited, raise a ticket with transaction ID."},
    {"question": "How can I track partner activity?", "answer": "Login to the admin dashboard and view the partner activity logs or integration reports."},
    {"question": "How do I request commercial approval?", "answer": "Submit a request through the ticketing system with all required details."},
    {"question": "Can I email support directly?", "answer": "Yes, but we recommend using the ticketing platform for faster resolution and tracking."},
    {"question": "How do I provide feedback after a ticket is resolved?", "answer": "A feedback form will be sent after ticket closure, or you can rate the ticket in your dashboard."},
    # 2xx Success Codes
    {"question": "What does HTTP 200 mean?", "answer": "HTTP 200 means the request was successful and the server returned the expected response."},
    {"question": "What does HTTP 201 mean?", "answer": "HTTP 201 means the request was successful and a new resource was created."},
    {"question": "What does HTTP 204 mean?", "answer": "HTTP 204 means the request was successful but there is no content to return."},

    # 3xx Redirection Codes
    {"question": "What does HTTP 301 error mean?", "answer": "HTTP 301 means the requested resource has been permanently moved to a new URL."},
    {"question": "What does HTTP 302 error mean?", "answer": "HTTP 302 means the requested resource has temporarily moved to a different URL."},
    {"question": "What does HTTP 304 error mean?", "answer": "HTTP 304 means the content has not changed since the last request, so the client can use the cached version."},

    # 4xx Client Errors
    {"question": "What does HTTP 400 error mean?", "answer": "HTTP 400 means a bad request. The server could not understand the request due to invalid syntax."},
    {"question": "What does HTTP 401 error mean?", "answer": "HTTP 401 means unauthorized. You need to provide valid authentication credentials."},
    {"question": "What does HTTP 403 error mean?", "answer": "HTTP 403 means forbidden. You are not allowed to access the requested resource."},
    {"question": "What does HTTP 404 error mean?", "answer": "HTTP 404 means the requested resource was not found on the server."},
    {"question": "What does HTTP 405 error mean?", "answer": "HTTP 405 means the HTTP method used is not allowed for the requested resource."},
    {"question": "What does HTTP 408 error mean?", "answer": "HTTP 408 means the server timed out waiting for the client’s request."},
    {"question": "What does HTTP 429 error mean?", "answer": "HTTP 429 means too many requests. The user has sent too many requests in a given amount of time."},

    # 5xx Server Errors
    {"question": "What does HTTP 500 error mean?", "answer": "HTTP 500 means internal server error. Something went wrong on the server."},
    {"question": "What does HTTP 501 error mean?", "answer": "HTTP 501 means not implemented. The server does not recognize or support the request method."},
    {"question": "What does HTTP 502 error mean?", "answer": "HTTP 502 means bad gateway. The server received an invalid response from an upstream server."},
    {"question": "What does HTTP 503 error mean?", "answer": "HTTP 503 means service unavailable. The server is not ready to handle the request (usually due to maintenance or overload)."},
    {"question": "What does HTTP 504 error mean?", "answer": "HTTP 504 means gateway timeout. The upstream server failed to send a request in time."}
]




# Precompute embeddings
faq_questions = [faq["question"] for faq in faq_data]
faq_embeddings = model.encode(faq_questions)

# Pydantic model for API
class QueryInput(BaseModel):
    query: str

# Shared logic
def get_faq_response(user_query: str):
    user_embedding = model.encode([user_query])
    similarities = cosine_similarity(user_embedding, faq_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < 0.6:
        return {"response": "Sorry, I couldn't find a good match. Please contact support."}

    return {"response": faq_data[best_idx]['answer'], "confidence": round(float(best_score), 2)}

# FastAPI endpoint
@app.post("/chatbot-faq")
def chatbot_faq(query: QueryInput):
    return get_faq_response(query.query)

# CLI testing
if __name__ == "__main__":
    print("🔹 FAQ Chatbot (Type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break
        result = get_faq_response(user_input)
        print("🤖 Bot:", result["response"])
        print("🔍 Confidence Score:", result.get("confidence", "N/A"))
