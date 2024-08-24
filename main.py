import streamlit as st
import requests
import json

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant. You are knowledgeable, polite, and always try to provide accurate and useful information."""

def generate_llama_response_stream(prompt, conversation_history):
    # Prepare the full conversation context
    full_prompt = f"System: {SYSTEM_PROMPT}\n" + "\n".join(conversation_history + [f"Human: {prompt}", "Assistant:"])
    
    # Prepare the request payload
    payload = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": True
    }

    # Send request to Ollama API
    with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get('response', '')
        else:
            yield f"Error: Unable to generate response. Status code: {response.status_code}"

def main():
    st.title("Chat with Llama 3 (Streaming)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add a button to reset the conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get conversation history for context
        conversation_history = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[:-1]]

        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream the response
            for response_chunk in generate_llama_response_stream(prompt, conversation_history):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()