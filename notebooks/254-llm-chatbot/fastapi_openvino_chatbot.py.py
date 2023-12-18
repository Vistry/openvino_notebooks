from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from threading import Thread
from transformers import AutoConfig, AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM
import json

# Define Pydantic models for request and response
class ChatRequest(BaseModel):
    system: str
    user: str

# Load the model and tokenizer
model_dir = 'neural-chat-7b-v3-2/INT4_compressed_weights'

# Model configuration
ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device='GPU',
    ov_config={'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', 'CACHE_DIR': ''},
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

# Tokenizer configuration
tok = AutoTokenizer.from_pretrained("Intel/neural-chat-7b-v3-2", trust_remote_code=True)

# Define ImmediateTextStreamer for text streaming
class ImmediateTextStreamer(TextIteratorStreamer):

    def put(self, value):
        # Validate input shape
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        # Handle prompt skipping
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Extend token cache and handle text streaming
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        printable_text = text[self.print_len :]
        self.print_len += len(printable_text)
        self.on_finalized_text(printable_text)

streamer = ImmediateTextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

# Tokenizer arguments
tokenizer_kwargs = {"add_special_tokens": False}

# Function to generate chat responses
def generate_response(system, user):
    # Prepare input text
    text = f"""### System:
{system}
### User:
{user}
### Assistant:
"""
    input_tokens = tok(text, return_tensors="pt", **tokenizer_kwargs)

    # Generation configuration
    generation_kwargs = dict(
        **input_tokens,
        max_new_tokens=1000,
        streamer=streamer,
        pad_token_id=tok.eos_token_id,
        do_sample=False
    )

    # Start response generation in a separate thread
    thread = Thread(target=ov_model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the generated responses
    for new_text in streamer:
        response_data = {"choices": [{"delta": {"content": new_text}}]}
        yield json.dumps(response_data) + '\n'  # Convert to JSON and yield

# Create FastAPI application
app = FastAPI()

# Define API endpoint for chat
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response_stream = generate_response(request.system, request.user)
    return StreamingResponse(response_stream, media_type="application/json")

# Run the API using a command like: uvicorn fastapi_openvino_chatbot:app --reload