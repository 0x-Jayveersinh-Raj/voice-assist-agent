from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)


# # Set up API key
# API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")

# # Initialize the Gen AI client
# client = genai.Client(api_key=API_KEY)

# # Define the model and prompt
# model = "gemini-1.5-flash"
# prompt = "Write a short haiku about the sunrise."

# # Create the generation configuration
# generation_config = types.GenerateContentConfig(
#     max_output_tokens=64
# )

# # Send the request
# response = client.models.generate_content(
#     model=model,
#     contents=[{"role": "user", "parts": [{"text": prompt}]}],
#     config=generation_config
# )

# # Output the response
print("Generated text:", response.text)
