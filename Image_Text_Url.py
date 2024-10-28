import requests
import openai
import json
import time

# Check OpenAI library version and set up client accordingly
if hasattr(openai, 'OpenAI'):
    client = openai.OpenAI(api_key="Your API Key")
else:
    openai.api_key = "Your API Key"
    client = openai

def extract_text_and_analyze_image(image_url):
    """Extract text and analyze the image using GPT-4 Vision."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key if hasattr(client, 'api_key') else client.api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image. Extract all visible text. If it's a flowchart, graph, or diagram, describe the relationships and structure. Provide a detailed description of the content. Format your response as a JSON object with keys for 'extracted_text', 'structure_description', and 'content_description'."
                    },
                    {
                        "type": "image_url",
                        "image_url": image_url
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        # Try to parse the content as JSON, if it fails, return it as is
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_content": content}
    except requests.exceptions.Timeout:
        return {"error": "The request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred while analyzing the image: {str(e)}"}

def answer_question(question, image_analysis):
    """Use ChatGPT to answer a question based on the image analysis."""
    try:
        if hasattr(client, 'chat'):
            # New version of the OpenAI library
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given image analysis."},
                    {"role": "user", "content": f"Image Analysis:\n{json.dumps(image_analysis, indent=2)}\n\nQuestion: {question}\n\nAnswer:"}
                ],
                timeout=30
            )
            return response.choices[0].message.content.strip()
        else:
            # Old version of the OpenAI library
            response = client.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given image analysis."},
                    {"role": "user", "content": f"Image Analysis:\n{json.dumps(image_analysis, indent=2)}\n\nQuestion: {question}\n\nAnswer:"}
                ],
                request_timeout=30
            )
            return response.choices[0].message['content'].strip()
    except openai.error.Timeout:
        return "Error: The request timed out. Please try again."
    except openai.error.APIError as e:
        return f"Error: An API error occurred: {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"

def main():
    image_url = input("Enter the URL of the image: ")
    
    print("Analyzing the image... This may take a moment.")
    image_analysis = extract_text_and_analyze_image(image_url)
    print("\nImage Analysis:")
    print(json.dumps(image_analysis, indent=2))
    print("\nImage analysis complete. You can now ask a question about the image.")
    
    question = input("\nAsk a question about the image (or press Enter to exit): ")
    
    if not question.strip():
        print("\nNo question asked. Exiting the program. Goodbye!")
        return

    print("Processing your question...")
    start_time = time.time()
    answer = answer_question(question, image_analysis)
    end_time = time.time()
    
    print(f"\nAnswer: {answer}")
    print(f"Time taken to answer: {end_time - start_time:.2f} seconds")

    print("\nExiting the program. Goodbye!")

if __name__ == "__main__":
    main()
