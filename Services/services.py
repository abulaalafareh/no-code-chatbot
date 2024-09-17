import random
import time


# Function to simulate streaming response
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi! Is there anything I can help you with?",
            "Do you need assistance with something?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.1)  # Adjust the sleep time for faster or slower typing effect