import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
#random: Python’s built-in module for generating random numbers and choices.
#json: Python’s built-in module for parsing JSON data.
#torch: The main module of the PyTorch library, used for tensor computations and deep learning.
#NeuralNet: A custom neural network model class imported from the model module.
#bag_of_words and tokenize: Utility functions imported from the nltk_utils module for text processing.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.device:torch.device is a PyTorch class that represents the device on which a tensor will be allocated or a computation 
# will be performed. It can either be a CPU or a CUDA device (GPU).
# torch.cuda.is_available() is a function that checks if CUDA (and hence GPU support) is available on the machine. 
# If a compatible GPU is installed and the necessary drivers and CUDA toolkit are properly configured, this function returns True; otherwise, it returns False 
# CUDA is a parallel computing platform and API model that allows for using NVIDIA GPUs for general-purpose computing.


with open('intents1.json', 'r') as json_data:
    intents = json.load(json_data)
#opens and reads JSON data from the file 'intents1.json'. 
# It utilizes Python's 'with' statement for efficient file handling, ensuring proper resource management throughout. 
# The parsed JSON content is then stored in the variable 'intents' as a Python dictionary. 


FILE = "data.pth"
data = torch.load(FILE)
print(data)
#initializes a variable named FILE with the string "data.pth", representing the file path or name. (trained data will be loaded)
# It then uses PyTorch's torch.load() function to load data from the file specified by FILE.

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
#variables assigned to extract specific information from data file

#neuralnet allows to recognize patterns, make decisions, and solve problems, similar to how we learn and adapt.
model = NeuralNet(input_size, hidden_size, output_size).to(device)
#The line model = NeuralNet(input_size, hidden_size, output_size).to(device) initializes a neural network (model) 
# with specified input, hidden layer, and output sizes, 
# and assigns it to a computational device (GPU or CPU) determined by device.


model.load_state_dict(model_state)
#model.load_state_dict(model_state) loads previously trained parameters (weights, biases) 
# from model_state into the model, ensuring it starts with its learned configuration.

model.eval()
#model.eval() sets the model to evaluation mode, adjusting certain layers for deterministic behavior during predictions. 

# code sets up a neural network model with specific sizes for its input, hidden layers, and output.
# It then loads previously trained settings into this model to start with what it has learned. 
# After that, it switches the model to evaluation mode, which means it's ready to make predictions accurately when given new information. 
# This prepares the neural network to work well and give reliable results in real-life situations
# where it needs to process and predict outcomes based on data.

bot_name = "ONGC"

def find_response(intent_data, tag):
    if 'tag' in intent_data and intent_data['tag'] == tag:
        return random.choice(intent_data['responses'])
    for key, value in intent_data.items():
        if isinstance(value, list):
            for sub_intent_data in value:
                if isinstance(sub_intent_data, dict):
                    response = find_response(sub_intent_data, tag)
                    if response:
                        return response
    return None
#his function is commonly used in chatbots or systems that handle conversations. 
# It looks through organized data that links types of messages (like greetings or questions) to pre-decided replies.
# By searching through this structured data, it finds the right reply for a specific type of message, 
# making sure the chatbot can answer correctly to different things people might say.


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    #X = X.reshape(1, X.shape[0]): Reshapes the numerical representation (X) to match the expected input shape for the neural network model.
    X = torch.from_numpy(X).to(device)
    #Converts the numerical representation (X) into a PyTorch tensor (torch.Tensor) and
    # moves it to a specified computational device (device), such as a GPU or CPU.

    output = model(X) #Feeds the tensor (X) into a pre-trained neural network model (model) to obtain predictions (output).
    _, predicted = torch.max(output, dim=1)  #Identifies the index of the highest predicted value in the output tensor, indicating the predicted category or tag

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1) #Softmax is a mathematical function that converts a vector of numerical values into a 
    #probability distribution. 
    # Calculates the softmax probabilities across the output tensor, providing confidence scores for each predicted category.
    prob = probs[0][predicted.item()]
    # Retrieves the probability score associated with the predicted category.
    if prob.item() > 0.75:
        for intent_category in intents.values():
            for intent in intent_category:
                response = find_response(intent, tag)
                if response:
                    return f"{response} (Probability: {prob.item():.2f})"
    
    return "I do not understand... (Probability: Low)"
#Checks if the probability (prob) exceeds a threshold (0.75 in this case).
#If the probability is high enough:
#Iterates through predefined intents (intents) to find a suitable response using the find_response function, based on the predicted tag (tag).
#Returns the chosen response along with its probability formatted to two decimal places.
#If no suitable response is found:
#Returns a default message indicating that the chatbot does not understand the input, along with a low probability indication.

#Tokenize: It breaks text into individual words or tokens.
# Tokenization is the process of breaking down a text or a sentence into smaller units, typically words or tokens. 
# The purpose of tokenization is to prepare text data for further analysis or processing, such as natural language processing (NLP) tasks.
#Bag of Words (BoW): It counts how often each word appears in a text and represents the text as a vector of word counts.
# every unique word gets a spot in a vector, and its value is how many times it shows up in the sentence.

#This function integrates natural language processing techniques (tokenization, bag of words), deep learning (neural network prediction), 
# and probabilistic reasoning to allow a chatbot to interpret user messages and generate appropriate responses based on 
# learned patterns and predefined intents.

if __name__ == "__main__": #This line checks if the script is being run directly as the main program.
    print("Let's chat! (type 'quit' to exit)") #This line prints a message to the console, indicating that 
    #the chat session is starting and informs the user how to exit the chat by typing "quit".
    while True:
        sentence = input("You: ") #This line prompts the user to input a message, which is stored in the variable sentence.
        if sentence == "quit":
            break

        resp = get_response(sentence) # If the user input is not "quit", this line calls the get_response function with sentence as an argument. 
        #It retrieves a response based on the input message using the neural network model and associated logic defined in the get_response function.
        print(f"{bot_name}: {resp}") #the program prints the bot's response (resp) to the console, formatted as "{bot_name}: {response}"
        

#Working of getresponse func
                                            #|--- Receive user message (msg)
                                            #|
                                            #|--- Tokenize the message into individual words (tokenize(msg))
                                            #|
                                            #|--- Convert tokenized words into a numerical bag-of-words representation (bag_of_words(sentence, all_words))
                                            #|
                                            #|--- Reshape the bag-of-words representation to match the model's input shape (X = X.reshape(1, X.shape[0]))
                                            #|
                                            #|--- Convert the reshaped representation into a PyTorch tensor and move it to the appropriate device (torch.from_numpy(X).to(device))
                                            #|
                                            #|--- Pass the tensor through the neural network model to get the output (output = model(X))
                                            #|
                                            #|--- Identify the predicted class label by taking the index of the highest value in the output tensor (predicted = torch.max(output, dim=1))
                                            #|       |
                                            #|       |--- Use the predicted index to find the corresponding tag (tag = tags[predicted.item()])
                                            #|
                                            #|--- Compute the probabilities of each class using softmax on the output tensor (probs = torch.softmax(output, dim=1))
                                            #|
                                            #|--- Check if the highest probability for the predicted class is greater than 0.75:
                                            #|       |
                                            #|       |--- If true, iterate over intents to find a suitable response using the tag (find_response(intent, tag))
                                            #|       |       |
                                            #|       |       |--- Return the selected response along with its probability formatted to two decimal places ("{response} (Probability: {prob.item():.2f})")
                                            #|       |
                                            #|       |--- If no suitable response is found, return a generic response indicating low confidence ("I do not understand... (Probability: Low)")
                                            #|
                                            #|--- If the highest probability is not greater than 0.75, return a generic response indicating low confidence ("I do not understand... (Probability: Low)")
                                            #|
