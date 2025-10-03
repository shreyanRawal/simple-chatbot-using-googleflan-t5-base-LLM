from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small" #Google's LLM 
tokenizer = AutoTokenizer.from_pretrained(model_name) 

model = AutoModelForSeq2SeqLM.from_pretrained(model_name) 

def run(): 
    while True:

        # Getting user input 
        input_text = input("Enter a text or query : ") 

        # Exit conditions 
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break 

        # Tokenizing input and generating response 
        inputs = tokenizer.encode(input_text, return_tensors= "pt")
        outputs = model.generate(inputs, max_new_tokens = 200) 
        response = tokenizer.decode(outputs[0],skip_special_tokens=True).strip() 

        # Displaying bot's response 
        print("Chatbot: ", response) 


# starting the program 
run()