# LSTM-based-textual-AI-Assistant
This is my 4th MCA Final Project 2025 and the project name is "Developing a LSTM based textual AI Assistant".
LSTM-Based Textual AI Assistant: Cake Store Chatbot
Project Overview
This project is about building a smart computer program that can talk to people, like a helpful assistant for an online cake shop. It uses a special kind of Artificial Intelligence called LSTM (Long Short-Term Memory) neural networks to understand what you type or say and create new, helpful answers. Our goal is to make it easy for customers to get quick help, even using their voice!

What You Need
Before you start, make sure you have these programs on your computer:

Python 3.8 or newer: This is the main programming language. You can get it from python.org.

PyCharm Community or Professional Edition: This is a friendly tool (an "IDE") that helps you write and run Python code. Get it from jetbrains.com/pycharm/download/.

How to Get Started
Follow these simple steps to get the project working in PyCharm:

Step 1: Get the Project Files
Download the project files (usually as a ZIP file) and save them to a folder on your computer.

Once downloaded, "unzip" or "extract" the files into that folder.

Step 2: Open the Project in PyCharm
Open PyCharm.

On the welcome screen, click "Open".

Go to the folder where you saved the project files and click "Open" again.

Step 3: Set Up the Project's "Brain" (Virtual Environment)
PyCharm is smart and will usually suggest setting up a special "virtual environment." This keeps all the project's necessary parts organized.

In PyCharm, go to File > Settings (or PyCharm > Preferences on a Mac).

Find Project:  on the left, then click on Python Interpreter.

Click the ⚙️ (gear icon) next to the "Python Interpreter" menu and choose "Add New Interpreter..." > "Virtualenv Environment".

Choose "New environment".

Location: Keep this as it is (it will usually be venv inside your project folder).

Base interpreter: Pick the Python 3.8+ you installed earlier.

Make sure the boxes for "Inherit global site-packages" and "Make available to all projects" are empty (unchecked).

Click "OK" and then "Apply" in the settings window. PyCharm will set this up for you.

Step 4: Install Necessary Libraries
Now we need to install the special "libraries" (extra tools) that Python needs to run this project.

Create a requirements.txt file: In your main project folder, create a new file named requirements.txt (if it's not already there). Copy and paste this text into it:

tensorflow==2.18.o
numpy
nltk
Flask
scikit-learn # for performance reports
difflib # already built-in, but good to list if used explicitly
matplotlib # for charts

Important: Change 2.x.x to the exact TensorFlow version you used when training your model (e.g., tensorflow==2.10.0). This helps avoid problems.

Install Libraries Automatically:

Find the "Terminal" tab at the bottom of PyCharm (it's usually next to "Run" or "Python Console"). Click on it.

You should see (venv) at the start of the line in the terminal, meaning your project's setup is active.

Type this command and press Enter:

pip install -r requirements.txt

This command will automatically download and install all the listed tools.

Download NLTK Data:

After the previous step, you need one more small download for the nltk tool. In the same PyCharm terminal, type this and press Enter:

python -m nltk.downloader punkt

Step 5: Put Your Model Files in Place
Make sure your trained chatbot model and its language understanding file are in the right spots:

chatbot_seq2seq_model.keras: Put this file directly inside your main project folder.

data/tokenizer.pkl: Create a folder named data inside your main project folder, then put tokenizer.pkl inside that data folder.

Running the Chatbot Application (The Web Interface)
This part starts the chatbot so you can talk to it in your web browser.

Open the PyCharm Terminal.

Make sure you still see (venv) at the start of the line.

Tell Flask (the web tool) where your app is:

If you're on Windows (Command Prompt-style terminal):

set FLASK_APP=app.py
set FLASK_ENV=development

If you're on Mac or Linux (Bash/Zsh terminal):

export FLASK_APP=app.py
export FLASK_ENV=development

Start the chatbot:

flask run

The terminal will show you a web address (like http://127.0.0.1:5000/). Copy this address and paste it into your web browser to open the chatbot!

Running the Evaluation Script (See How Well it Works)
This script helps you check how accurate your chatbot is.

Open the PyCharm Terminal.

Make sure (venv) is active.

Run the script:

python test_script.py

This will show you performance numbers and a chart to understand the chatbot's accuracy.

Project Files (Where Everything Is)



