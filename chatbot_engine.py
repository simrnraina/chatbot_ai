import json
import nltk
import numpy as np
import random
import pickle
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from transformers import pipeline
from datetime import datetime

# --- NLTK setup ---
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# --- File paths ---
INTENTS_FILE = "intents.json"
WORDS_FILE = "words.pkl"
CLASSES_FILE = "classes.pkl"
MODEL_FILE = "chatbot_model.h5"
TRAINING_LOG = "training_data.json"

# --- Load data ---
with open(INTENTS_FILE, "r") as file:
    data = json.load(file)

words = pickle.load(open(WORDS_FILE, 'rb'))
classes = pickle.load(open(CLASSES_FILE, 'rb'))
model = load_model(MODEL_FILE)

chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
chat_history = []

# --- User Data Storage ---
user_tasks = []
reminders = []
user_preferences = {
    "name": None,
    "productivity_style": None,
    "sleep_time": None,
    "goals": []
}

# SR Beauty tips
sr_beauty_tips = [
    "💄 SR Beauty Tip: Always remove your makeup before sleeping to let your skin breathe.",
    "🌸 SR Glow Tip: Use a vitamin C serum daily for radiant, glowing skin.",
    "💅 SR Nail Tip: Moisturize your cuticles daily to avoid breakage and dryness.",
    "🌿 SR Natural Tip: Drink 8+ glasses of water daily to keep your skin hydrated.",
    "💋 SR Lip Care: Exfoliate your lips twice a week for a soft, plump look."
]

# --- Task Management Functions ---
def add_task(task_text, deadline=None):
    user_tasks.append({"text": task_text, "done": False, "deadline": deadline})
    return f"Added: '{task_text}'" + (f" (due {deadline})" if deadline else "")

def list_tasks():
    if not user_tasks:
        return "No tasks in your list!"
    return "Your tasks:\n" + "\n".join(
        f"{i+1}. [{'✓' if t['done'] else ' '}] {t['text']}" + 
        (f" (due {t['deadline']})" if t['deadline'] else "")
        for i, t in enumerate(user_tasks))

def complete_task(task_num):
    try:
        task_num = int(task_num) - 1
        if 0 <= task_num < len(user_tasks):
            user_tasks[task_num]['done'] = True
            return f"Completed task: {user_tasks[task_num]['text']}"
        return "Invalid task number!"
    except ValueError:
        return "Please enter a valid task number."

# --- Reminder Functions ---
def set_reminder(reminder_text, when):
    reminders.append({"text": reminder_text, "time": when})
    return f"Reminder set: '{reminder_text}' for {when}"

def check_reminders():
    now = datetime.now().strftime("%H:%M")
    due_reminders = [r for r in reminders if r["time"] == now]
    if due_reminders:
        return "Reminder!\n" + "\n".join(r["text"] for r in due_reminders)
    return None

# --- Preference Management ---
def remember_preference(key, value):
    user_preferences[key] = value
    return f"I'll remember your {key} is {value}"

def add_goal(goal_text):
    user_preferences["goals"].append(goal_text)
    return f"Added goal: '{goal_text}'"

# --- Preprocessing ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# --- Prediction ---
def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return None
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

# --- Save user data for retraining ---
def save_to_dataset(user_input, bot_response):
    with open(TRAINING_LOG, "a") as f:
        json.dump({"input": user_input, "response": bot_response}, f)
        f.write("\n")

# --- Retraining function ---
def retrain_model():
    global words, classes, model, data
    print("\n🛠 Retraining model with new data...")

    # Load current intents and new training data
    with open(TRAINING_LOG, "r") as f:
        new_data = [json.loads(line) for line in f if line.strip()]

    # Convert new_data into intents format
    for entry in new_data:
        data["intents"].append({
            "tag": f"user_added_{len(data['intents'])}",
            "patterns": [entry["input"]],
            "responses": [entry["response"]]
        })

    # Save updated intents
    with open(INTENTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    # Create training dataset
    documents = []
    ignore_words = ['?', '!']
    words = []
    classes = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            w = clean_up_sentence(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    pickle.dump(words, open(WORDS_FILE, 'wb'))
    pickle.dump(classes, open(CLASSES_FILE, 'wb'))

    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1 if w in pattern_words else 0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Build the model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    model.save(MODEL_FILE)
    print("✅ Model retrained and saved!")

    # Clear training log so we don't retrain on same data repeatedly
    open(TRAINING_LOG, "w").close()

# --- Main chatbot function ---
def generate_response(user_input):
    # Check for reminders first
    reminder_check = check_reminders()
    if reminder_check:
        chat_history.append(f"Bot: {reminder_check}")
        return reminder_check

    chat_history.append(f"User: {user_input}")

    # SR Beauty
    beauty_keywords = ["beauty", "skin", "makeup", "lip", "hair", "glow", "sr beauty"]
    if any(word in user_input.lower() for word in beauty_keywords):
        tip = random.choice(sr_beauty_tips) + " — SR 🌸"
        chat_history.append(f"Bot: {tip}")
        save_to_dataset(user_input, tip)
        return tip

    # Task Management
    if "add task" in user_input.lower():
        task_text = user_input.lower().replace("add task", "").strip()
        if "due" in task_text:
            task_parts = task_text.split("due")
            task_text = task_parts[0].strip()
            deadline = task_parts[1].strip()
            response = add_task(task_text, deadline)
        else:
            response = add_task(task_text)
        chat_history.append(f"Bot: {response}")
        return response

    if "list tasks" in user_input.lower():
        response = list_tasks()
        chat_history.append(f"Bot: {response}")
        return response

    if "complete task" in user_input.lower():
        task_num = user_input.lower().replace("complete task", "").strip()
        response = complete_task(task_num)
        chat_history.append(f"Bot: {response}")
        return response

    # Reminders
    if "remind me to" in user_input.lower():
        parts = user_input.split("remind me to")[1].split("at")
        if len(parts) > 1:
            response = set_reminder(parts[0].strip(), parts[1].strip())
        else:
            response = "Please specify when with 'at' (e.g. 'remind me to drink water at 3pm')"
        chat_history.append(f"Bot: {response}")
        return response

    # Personal Preferences
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[1].strip()
        response = remember_preference("name", name)
        chat_history.append(f"Bot: {response}")
        return response

    if "i go to sleep at" in user_input.lower():
        time = user_input.lower().split("i go to sleep at")[1].strip()
        response = remember_preference("sleep_time", time)
        chat_history.append(f"Bot: {response}")
        return response

    if "add goal" in user_input.lower():
        goal = user_input.lower().replace("add goal", "").strip()
        response = add_goal(goal)
        chat_history.append(f"Bot: {response}")
        return response

    # Intent model
    intents = predict_class(user_input)
    response = get_response(intents, data)
    if response:
        chat_history.append(f"Bot: {response}")
        save_to_dataset(user_input, response)
    else:
        # Check if it's a personal question
        if "my" in user_input.lower() and ("name" in user_input.lower() or "sleep" in user_input.lower()):
            if not user_preferences.get("name"):
                response = "I don't know your personal details yet. Tell me something about yourself?"
            else:
                response = f"I remember your name is {user_preferences['name']}" + (
                    f" and you sleep at {user_preferences['sleep_time']}" 
                    if user_preferences.get('sleep_time') else "")
        else:
            # Fallback to DialoGPT
            prompt = "\n".join(chat_history[-5:])
            result = chatbot(prompt, max_length=1000, pad_token_id=50256, do_sample=True, top_p=0.92, top_k=50)
            response = result[0]['generated_text'].split("User:")[-1].strip().split("\n")[-1]
        chat_history.append(f"Bot: {response}")
        save_to_dataset(user_input, response)

    # 🔄 Auto-retrain every 10 new entries
    if os.path.exists(TRAINING_LOG) and sum(1 for _ in open(TRAINING_LOG)) >= 10:
        retrain_model()

    return response

# --- Example usage ---
if __name__ == "__main__":
    print("SR Personal Assistant initialized! Type 'quit' to exit.")
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit"]:
            break
        print("Bot:", generate_response(msg))