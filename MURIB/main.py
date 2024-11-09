from kivymd.uix.list import ThreeLineListItem
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
import sqlite3
from bidi.algorithm import get_display
import arabic_reshaper
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
import random
import uuid
from kivy.properties import StringProperty
import cv2
import time
import re
import bcrypt
import pickle
import mediapipe as mp
import json
import requests
from typing import List, Dict, Set
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model


Window.size  = (350, 580)


class ArabicGrammarAnalyzer:
    def __init__(self, api_key: str, project_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.access_token = self._get_access_token()
        self.credentials = self._setup_credentials()
        self.model = self._setup_model()
        self.grammar_index = {}  # Index storing sentences by their grammatical patterns
        self.sentence_data = {}  # Store full sentence data

        # List of all possible sentence types
        self.sentence_types = [
            "مبتدأ وخبر", "كان وأخواتها", "إن وأخواتها", "إضافة", "نعت",
            "بدل الكل من الكل", "بدل بعض من كل", "بدل الاشتمال", "البدل المباين",
            "توكيد لفظي", "توكيد معنوي", "ضمير متصل", "ضمير مستتر",
            "فعل ماضٍ مبني للمجهول", "أدوات النصب", "الفعل المضارع",
            "حرف الجر", "حرف جزم", "حرف شرط جازم", "اسم شرط جازم",
            "مفعول فيه - ظرف زمان", "مفعول فيه - ظرف مكان - ظرف زمان",
            "مفعول فيه - ظرف مكان", "مفعول مطلق", "مفعول لأجله", "مفعول معه",
            "حال مفرد", "حال جملة اسمية", "حال جملة فعلية",
            "تمييز الذات", "تمييز النسبة"
        ]

    def _get_access_token(self) -> str:
        url = 'https://iam.cloud.ibm.com/identity/token'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
            'apikey': self.api_key
        }
        response = requests.post(url, headers=headers, data=data)
        return json.loads(response.content)['access_token']

    def _setup_credentials(self) -> Credentials:
        return Credentials(
            url="https://eu-de.ml.cloud.ibm.com/",
            token=self.access_token
        )

    def _setup_model(self) -> Model:
        parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "repetition_penalty": 1.2,
            "temperature": 0.7,
            "top_p": 0.9
        }
        return Model(
            model_id="sdaia/allam-1-13b-instruct",
            params=parameters,
            credentials=self.credentials,
            project_id=self.project_id
        )

    def _extract_grammatical_patterns(self, sentence_data: Dict) -> Set[str]:
        """Extract all grammatical patterns from a sentence"""
        patterns = set()

        # Add the main sentence type
        if 'sentence_type' in sentence_data:
            patterns.add(sentence_data['sentence_type'])

        # Add patterns from word analysis
        for word in sentence_data.get('words', []):
            # Add grammatical functions as patterns
            if 'grammatical_function' in word:
                patterns.add(word['grammatical_function'])

            # Add case patterns
            if 'grammatical_case' in word:
                patterns.add(word['grammatical_case'])

        return patterns

    def load_and_index_data(self, json_file_path: str):
        """Load data and index it by grammatical patterns"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create indices for each grammatical pattern
        for idx, item in enumerate(data):
            # Generate a unique ID for the sentence
            sentence_id = f"sentence_{idx}"

            # Store the full sentence data
            self.sentence_data[sentence_id] = item

            # Extract and index all grammatical patterns
            patterns = self._extract_grammatical_patterns(item)

            # Index the sentence under each of its patterns
            for pattern in patterns:
                if pattern not in self.grammar_index:
                    self.grammar_index[pattern] = set()
                self.grammar_index[pattern].add(sentence_id)

    def _find_pattern_matches(self, input_patterns: Set[str], n_results: int = 3) -> List[str]:
        """Find sentences that share the most grammatical patterns with the input"""
        # Count pattern matches for each sentence
        sentence_matches = {}

        for pattern in input_patterns:
            if pattern in self.grammar_index:
                for sentence_id in self.grammar_index[pattern]:
                    sentence_matches[sentence_id] = sentence_matches.get(sentence_id, 0) + 1

        # Sort sentences by number of matching patterns
        sorted_matches = sorted(
            sentence_matches.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top n sentence IDs
        return [sentence_id for sentence_id, _ in sorted_matches[:n_results]]

    def find_similar_sentences(self, query_sentence_data: Dict, n_results: int = 3) -> List[Dict]:
        """Find sentences with similar grammatical patterns - improved version"""
        similar_sentences = []
        sentence_types = query_sentence_data.get("sentence_types", [])

        # Create a scoring system for similarity
        sentence_scores = {}

        for sid, stored_sentence in self.sentence_data.items():
            score = 0
            stored_types = [stored_sentence.get("sentence_type")] if stored_sentence.get("sentence_type") else []

            # Score based on matching sentence types
            for qtype in sentence_types:
                if qtype in stored_types:
                    score += 3  # Higher weight for matching types

            # Score based on grammatical patterns
            query_patterns = self._extract_grammatical_patterns(query_sentence_data)
            stored_patterns = self._extract_grammatical_patterns(stored_sentence)

            common_patterns = query_patterns.intersection(stored_patterns)
            score += len(common_patterns)

            if score > 0:
                sentence_scores[sid] = score

        # Get top scoring sentences
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top n sentences
        for sid, _ in sorted_sentences[:n_results]:
            similar_sentences.append(self.sentence_data[sid])

        return similar_sentences

    # Add this new method to your ArabicGrammarAnalyzer class
    def detect_sentence_types(self, sentence: str) -> List[str]:
        """Detect all applicable sentence types from the input sentence"""
        detected_types = set()
        words = sentence.split()

        # Dictionary of pattern indicators
        patterns = {
            "كان وأخواتها": [
                "كان", "أصبح", "أمسى", "أضحى", "ظل", "بات", "صار", "ليس",
                "مازال", "مابرح", "مافتئ", "ماانفك", "مادام"
            ],
            "إن وأخواتها": [
                "إن", "أن", "كأن", "لكن", "ليت", "لعل"
            ],
            "حرف الجر": [
                "من", "إلى", "عن", "على", "في", "ب", "ل", "ك", "حتى",
                "منذ", "مذ", "رب", "خلا", "عدا", "حاشا"
            ],
            "مفعول فيه - ظرف زمان": [
                "اليوم", "غداً", "أمس", "صباحاً", "مساءً", "ليلاً", "نهاراً",
                "حين", "وقت", "ساعة", "يوم", "أسبوع", "شهر", "سنة"
            ],
            "مفعول فيه - ظرف مكان": [
                "فوق", "تحت", "أمام", "خلف", "يمين", "شمال", "وراء",
                "قدام", "عند", "مع", "إزاء", "حذاء", "تلقاء", "بين"
            ]
        }

        def has_tanween(word: str) -> bool:
            """Check if word has tanween (ًٌٍ)"""
            return any(char in word for char in "ًٌٍ")

        def ends_with_kasra(word: str) -> bool:
            """Check if word ends with kasra (ِ)"""
            return 'ِ' in word

        # Check for patterns in each word
        for word in words:
            # Check for كان وأخواتها
            if any(word.startswith(pattern) for pattern in patterns["كان وأخواتها"]):
                detected_types.add("كان وأخواتها")

            # Check for إن وأخواتها
            if any(word.startswith(pattern) for pattern in patterns["إن وأخواتها"]):
                detected_types.add("إن وأخواتها")

            # Check for حروف الجر
            if word in patterns["حرف الجر"]:
                detected_types.add("حرف الجر")

            # Check for ظروف
            if word in patterns["مفعول فيه - ظرف زمان"]:
                detected_types.add("مفعول فيه - ظرف زمان")
            if word in patterns["مفعول فيه - ظرف مكان"]:
                detected_types.add("مفعول فيه - ظرف مكان")

        # Check for إضافة
        for i in range(len(words) - 1):
            if ends_with_kasra(words[i + 1]):
                detected_types.add("إضافة")
                break

        # If no other major structure is detected and sentence starts with noun
        if not detected_types and not any(words[0].startswith(pattern)
                                          for pattern in patterns["كان وأخواتها"] + patterns["إن وأخواتها"]):
            detected_types.add("مبتدأ وخبر")

        return list(detected_types)

    # Add this new method
    def analyze_input_sentence(self, sentence: str) -> Dict:
        """Analyze input sentence and prepare it for finding similar sentences"""
        # Detect sentence types
        sentence_types = self.detect_sentence_types(sentence)

        # Create initial analysis structure
        sentence_data = {
            "sentence": sentence,
            "sentence_types": sentence_types,  # Now multiple types possible
            "words": []  # Will be filled by the model's analysis
        }

        return sentence_data

    def create_prompt_with_examples(self, query_sentence: str, similar_sentences: List[Dict]) -> str:
        """Create a token-efficient prompt"""
        prompt = f"""<|system|>
    محلل نحوي للجمل العربية.

    مثال مشابه:
    """
        # Use only the most relevant example
        if similar_sentences:
            example = similar_sentences[0]
            prompt += f"الجملة: {example['sentence']}\n"
            prompt += f"النوع: {example['sentence_type']}\n"
            prompt += "التحليل:\n"

            # Only include key grammatical elements
            for word in example['words']:
                prompt += f"{word['word']}: {word['grammatical_function']} - {word['grammatical_case']}"
                if 'grammatical_marker_explanation' in word:
                    prompt += f" ({word['grammatical_marker']})\n"
                else:
                    prompt += "\n"

        prompt += f"""<|human|>
    حلل: {query_sentence}

    <|assistant|>
    """
        return prompt

    def analyze_sentence(self, sentence: str, sentence_data: Dict) -> str:
        """Analyze Arabic sentence with token efficiency"""
        # Find similar sentences
        similar_sentences = self.find_similar_sentences(sentence_data)

        # Print detected types
        print("=== أنواع الجملة ===")
        for stype in sentence_data.get("sentence_types", []):
            print(f"• {stype}")
        print()

        # Create efficient prompt
        prompt = self.create_prompt_with_examples(sentence, similar_sentences)

        # Query model
        response = self._query_model(prompt)
        return response

    def _query_model(self, prompt: str) -> str:
        """Separate method for querying the model"""
        request_url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 1000,
                "repetition_penalty": 1
            },
            "model_id": "sdaia/allam-1-13b-instruct",
            "project_id": self.project_id
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        response = requests.post(request_url, headers=headers, json=body)

        if response.status_code == 200:
            return response.json()['results'][0]['generated_text']
        else:
            return f"Error: {response.status_code}, {response.text}"


class FirstWindow(Screen):
    pass


class SignInWindow(Screen):

    def verify_login(self, username, password):
        conn = sqlite3.connect('users_data.db')
        cursor = conn.cursor()

        # Query the database for the provided username
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        if user is not None:
            stored_password = user[3]  # 3المكان المخزن فيه الباسورد tuple in the databases

            if user and bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):

                print("Login Successfully!")

                self.clear_fields()

                # Pass the username to the Q_alphabet screen
                self.manager.get_screen('q_alphabet').username = username

                # Pass the username to the Q_numbers screen
                self.manager.get_screen('q_numbers').username = username

                # Pass the user information to the UserAccount screen
                user_account_screen = self.manager.get_screen('useraccount')
                user_account_screen.res00 = user[0]  # Username
                user_account_screen.res01 = user[1]  # Email
                user_account_screen.res02 = user[2]  # Mobile Number
                user_account_screen.res03 = user[3]  # Password

                # Pass the user information to the UserAccount screen
                edit_user_account_screen = self.manager.get_screen('edituserinfo')
                edit_user_account_screen.res00 = user[0]  # Username
                edit_user_account_screen.res01 = user[1]  # Email
                edit_user_account_screen.res02 = user[2]  # Mobile Number
                edit_user_account_screen.res03 = password  # Password

                # Pass the user information to the history
                self.manager.get_screen('history').username = username

                # Navigate to the homepage upon successful login
                self.manager.current = 'homepage'

            else:
                print("Invalid password, please try again")  # debug
                self.show_invalid_login_popup()
        else:
            print("Invalid username, please try again")  # debug
            self.show_invalid_login_popup()

    def show_invalid_login_popup(self):
        dialog = MDDialog(
            title=" Login Failed",
            text="The username or password you entered is incorrect. Please try again.",
            buttons=[
                MDFlatButton(
                    text="OK", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()

    def clear_fields(self):
        # Clear input fields after successful signup
        self.ids.text_field5.text = ""
        self.ids.text_field6.text = ""


class SignUpWindow(Screen):

    def sign_up(self, username, email, mobileNum, password):
        hashed_password = self.hash_password(password)
        print("Hashed Password:",
              hashed_password)  # here I want to make sure that the pass is hashed ,so I print it
        # Validate user inputs
        if not self.validate_username(username):
            self.ids.username_error_label.text = " The username should be between 1 and 10 characters long"
            return False
        if not self.validate_email(email):
            self.ids.email_error_label.text = "Invalid email format"
            return False
        if not self.validate_mobile_number(mobileNum):
            self.ids.mobile_error_label.text = "Invalid mobile number format"
            return False
        if not self.check_password_strength(password):
            self.ids.password_strength_label.text = " "
            return False

        # If all validation passes and signup is successful, set the flag to True
        self.sign_up_success = True

        # Connect to the SQLite database and perform signup
        signup_success = self.perform_signup(username, email, mobileNum, hashed_password)
        if signup_success:
            # Clear input fields and transition to homepage on successful signup
            self.clear_fields()
            self.manager.current = 'signin'
            self.manager.transition.direction = "right"

    def perform_signup(self, username, email, mobileNum, password):
        conn = None  # Initialize conn variable
        try:
            # Connect to the database
            conn = sqlite3.connect('users_data.db')
            cursor = conn.cursor()

            # Check if email already exists
            cursor.execute("SELECT * FROM users WHERE email=?", (email,))
            existing_user = cursor.fetchone()
            if existing_user:
                print("Error: Email already exists.")  # Debug message
                self.show_existing_email_popup()
                return False  # Email already exists, signup failed

            # Insert the new user info to the database
            cursor.execute("INSERT INTO users (username, email, mobileNum, password) VALUES (?, ?, ?, ?)",
                           (username, email, mobileNum, password))
            conn.commit()
            print("User successfully signed up.")  # Debug message

            # Fetch the user's data from the database after successful signup
            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            # user = cursor.fetchone()

            # Close the database connection
            conn.close()

            # Pass the user information and selected avatar to the UserAccount screen
            user_account_screen = self.manager.get_screen('useraccount')
            user_account_screen.res00 = username
            user_account_screen.res01 = email
            user_account_screen.res02 = mobileNum

            self.manager.current = 'useraccount'

            self.manager.current = 'signin'

            return True  # Return True to indicate successful signup

        except sqlite3.Error as e:
            print("Error E:", e)  # Debug message
            # self.show_signup_failure_popup()
            return False  # Signup failed due to database error

        finally:
            if conn:
                conn.close()

            # Handle signup failure by displaying an error message
            # self.show_signup_failure_popup()
            # print("FINALLY")
            return False

    def hash_password(self, password):
        # Hash the password using bcrypt
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return hashed.decode('utf-8')  # Decode bytes to string for storage

    def show_signup_failure_popup(self):
        dialog = MDDialog(
            title="Signup Failed",
            text="There was an error during signup. Please try again.",
            buttons=[
                MDFlatButton(
                    text="OK", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()

    def show_existing_email_popup(self):
        dialog = MDDialog(
            title="Invalid Email",
            text="This email address is already used. Please try another one.",
            buttons=[
                MDFlatButton(
                    text="OK", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()

    def show_username_exists_popup(self):
        dialog = MDDialog(
            title="Username Exists",
            text="This username already exists. Please choose a different username.",
            buttons=[
                MDFlatButton(
                    text="OK", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()

    def clear_fields(self):
        # Clear input fields after successful signup
        self.ids.text_field1.text = ""
        self.ids.text_field2.text = ""
        self.ids.text_field3.text = ""
        self.ids.text_field4.text = ""
        self.ids.email_error_label.text = ""
        self.ids.mobile_error_label.text = ""
        self.ids.password_strength_label.text = ""
        self.ids.password_strength_icon.icon = "eye-off"

    def validate_username(self, username):
        username_pattern = r"^\w{1,10}$"
        conn = sqlite3.connect('users_data.db')
        cursor = conn.cursor()
        # Check if the username already exists
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            self.show_username_exists_popup()
            print("Username already exists in the database.")  # Debug message
            return False  # Stop the signup process if username already exists
        if not re.match(username_pattern, username):
            self.ids.username_error_label.text = " The username should be between 1 and 10 characters long"
            return False
        else:
            self.ids.email_error_label.text = ""  # Clear error message
        return True

    def validate_email(self, email):
        # Regular expression pattern for email validation
        email_pattern = r"^[a-zA-Z0-9_.+-]+@(?:gmail\.com|yahoo\.com|hotmail\.com)$"
        if not re.match(email_pattern, email):
            self.ids.email_error_label.text = "Invalid email format"
            return False
        else:
            self.ids.email_error_label.text = ""  # Clear error message
        return True

    def validate_mobile_number(self, mobileNum):
        # Define the regex pattern for a valid Saudi Arabian mobile number starting with +966
        pattern = r"^\+9665\d{8}$"

        # Check if the input matches the pattern
        if re.match(pattern, mobileNum):
            self.ids.mobile_error_label.text = ""
            return True
        else:
            self.ids.mobile_error_label.text = "Invalid mobile number format"
            return False

    def check_password_strength(self, password):
        # Check password strength
        has_capital = any(char.isupper() for char in password)
        has_small = any(char.islower() for char in password)
        has_number = any(char.isdigit() for char in password)
        is_valid_length = len(password) >= 8

        # Update labels based on password strength
        self.ids.eight_character_requirement_label.text_color = "#1C3B16" if is_valid_length else "#FF0000"
        self.ids.capital_requirement_label.text_color = "#1C3B16" if has_capital else "#FF0000"
        self.ids.small_requirement_label.text_color = "#1C3B16" if has_small else "#FF0000"
        self.ids.number_requirement_label.text_color = "#1C3B16" if has_number else "#FF0000"

        # Change password field color if all requirements are met
        if all([has_capital, has_small, has_number, is_valid_length]):
            # self.ids.password_strength_label.text = ""
            self.ids.text_field4.line_color_normal = "#1C3B16"  # Green
            return True

        else:
            # self.ids.password_strength_label.text = "Password does not meet requirements"
            self.ids.text_field4.line_color_normal = "#FF0000"  # Red
            return False


class HomePage(Screen):
    # Initialize the analyzer with the correct credentials
    analyzer = ArabicGrammarAnalyzer(
        api_key="EOQqBdGAQ866AVloO7CK3WFVtFqixZxoTBuWlyq1fgfu",
        project_id="09d6ad16-4d32-422b-9bae-86f3ead73c8d"
    )

    # Load and index the data when the app starts
    analyzer.load_and_index_data('all_arabic_grammar_dataset.json')

    def send_message(self, message):
        # Check if message is empty
        if message:
            print("Message:", message)  # Print the message in the terminal
            result = self.send_to_allam_model(message)  # Send the entered message to Allam model
        else:
            print("No message entered.")
            result = self.send_to_allam_model("No message entered.")  # Default message if none entered

        # After analyzing, pass the result to the Result_alpha screen
        self.manager.get_screen('result_alpha').update_result(result)

    def send_to_allam_model(self, text):
        # Call the Allam model (ArabicGrammarAnalyzer) to process the text
        # First analyze the input sentence to detect its types
        sentence_data = self.analyzer.analyze_input_sentence(text)

        # Print detected sentence types (optional)
        print("Sentence Types:", sentence_data)

        # Analyze the sentence
        result = self.analyzer.analyze_sentence(sentence_data["sentence"], sentence_data)

        # Output the result (this will be passed to Result_alpha)
        print("Analysis Result:", result)

        return result


class UserAccount(Screen):
    res00 = StringProperty()  # to hold the username
    res01 = StringProperty()  # to hold the email
    res02 = StringProperty()  # to hold the mobile number

    def __init__(self, **kwargs):
        super(UserAccount, self).__init__(**kwargs)

        # Fetch user data from the database
        connection = sqlite3.connect("users_data.db")
        cursor = connection.cursor()
        cursor.execute("SELECT username, email, mobileNum FROM users;")
        user_data = cursor.fetchone()  # Assuming there's only one user for simplicity

        if user_data:
            # Set the properties only if user_data is not None
            self.res00 = user_data[0]  # Set the username
            self.res01 = user_data[1]  # Set the email
            self.res02 = user_data[2]  # Set the mobile number

        connection.close()

    def logout(self):
        # Clear registration information here
        self.manager.get_screen('signup').clear_fields()
        # Navigate to the first window
        self.manager.current = 'first'
        self.manager.transition.direction = "left"


class EditUserAccount(Screen):  # edituserinfo [Screen name]
    res00 = StringProperty()  # to hold the username
    res01 = StringProperty()  # to hold the email
    res02 = StringProperty()  # to hold the mobile number
    res03 = StringProperty()  # to hold the password

    def __init__(self, **kwargs):
        super(EditUserAccount, self).__init__(**kwargs)
        # Fetch user data from the database
        connection = sqlite3.connect("users_data.db")
        cursor = connection.cursor()
        cursor.execute("SELECT username, email, mobileNum , password FROM users;")
        user_data = cursor.fetchone()  # Assuming there's only one user for simplicity
        if user_data:
            # Set the properties only if user_data is not None
            self.res00 = user_data[0]  # Set the username
            self.res01 = user_data[1]  # Set the email
            self.res02 = user_data[2]  # Set the mobile number
            self.res03 = user_data[3]
        connection.close()

    def update_user_info(self):
        # Get the updated information from the text fields
        new_email = self.ids.text_field2.text
        new_mobile_number = self.ids.text_field3.text
        new_password = self.ids.text_field4.text

        # Hash the new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        # Connect to the database
        conn = sqlite3.connect('users_data.db')
        cursor = conn.cursor()

        try:
            # Check if the new email already exists in the database
            if new_email:
                cursor.execute("SELECT COUNT(*) FROM users WHERE email=? AND username != ?", (new_email, self.res00))
                count = cursor.fetchone()[0]
                if count > 0:
                    raise ValueError("Email address already exists.")

            # Update the user's information in the database with the hashed password
            cursor.execute("UPDATE users SET email=?, mobileNum=?, password=? WHERE username=?",
                           (new_email, new_mobile_number, hashed_password.decode('utf-8'), self.res00))
            conn.commit()
            print("User information updated successfully.")

        except ValueError as e:
            print(e)  # Handle the error gracefully

        finally:
            conn.close()

    def update_user_info_and_redirect(self):
        self.update_user_info()  # Call the update_user_info method to update user info
        self.manager.current = 'useraccount'


class Categories(Screen):
    pass


class Translatepage(Screen):
    pass


class Lessons1(Screen):
    pass


class Lessons2(Screen):
    pass


class Lessons3(Screen):
    pass


class Test_alphabet(Screen):
    pass


class Test_numbers(Screen):
    pass


class Q_alphabet(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.username = None
        self.question_index = 0  # Track the current question index
        self.score = 0  # Track the user's score
        self.total_questions = 0  # Track the total number of questions
        self.incorrect_questions = []

        res0 = get_display(arabic_reshaper.reshape("حرف أ"))
        res1 = get_display(arabic_reshaper.reshape("حرف ب"))
        res2 = get_display(arabic_reshaper.reshape("حرف ت"))
        res3 = get_display(arabic_reshaper.reshape("حرف ث"))
        res4 = get_display(arabic_reshaper.reshape("حرف ج"))
        res5 = get_display(arabic_reshaper.reshape("حرف ح"))
        res6 = get_display(arabic_reshaper.reshape("حرف خ"))
        res7 = get_display(arabic_reshaper.reshape("حرف د"))
        res8 = get_display(arabic_reshaper.reshape("حرف ذ"))
        res9 = get_display(arabic_reshaper.reshape("حرف ر"))
        res10 = get_display(arabic_reshaper.reshape("حرف ز"))
        res11 = get_display(arabic_reshaper.reshape("حرف س"))
        res12 = get_display(arabic_reshaper.reshape("حرف ش"))
        res13 = get_display(arabic_reshaper.reshape("حرف ص"))
        res14 = get_display(arabic_reshaper.reshape("حرف ض"))
        res15 = get_display(arabic_reshaper.reshape("حرف ط"))
        res16 = get_display(arabic_reshaper.reshape("حرف ظ"))
        res17 = get_display(arabic_reshaper.reshape("حرف ع"))
        res18 = get_display(arabic_reshaper.reshape("حرف غ"))
        res19 = get_display(arabic_reshaper.reshape("حرف ف"))
        res20 = get_display(arabic_reshaper.reshape("حرف ق"))
        res21 = get_display(arabic_reshaper.reshape("حرف ك"))
        res22 = get_display(arabic_reshaper.reshape("حرف ل"))
        res23 = get_display(arabic_reshaper.reshape("حرف م"))
        res24 = get_display(arabic_reshaper.reshape("حرف ن"))
        res25 = get_display(arabic_reshaper.reshape("حرف هـ"))
        res26 = get_display(arabic_reshaper.reshape("حرف و"))
        res27 = get_display(arabic_reshaper.reshape("حرف ي"))

        # Define the questions data
        self.questions = [
            {
                "question": res0,
                "correct_answer": "0032"
            },
            {
                "question": res1,
                "correct_answer": "0033"
            },
            {
                "question": res2,
                "correct_answer": "0034"
            },
            {
                "question": res3,
                "correct_answer": "0035"
            },
            {
                "question": res4,
                "correct_answer": "0036"
            },
            {
                "question": res5,
                "correct_answer": "0037"
            },
            {
                "question": res6,
                "correct_answer": "0038"
            },
            {
                "question": res7,
                "correct_answer": "0039"
            },
            {
                "question": res8,
                "correct_answer": "0040"
            },
            {
                "question": res9,
                "correct_answer": "0041"
            },
            {
                "question": res10,
                "correct_answer": "0042"
            },
            {
                "question": res11,
                "correct_answer": "0043"
            },
            {
                "question": res12,
                "correct_answer": "0044"
            },
            {
                "question": res13,
                "correct_answer": "0045"
            },
            {
                "question": res14,
                "correct_answer": "0046"
            },
            {
                "question": res15,
                "correct_answer": "0047"
            },
            {
                "question": res16,
                "correct_answer": "0048"
            },
            {
                "question": res17,
                "correct_answer": "0049"
            },
            {
                "question": res18,
                "correct_answer": "0050"
            },
            {
                "question": res19,
                "correct_answer": "0051"
            },
            {
                "question": res20,
                "correct_answer": "0052"
            },
            {
                "question": res21,
                "correct_answer": "0053"
            },
            {
                "question": res22,
                "correct_answer": "0054"
            },
            {
                "question": res23,
                "correct_answer": "0055"
            },
            {
                "question": res24,
                "correct_answer": "0056"
            },
            {
                "question": res25,
                "correct_answer": "0057"
            },
            {
                "question": res26,
                "correct_answer": "0058"
            },
            {
                "question": res27,
                "correct_answer": "0059"
            },
        ]
        # Randomize the order of questions
        random.shuffle(self.questions)

    def on_pre_enter(self):
        # Reset the question index and score
        self.question_index = 0
        self.score = 0

        # Calculate the total number of questions
        self.total_questions = len(self.questions)

        # Display the current question and options
        self.display_current_question()

    def display_current_question(self):

        # Get the current question data
        question_data = self.questions[self.question_index]

        # Set the question text
        res1 = get_display(arabic_reshaper.reshape("السؤال"))
        self.ids.question_label.text = f"{question_data['question']} :{self.question_index + 1}{res1} "

        # Reset the progress bar to 0
        self.ids.progress_bar.value = 0
        # Update the progress bar
        progress = (self.question_index / len(self.questions)) * 100

        self.ids.progress_bar.value = progress

    def video_recorder_alpha(self):
        # Check if there are more questions to process
        if self.question_index < len(self.questions):
            # question_data = self.questions[self.question_index]
            # correct_answer = question_data.get("correct_answer", "")

            # Set up video capture
            window_name = "Image Capture"
            # cv2.namedWindow(window_name)

            camera_window_x = 470
            camera_window_y = 230

            # Position the camera feed window above the app window
            cv2.moveWindow(window_name, camera_window_x, camera_window_y)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Start recording
            start_time = time.time()
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Display the resulting frame
                cv2.imshow(window_name, frame)  # Use window_name here

                # Check if 5 seconds have elapsed
                if time.time() - start_time >= 5:
                    # Capture a picture after 5 seconds
                    image_path = f'alpha_pic_{self.question_index + 1}.jpg'
                    cv2.imwrite(image_path, frame)
                    self.preprocess_image(image_path)
                    break  # Exit the loop after capturing the picture

                # Check for key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Release the VideoCapture and VideoWriter objects
            cap.release()
            # Destroy the window
            cv2.destroyWindow(window_name)

        else:
            print("All questions processed.")

    def preprocess_image(self, image_path):

        print(image_path)
        # Load the captured image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image from {image_path}")
            return None
        # Resize the image to 244x244
        # img_resized = cv2.resize(img, (224, 224))
        # print("resize: ", img_resized)
        img_enhanced = self.perform_image_enhancement(img)

        # Initialize mediapipe hands module
        mp_hands = mp.solutions.hands
        # mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        data_aux_test = []
        data_test = []

        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        print("img_rgb", img_rgb)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # print(x)
                    data_aux_test.append(x)
                    data_aux_test.append(y)
            data_test.append(data_aux_test)
            print("datatest", data_test)
            max_length = 84
            # Pad the testing data to the maximum length
            padded_testing_data = data_aux_test + [0] * (max_length - len(data_aux_test))
            self.predict_alphabet([padded_testing_data])

        else:
            print("Error: No hand landmarks detected in the image.")
            return None

    def perform_image_enhancement(self, img):
        # Perform image enhancement (e.g., contrast adjustment, sharpening, noise reduction)
        img_enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
        return img_enhanced

    def predict_alphabet(self, preprocessed_image):
        # Load the trained model from the pickle file
        filename = 'modelRFMLetter.p'
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        # Extract the model from the dictionary
        model = data['model']
        # Debugging: Print the shape or type of preprocessed_image
        print("Preprocessed image data:", preprocessed_image)

        # Now you can use the predict method
        prediction = model.predict(preprocessed_image)
        print("Model prediction:", prediction)  # Debugging: Check what the model predicts
        self.check_answer(prediction)

    def check_answer(self, prediction):
        # if self.question_index < len(self.questions):
        question_data = self.questions[self.question_index]
        correct_answer = question_data.get("correct_answer", "")

        # Print the data types of prediction and correct_answer
        print("Data type of prediction:", type(prediction))
        print("Data type of correct_answer:", type(correct_answer))

        print("Correct answer:", correct_answer)
        if prediction == correct_answer:
            self.score += 1
        else:
            # If the answer is incorrect, append the question data to the list
            self.incorrect_questions.append(question_data)
        # Navigate to the next question or the result screen
        self.question_index += 1
        if self.question_index < len(self.questions):
            self.display_current_question()
        else:
            # Calculate the total score
            total_score = self.score

            # Generate a unique ID for the test
            test_id = str(uuid.uuid4())

            # Store the total score in the database
            self.store_test_score(test_id, total_score)
            # Navigate to the result screen
            self.manager.current = 'result_alpha'
            result_screen = self.manager.get_screen('result_alpha')
            result_screen.update_result(self.score, self.total_questions, self.incorrect_questions)
            self.incorrect_questions = []

    def store_test_score(self, test_id, total):
        # Connect to the SQLite database
        conn = sqlite3.connect('users_data.db')
        cursor = conn.cursor()

        res1 = get_display(arabic_reshaper.reshape("الحروف "))

        # Insert the user's test score into the 'test' table
        cursor.execute("INSERT INTO tests (id, score, username, testName) VALUES (?, ?, ?, ?)",
                       (test_id, total, self.username, "Letters"))

        # Commit changes to the database
        conn.commit()

        # Close the connection
        conn.close()


class Q_numbers(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.username = None
        self.question_index = 0  # current question index
        self.score = 0  # user's score
        self.total_questions = 0  # total number of questions
        self.incorrect_questions = []

        # convert to arbic
        res0 = get_display(arabic_reshaper.reshape(" صفر    "))
        res1 = get_display(arabic_reshaper.reshape(" واحد   "))
        res2 = get_display(arabic_reshaper.reshape(" اثنين  "))
        res3 = get_display(arabic_reshaper.reshape(" ثلاثة   "))
        res4 = get_display(arabic_reshaper.reshape(" اربعة "))
        res5 = get_display(arabic_reshaper.reshape(" خمسة   "))
        res6 = get_display(arabic_reshaper.reshape(" ستة    "))
        res7 = get_display(arabic_reshaper.reshape(" سبعة   "))
        res8 = get_display(arabic_reshaper.reshape(" ثمانية "))
        res9 = get_display(arabic_reshaper.reshape(" تسعة   "))

        # Define the questions data
        self.questions = [
            {
                "question": res0,
                "correct_answer": "0001"
            },
            {
                "question": res1,
                "correct_answer": "0002"
            },
            {
                "question": res2,
                "correct_answer": "0003"
            },
            {
                "question": res3,
                "correct_answer": "0004"
            },
            {
                "question": res4,
                "correct_answer": "0005"
            },
            {
                "question": res5,
                "correct_answer": "0006"
            },
            {
                "question": res6,
                "correct_answer": "0007"
            },
            {
                "question": res7,
                "correct_answer": "0008"
            },
            {
                "question": res8,
                "correct_answer": "0009"
            },
            {
                "question": res9,
                "correct_answer": "0010"
            },

        ]
        # Randomize the order of questions
        random.shuffle(self.questions)

    def on_pre_enter(self):
        # Reset the question index and score
        self.question_index = 0
        self.score = 0

        # Calculate the total number of questions
        self.total_questions = len(self.questions)

        # Display the current question and options
        self.display_current_question()

    def display_current_question(self):

        # Get the current question data
        question_data = self.questions[self.question_index]

        # Set the question text
        res1 = get_display(arabic_reshaper.reshape("السؤال"))
        self.ids.question_label.text = f"{question_data['question']} :{self.question_index + 1}{res1} "

        # Update the progress bar
        progress = (self.question_index / len(self.questions)) * 100

        self.ids.progress_bar.value = progress

    def image_capture_num(self):
        # Check if there are more questions to process
        if self.question_index < len(self.questions):
            # question_data = self.questions[self.question_index]
            # correct_answer = question_data.get("correct_answer", "")

            # Set up video capture
            window_name = "Image Capture"
            cv2.namedWindow(window_name)

            camera_window_x = 470
            camera_window_y = 230

            # Position the camera feed window above the app window
            cv2.moveWindow(window_name, camera_window_x, camera_window_y)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Start recording
            start_time = time.time()
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Display the resulting frame
                cv2.imshow(window_name, frame)  # Use window_name here

                # Check if 5 seconds have elapsed
                if time.time() - start_time >= 5:
                    # Capture a picture after 5 seconds
                    image_path = f'num_pic_{self.question_index + 1}.jpg'
                    cv2.imwrite(image_path, frame)
                    self.preprocess_image(image_path)
                    break  # Exit the loop after capturing the picture

                # Check for key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Release the VideoCapture and VideoWriter objects
            cap.release()
            # Destroy the window
            cv2.destroyWindow(window_name)

        else:
            print("All questions processed.")

    def preprocess_image(self, image_path):

        print(image_path)
        # Load the captured image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image from {image_path}")
            return None
        # Resize the image to 244x244
        # img_resized = cv2.resize(img, (244, 244))
        # print("resize: ", img_resized)
        img_enhanced = self.perform_image_enhancement(img)

        # Initialize mediapipe hands module
        mp_hands = mp.solutions.hands
        # mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        data_aux_test = []
        data_test = []

        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # print(x)
                    data_aux_test.append(x)
                    data_aux_test.append(y)
            data_test.append(data_aux_test)
            max_length = 84
            # Pad the testing data to the maximum length
            padded_testing_data = data_aux_test + [0] * (max_length - len(data_aux_test))
            self.predict_number([padded_testing_data])

        else:
            print("Error: No hand landmarks detected in the image.")
            return None

    def perform_image_enhancement(self, img):
        # Perform image enhancement (e.g., contrast adjustment, sharpening, noise reduction)
        img_enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
        return img_enhanced

    def predict_number(self, preprocessed_image):
        # Load the trained model
        filename = 'modelRFM3.p'
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        # Extract the model from the dictionary
        model = data['model']
        # Debugging
        print("Preprocessed image data:", preprocessed_image)

        prediction = model.predict(preprocessed_image)
        print("Model prediction:", prediction)  # Debugging
        self.check_answer(prediction)

    def check_answer(self, prediction):
        # if self.question_index < len(self.questions):
        question_data = self.questions[self.question_index]
        correct_answer = question_data.get("correct_answer", "")
        # len_ques = len(self.questions);
        # Print the data types of prediction and correct_answer
        print("Data type of prediction:", type(prediction))
        print("Data type of correct_answer:", type(correct_answer))

        print("Correct answer:", correct_answer)
        if prediction == correct_answer:
            self.score += 1
        else:
            # If the answer is incorrect, append the question data to the list
            self.incorrect_questions.append(question_data)

        # Navigate to the next question or the result screen
        self.question_index += 1
        if self.question_index < len(self.questions):
            self.display_current_question()
        else:
            # Calculate the total score
            total_score = self.score

            # Generate a unique ID for the test
            test_id = str(uuid.uuid4())

            # Store the total score in the database
            self.store_test_score(test_id, total_score)
            # Navigate to the result screen
            self.manager.current = 'result_num'
            result_screen = self.manager.get_screen('result_num')
            result_screen.update_result(self.score, self.total_questions, self.incorrect_questions)
            self.incorrect_questions = []

    def store_test_score(self, test_id, total):
        # Connect to the SQLite database
        conn = sqlite3.connect('users_data.db')
        cursor = conn.cursor()
        res1 = get_display(arabic_reshaper.reshape("أرقام"))
        # Insert the user's test score into the 'test' table
        cursor.execute("INSERT INTO tests (id, score, username, testName) VALUES (?, ?, ?, ?)",
                       (test_id, total, self.username, "Numbers"))

        # Commit changes to the database
        conn.commit()

        # Close the connection
        conn.close()


class Result_alpha(Screen):
    res1 = StringProperty("")  # This will hold the reshaped Arabic result

    def update_result(self, text):
        # Reshape and render Arabic text properly for display
        reshaped_text = arabic_reshaper.reshape(text)  # Reshape the Arabic text
        self.res1 = get_display(reshaped_text)  # Get the display-ready text

        # Update the result button with reshaped Arabic text
        self.ids.result_button.text = self.res1  # Set text on the butto


class MyApp(MDApp):
    pass
    # def build(self):
      #  return Builder.load_string(Result_alpha)


    #def show_result(self, text):
        # Update the result screen with the text output from the model
     #   result_screen = self.root.get_screen('result_alpha')
      #  result_screen.update_result(text)





class Result_num(Screen):
    def update_result(self, score, total_questions, incorrect_questions):

        # Calculate and display the final score
        res1 = get_display(arabic_reshaper.reshape("نتيجتك "))
        self.ids.score_label.text = f"{score}/{total_questions} :{res1} "

        # Provide feedback based on the score
        if score == total_questions:
            feedback = get_display(arabic_reshaper.reshape(" رائع! لقد أجبت على جميع الإسئلة بشكل صحيح "))

        elif score >= total_questions * 0.5:
            feedback = get_display(arabic_reshaper.reshape(" عمل عظيم! لقد اجتزت الاختبار. "))

        else:
            feedback = get_display(arabic_reshaper.reshape(" يمكنك القيام بعمل أفضل. استمر بالتدريب! "))

        self.ids.feedback_label.text = feedback

        # Navigate to the IncorrectNumScreen and pass the list of incorrect questions
        incorrect_num_screen = self.manager.get_screen('incorrect_num')
        incorrect_num_screen.incorrectnum(incorrect_questions)
        self.manager.get_screen('incorrect_num')


class IncorrectAlphaScreen(Screen):

    def incorrectalpha(self, incorrect_questions):
        res1 = get_display(arabic_reshaper.reshape("السؤال"))

        questions_answers_text = ""

        # Concatenate all questions and answers
        if incorrect_questions:
            for i, question_data in enumerate(incorrect_questions, 1):
                questions_answers_text += f"{question_data['question']} :{i}{res1}\n\n"
            self.ids.question_answer.text = questions_answers_text


class IncorrectNumScreen(Screen):

    def incorrectnum(self, incorrect_questions):
        res1 = get_display(arabic_reshaper.reshape("السؤال"))

        questions_answers_text = ""
        # Concatenate all questions and answers
        if incorrect_questions:
            for i, question_data in enumerate(incorrect_questions, 1):
                questions_answers_text += f"{question_data['question']} :{i}{res1}\n\n"
            self.ids.question_answer.text = questions_answers_text


class History(Screen):
    def on_pre_enter(self):
        self.populate_data_list()

    def populate_data_list(self):
        # Connect to the database
        conn = sqlite3.connect('users_data.db')
        cursor = conn.cursor()

        # Fetch records from the tests table
        cursor.execute("SELECT testName,score FROM tests WHERE username=?", (self.username,))
        records = cursor.fetchall()

        # Close the database connection
        conn.close()

        # Clear any existing items in the MDList
        self.ids.data_list.clear_widgets()

        # Iterate over fetched records and create TwoLineListItem for each record
        for record in records:
            item = ThreeLineListItem(
                secondary_text=f"Category Name: {record[0]}\n, Score: {record[1]}"
            )

            self.ids.data_list.add_widget(item)


class Cry(Screen):
    pass


class Happy(Screen):
    pass


class Brave(Screen):
    pass


class Humble(Screen):
    pass


class Light(Screen):
    pass


class Heavy(Screen):
    pass


class Tired(Screen):
    pass


class Egoist(Screen):
    pass


class Active(Screen):
    pass


class Scared(Screen):
    pass


class Zero(Screen):
    pass


class One(Screen):
    pass


class Two(Screen):
    pass


class Three(Screen):
    pass


class Four(Screen):
    pass


class Five(Screen):
    pass


class Six(Screen):
    pass


class Seven(Screen):
    pass


class Eight(Screen):
    pass


class Nine(Screen):
    pass


class Letter1(Screen):
    pass


class Letter2(Screen):
    pass


class Letter3(Screen):
    pass


class Letter4(Screen):
    pass


class Letter5(Screen):
    pass


class Letter6(Screen):
    pass


class Letter7(Screen):
    pass


class Letter8(Screen):
    pass


class Letter9(Screen):
    pass


class Letter10(Screen):
    pass


class Letter11(Screen):
    pass


class Letter12(Screen):
    pass


class Letter13(Screen):
    pass


class Letter14(Screen):
    pass


class Letter15(Screen):
    pass


class Letter16(Screen):
    pass


class Letter17(Screen):
    pass


class Letter18(Screen):
    pass


class Letter19(Screen):
    pass


class Letter20(Screen):
    pass


class Letter21(Screen):
    pass


class Letter22(Screen):
    pass


class Letter23(Screen):
    pass


class Letter24(Screen):
    pass


class Letter25(Screen):
    pass


class Letter26(Screen):
    pass


class Letter27(Screen):
    pass


class Letter28(Screen):
    pass


sm = ScreenManager()
sm.add_widget(FirstWindow(name='first'))
sm.add_widget(SignInWindow(name='signin'))
sm.add_widget(SignUpWindow(name='signup'))
sm.add_widget(HomePage(name='homepage'))
sm.add_widget(UserAccount(name='useraccount'))
sm.add_widget(EditUserAccount(name='edituserinfo'))
sm.add_widget(Categories(name='categories'))
sm.add_widget(Translatepage(name='translate'))
sm.add_widget(Lessons1(name='lessons1'))
sm.add_widget(Lessons2(name='lessons2'))
sm.add_widget(Lessons3(name='lessons3'))
sm.add_widget(Test_alphabet(name='test_alphabet'))
sm.add_widget(Test_numbers(name='test_numbers'))
sm.add_widget(Q_alphabet(name='q_alphabet'))
sm.add_widget(Q_numbers(name='q_numbers'))
sm.add_widget(Result_alpha(name='result_alpha'))
sm.add_widget(Result_num(name='result_num'))
sm.add_widget(IncorrectNumScreen(name='incorrect_num'))
sm.add_widget(IncorrectAlphaScreen(name='incorrect_alpha'))
sm.add_widget(History(name='history'))
sm.add_widget(Cry(name='cry'))
sm.add_widget(Happy(name='happy'))
sm.add_widget(Tired(name='tired'))
sm.add_widget(Light(name='light'))
sm.add_widget(Humble(name='humble'))
sm.add_widget(Heavy(name='heavy'))
sm.add_widget(Brave(name='brave'))
sm.add_widget(Zero(name='zero'))
sm.add_widget(One(name='one'))
sm.add_widget(Two(name='two'))
sm.add_widget(Three(name='three'))
sm.add_widget(Four(name='four'))
sm.add_widget(Five(name='five'))
sm.add_widget(Six(name='six'))
sm.add_widget(Seven(name='seven'))
sm.add_widget(Eight(name='eight'))
sm.add_widget(Nine(name='nine'))
sm.add_widget(Cry(name='cry'))
sm.add_widget(Happy(name='happy'))
sm.add_widget(Tired(name='tired'))
sm.add_widget(Light(name='light'))
sm.add_widget(Humble(name='humble'))
sm.add_widget(Heavy(name='heavy'))
sm.add_widget(Brave(name='brave'))
sm.add_widget(Egoist(name='egoist'))
sm.add_widget(Active(name='active'))
sm.add_widget(Scared(name='scared'))
sm.add_widget(Zero(name='zero'))
sm.add_widget(One(name='one'))
sm.add_widget(Two(name='two'))
sm.add_widget(Three(name='three'))
sm.add_widget(Four(name='four'))
sm.add_widget(Five(name='five'))
sm.add_widget(Six(name='six'))
sm.add_widget(Seven(name='seven'))
sm.add_widget(Eight(name='eight'))
sm.add_widget(Nine(name='nine'))
sm.add_widget(Letter1(name='letter1'))
sm.add_widget(Letter2(name='letter2'))
sm.add_widget(Letter3(name='letter3'))
sm.add_widget(Letter4(name='letter4'))
sm.add_widget(Letter5(name='letter5'))
sm.add_widget(Letter6(name='letter6'))
sm.add_widget(Letter7(name='letter7'))
sm.add_widget(Letter8(name='letter8'))
sm.add_widget(Letter9(name='letter9'))
sm.add_widget(Letter10(name='letter10'))
sm.add_widget(Letter11(name='letter11'))
sm.add_widget(Letter12(name='letter12'))
sm.add_widget(Letter13(name='letter13'))
sm.add_widget(Letter14(name='letter14'))
sm.add_widget(Letter15(name='letter15'))
sm.add_widget(Letter16(name='letter16'))
sm.add_widget(Letter17(name='letter17'))
sm.add_widget(Letter18(name='letter18'))
sm.add_widget(Letter19(name='letter19'))
sm.add_widget(Letter20(name='letter20'))
sm.add_widget(Letter21(name='letter21'))
sm.add_widget(Letter22(name='letter22'))
sm.add_widget(Letter23(name='letter23'))
sm.add_widget(Letter24(name='letter24'))
sm.add_widget(Letter25(name='letter25'))
sm.add_widget(Letter26(name='letter26'))
sm.add_widget(Letter27(name='letter27'))
sm.add_widget(Letter28(name='letter28'))


class Murib(MDApp):
    # Initialize the analyzer
    analyzer = ArabicGrammarAnalyzer(
        api_key="EOQqBdGAQ866AVloO7CK3WFVtFqixZxoTBuWlyq1fgfu",
        project_id="09d6ad16-4d32-422b-9bae-86f3ead73c8d"
    )

    # Load and index the data
    analyzer.load_and_index_data('all_arabic_grammar_dataset.json')

    # Test sentence
   # test_sentence =  "اشتريتُ عشرين كتاباً علمياً من مكتبةِ الجامعةِ الجديدةًِ"

    # First analyze the input sentence to detect its types
   # sentence_data = analyzer.analyze_input_sentence(test_sentence)

    # Print detected types


    # Analyze the sentence
    #result = analyzer.analyze_sentence(sentence_data["sentence"], sentence_data)
   # print(result)


    def build(self):
        self.title = "Murib"
        # Reshape and display Arabic text
        text1 = "تسجيل الدخول"
        reshaped_text1 = arabic_reshaper.reshape(text1)
        res1 = get_display(reshaped_text1)

        text2 = " إنشاء حساب جديد "
        reshaped_text2 = arabic_reshaper.reshape(text2)
        res2 = get_display(reshaped_text2)

        text5 = " إعادة "
        reshaped_text5 = arabic_reshaper.reshape(text5)
        res5 = get_display(reshaped_text5)

        text7 = " ليس لديك حساب  ؟ "
        reshaped_text7 = arabic_reshaper.reshape(text7)
        res7 = get_display(reshaped_text7)

        text8 = "إنشاء حساب "
        reshaped_text8 = arabic_reshaper.reshape(text8)
        res8 = get_display(reshaped_text8)

        text9 = "حساب جديد "
        reshaped_text9 = arabic_reshaper.reshape(text9)
        res9 = get_display(reshaped_text9)

        text10 = "هل لديك حساب بالفعل ؟ "
        reshaped_text10 = arabic_reshaper.reshape(text10)
        res10 = get_display(reshaped_text10)

        text11 = " إختبار"
        reshaped_text11 = arabic_reshaper.reshape(text11)
        res11 = get_display(reshaped_text11)

        text12 = " إبدأ الإختبار"
        reshaped_text12 = arabic_reshaper.reshape(text12)
        res12 = get_display(reshaped_text12)

        text13 = " النتيجة "
        reshaped_text13 = arabic_reshaper.reshape(text13)
        res13 = get_display(reshaped_text13)

        text14 = "التالي"
        reshaped_text14 = arabic_reshaper.reshape(text14)
        res14 = get_display(reshaped_text14)

        text15 = "الحساب"
        reshaped_text15 = arabic_reshaper.reshape(text15)
        res15 = get_display(reshaped_text15)

        text16 = "سيتم التقاط الصورة بعد مضي 5 ثوان"
        reshaped_text16 = arabic_reshaper.reshape(text16)
        res16 = get_display(reshaped_text16)

        text17 = "أكمل الدرس "
        reshaped_text17 = arabic_reshaper.reshape(text17)
        res17 = get_display(reshaped_text17)

        text18 = "تعلم لغة الإشارة"
        reshaped_text18 = arabic_reshaper.reshape(text18)
        res18 = get_display(reshaped_text18)

        text19 = "سِجِل المُعرِبات "
        reshaped_text19 = arabic_reshaper.reshape(text19)
        res19 = get_display(reshaped_text19)

        text20 = "ترجمة لغة الإشارة الى نص "
        reshaped_text20 = arabic_reshaper.reshape(text20)
        res20 = get_display(reshaped_text20)

        text21 = "الحروف "
        reshaped_text21 = arabic_reshaper.reshape(text21)
        res21 = get_display(reshaped_text21)

        text22 = "لغة الإشارة السعودية "
        reshaped_text22 = arabic_reshaper.reshape(text22)
        res22 = get_display(reshaped_text22)

        text23 = "تسجيل الخروج "
        reshaped_text23 = arabic_reshaper.reshape(text23)
        res23 = get_display(reshaped_text23)

        text24 = "صفات وحالات "
        reshaped_text24 = arabic_reshaper.reshape(text24)
        res24 = get_display(reshaped_text24)

        text25 = "الأرقـــام"
        reshaped_text25 = arabic_reshaper.reshape(text25)
        res25 = get_display(reshaped_text25)

        text26 = "تحويل من لغة إشارة إلى نص  "
        reshaped_text26 = arabic_reshaper.reshape(text26)
        res26 = get_display(reshaped_text26)

        text27 = "النتيجة  "
        reshaped_text27 = arabic_reshaper.reshape(text27)
        res27 = get_display(reshaped_text27)

        text28 = "الإرسال  "
        reshaped_text28 = arabic_reshaper.reshape(text28)
        res28 = get_display(reshaped_text28)

        text29 = "تعليمات الاختبار:  "
        reshaped_text29 = arabic_reshaper.reshape(text29)
        res29 = get_display(reshaped_text29)

        text30 = "1- هناك 10 أسئلة لكل الأرقام التي تعلمتها.  "
        reshaped_text30 = arabic_reshaper.reshape(text30)
        res30 = get_display(reshaped_text30)

        text31 = "2- سيُطلب منك فتح الكاميرا لتمثيل الرقم.    "
        reshaped_text31 = arabic_reshaper.reshape(text31)
        res31 = get_display(reshaped_text31)

        text32 = "3- لا يوجد وقت محدد للاختبار، ويمكن إعادته أكثر من مرة."
        reshaped_text32 = arabic_reshaper.reshape(text32)
        res32 = get_display(reshaped_text32)

        text33 = "اختبار الحروف "
        reshaped_text33 = arabic_reshaper.reshape(text33)
        res33 = get_display(reshaped_text33)

        text34 = "اختبار الأرقام "
        reshaped_text34 = arabic_reshaper.reshape(text34)
        res34 = get_display(reshaped_text34)

        text35 = "معرفة الاخطاء "
        reshaped_text35 = arabic_reshaper.reshape(text35)
        res35 = get_display(reshaped_text35)

        text36 = "سعيد "
        reshaped_text36 = arabic_reshaper.reshape(text36)
        res36 = get_display(reshaped_text36)

        text37 = "شجاع "
        reshaped_text37 = arabic_reshaper.reshape(text37)
        res37 = get_display(reshaped_text37)

        text38 = "خفيف "
        reshaped_text38 = arabic_reshaper.reshape(text38)
        res38 = get_display(reshaped_text38)

        text39 = "ثقيل "
        reshaped_text39 = arabic_reshaper.reshape(text39)
        res39 = get_display(reshaped_text39)

        text40 = "يبكي "
        reshaped_text40 = arabic_reshaper.reshape(text40)
        res40 = get_display(reshaped_text40)

        text41 = "تواضع"
        reshaped_text41 = arabic_reshaper.reshape(text41)
        res41 = get_display(reshaped_text41)

        text42 = "متعب"
        reshaped_text42 = arabic_reshaper.reshape(text42)
        res42 = get_display(reshaped_text42)

        text43 = "حساب المستخدم"
        reshaped_text43 = arabic_reshaper.reshape(text43)
        res43 = get_display(reshaped_text43)

        text44 = "تعديل معلوماتي"
        reshaped_text44 = arabic_reshaper.reshape(text44)
        res44 = get_display(reshaped_text44)

        text45 = "مرحبًا"
        reshaped_text45 = arabic_reshaper.reshape(text45)
        res45 = get_display(reshaped_text45)

        text46 = " حفظ التعديل "
        reshaped_text46 = arabic_reshaper.reshape(text46)
        res46 = get_display(reshaped_text46)

        text51 = "جميع الدروس > "
        reshaped_text51 = arabic_reshaper.reshape(text51)
        res51 = get_display(reshaped_text51)

        text52 = "1- هناك 28 سؤال لكل الحروف التي تعلمتها.  "
        reshaped_text52 = arabic_reshaper.reshape(text52)
        res52 = get_display(reshaped_text52)

        text53 = "2- سيُطلب منك فتح الكاميرا لتمثيل الحرف.   "
        reshaped_text53 = arabic_reshaper.reshape(text53)
        res53 = get_display(reshaped_text53)

        text54 = "الغاء"
        reshaped_text54 = arabic_reshaper.reshape(text54)
        res54 = get_display(reshaped_text54)

        text55 = "سجل المُعرِبات  "
        reshaped_text55 = arabic_reshaper.reshape(text55)
        res55 = get_display(reshaped_text55)

        text56 = "أعرِب"
        reshaped_text56 = arabic_reshaper.reshape(text56)
        res56 = get_display(reshaped_text56)

        text57 = "أدخل الجملة التي تريد إعرابها"
        reshaped_text57 = arabic_reshaper.reshape(text57)
        res57 = get_display(reshaped_text57)

        # Store the reshaped text in the Screen class for access in KV file
        FirstWindow.res1 = res1
        FirstWindow.res2 = res2
        SignInWindow.res1 = res1
        SignInWindow.res7 = res7
        SignInWindow.res8 = res8
        SignUpWindow.res9 = res9
        SignUpWindow.res2 = res2
        SignUpWindow.res1 = res1
        SignUpWindow.res10 = res10
        Lessons1.res11 = res11
        Cry.res11 = res11
        Happy.res11 = res11
        Heavy.res11 = res11
        Light.res11 = res11
        Tired.res11 = res11
        Humble.res11 = res11
        Brave.res11 = res11
        Egoist.res11 = res11
        Active.res11 = res11
        Scared.res11 = res11
        Lessons1.res18 = res18
        Lessons1.res24 = res24
        Cry.res24 = res24
        Happy.res24 = res24
        Heavy.res24 = res24
        Light.res24 = res24
        Tired.res24 = res24
        Humble.res24 = res24
        Brave.res24 = res24
        Egoist.res24 = res24
        Active.res24 = res24
        Scared.res24 = res24
        Lessons2.res11 = res11
        Lessons2.res21 = res21
        Lessons2.res18 = res18
        Lessons3.res11 = res11
        Lessons3.res18 = res18
        Lessons3.res25 = res25
        Zero.res18 = res18
        One.res18 = res18
        Two.res18 = res18
        Three.res18 = res18
        Four.res18 = res18
        Five.res18 = res18
        Six.res18 = res18
        Seven.res18 = res18
        Eight.res18 = res18
        Nine.res18 = res18
        Zero.res25 = res25
        One.res25 = res25
        Two.res25 = res25
        Three.res25 = res25
        Four.res25 = res25
        Five.res25 = res25
        Six.res25 = res25
        Seven.res25 = res25
        Eight.res25 = res25
        Nine.res25 = res25
        Letter1.res18 = res18
        Letter2.res18 = res18
        Letter3.res18 = res18
        Letter4.res18 = res18
        Letter5.res18 = res18
        Letter6.res18 = res18
        Letter7.res18 = res18
        Letter8.res18 = res18
        Letter9.res18 = res18
        Letter10.res18 = res18
        Letter11.res18 = res18
        Letter12.res18 = res18
        Letter13.res18 = res18
        Letter14.res18 = res18
        Letter15.res18 = res18
        Letter16.res18 = res18
        Letter17.res18 = res18
        Letter18.res18 = res18
        Letter19.res18 = res18
        Letter20.res18 = res18
        Letter21.res18 = res18
        Letter22.res18 = res18
        Letter23.res18 = res18
        Letter24.res18 = res18
        Letter25.res18 = res18
        Letter26.res18 = res18
        Letter27.res18 = res18
        Letter28.res18 = res18
        Letter1.res21 = res21
        Letter2.res21 = res21
        Letter3.res21 = res21
        Letter4.res21 = res21
        Letter5.res21 = res21
        Letter6.res21 = res21
        Letter7.res21 = res21
        Letter8.res21 = res21
        Letter9.res21 = res21
        Letter10.res21 = res21
        Letter11.res21 = res21
        Letter12.res21 = res21
        Letter13.res21 = res21
        Letter14.res21 = res21
        Letter15.res21 = res21
        Letter16.res21 = res21
        Letter17.res21 = res21
        Letter18.res21 = res21
        Letter19.res21 = res21
        Letter20.res21 = res21
        Letter21.res21 = res21
        Letter22.res21 = res21
        Letter23.res21 = res21
        Letter24.res21 = res21
        Letter25.res21 = res21
        Letter26.res21 = res21
        Letter27.res21 = res21
        Letter28.res21 = res21
        Translatepage.res22 = res22
        Translatepage.res26 = res26
        Translatepage.res27 = res27
        Translatepage.res28 = res28

        Test_alphabet.res12 = res12

        Test_numbers.res12 = res12

        Result_alpha.res13 = res13
        Result_num.res13 = res13
        Q_alphabet.res14 = res14
        Q_numbers.res14 = res14
        Q_alphabet.res16 = res16
        Q_numbers.res16 = res16
        Translatepage.res14 = res14
        HomePage.res15 = res15
        HomePage.res17 = res17
        HomePage.res18 = res18
        HomePage.res19 = res19
        HomePage.res20 = res20
        HomePage.res56 = res56
        HomePage.res57 = res57
        UserAccount.res23 = res23
        Lessons1.res21 = res20

        Test_alphabet.res29 = res29
        Test_alphabet.res52 = res52
        Test_alphabet.res53 = res53
        Test_alphabet.res32 = res32
        Test_alphabet.res33 = res33

        Test_numbers.res29 = res29
        Test_numbers.res30 = res30
        Test_numbers.res31 = res31
        Test_numbers.res32 = res32
        Test_numbers.res34 = res34

        Result_alpha.res35 = res35
        Result_alpha.res5 = res5
        Result_num.res35 = res35
        Result_num.res5 = res5
        # قائمة الليست

        Categories.res36 = res36
        Categories.res37 = res37
        Categories.res38 = res38
        Categories.res39 = res39
        Categories.res40 = res40
        Categories.res41 = res41
        Categories.res42 = res42

        # لحساب المستخدم

        UserAccount.res43 = res43
        UserAccount.res44 = res44
        UserAccount.res45 = res45
        EditUserAccount.res44 = res44
        EditUserAccount.res46 = res46
        EditUserAccount.res54 = res54

        # History
        History.res55 = res55
        HomePage.res51 = res51

        screen = Builder.load_file("FirstWindow.kv")
        return screen


if __name__ == '__main__':
    Murib().run()

