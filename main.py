import json
import requests
from typing import List, Dict, Set
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from rouge_score import rouge_scorer


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
        """Create a prompt for grammatical analysis"""
        prompt = f"""<|system|>
    محلل نحوي متخصص في التحليل النحوي للجمل العربية. أقوم بتحديد:
    ١. نوع الجملة ونمطها النحوي
    ٢. إعراب كل كلمة من حيث:
       - موقعها الإعرابي (مبتدأ، خبر، فاعل، مفعول به، الخ)
       - حالتها الإعرابية (رفع، نصب، جر)
       - علامة إعرابها
       - سبب العلامة الإعرابية

    <|human|>
    حلل الجملة التالية نحوياً: {query_sentence}

    <|assistant|>
    التحليل النحوي المفصل:

    نوع الجملة: جملة اسمية (مبتدأ وخبر)

    إعراب المفردات:
    """
        return prompt

    def _format_response(self, response: str) -> str:
        """Format the model's response with improved parsing"""
        try:
            print("\nStarting response formatting...")

            # Split the response into lines and clean them
            lines = [line.strip() for line in response.split('\n') if line.strip()]

            # Debug print
            print(f"Found {len(lines)} lines in response")

            formatted_lines = []

            for line in lines:
                # Skip unwanted lines
                if any(skip in line for skip in ['إذاً', 'فلا تتردد', 'مزيد من التوضيح']):
                    continue

                # Add lines that contain Arabic analysis (numbered items or headers)
                if (any('\u0600' <= c <= '\u06FF' for c in line) and  # Contains Arabic
                        ('(' in line or 'نوع الجملة' in line or 'التحليل النحوي' in line)):  # Contains analysis
                    formatted_lines.append(line.strip())

            # Join the formatted lines and add proper spacing
            result = '\n'.join(formatted_lines)

            # Debug print
            print(f"Formatted {len(formatted_lines)} lines")

            if not result.strip():
                print("Warning: No formatted content generated")
                return "لم يتم العثور على تحليل مناسب للجملة"

            return result

        except Exception as e:
            print(f"Error in formatting response: {e}")
            return str(response)  # Return raw response if formatting fails

    def analyze_sentence(self, sentence: str, sentence_data: Dict) -> str:
        """Analyze Arabic sentence"""
        # Find similar sentences and create prompt
        similar_sentences = self.find_similar_sentences(sentence_data)
        prompt = self.create_prompt_with_examples(sentence, similar_sentences)

        # Get and return raw model response
        return self._query_model(prompt)

    def _query_model(self, prompt: str) -> str:
        """Query the model with improved parameters"""
        request_url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 1000,
                "min_new_tokens": 100,  # Add minimum tokens
                "repetition_penalty": 1.2,
                "temperature": 0.7,
                "top_p": 0.9
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
            result = response.json()['results'][0]['generated_text']
            if not result:
                return "عذراً، لم يتم الحصول على تحليل من النموذج"
            return result
        else:
            return f"Error: {response.status_code}, {response.text}"

    def analyze_multiple_sentences(self, sentences: List[str]) -> Dict:
        """Analyze multiple sentences and save outputs in the correct format"""
        system_outputs = {
            "sentences": []
        }

        for sentence in sentences:
            print(f"\nAnalyzing: {sentence}")
            # Analyze the sentence
            sentence_data = self.analyze_input_sentence(sentence)
            analysis = self.analyze_sentence(sentence, sentence_data)

            # Add to outputs
            system_outputs["sentences"].append({
                "sentence": sentence,
                "analysis": analysis
            })

        # Save outputs to JSON
        with open('system_outputs.json', 'w', encoding='utf-8') as f:
            json.dump(system_outputs, f, ensure_ascii=False, indent=4)

        return system_outputs


def evaluate_rouge(system_file: str, reference_file: str):
    """Evaluate system outputs against reference using ROUGE-1"""
    # Load files
    with open(system_file, 'r', encoding='utf-8') as f:
        system_data = json.load(f)
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    # Calculate scores for each sentence
    scores = []
    for sys, ref in zip(system_data['sentences'], reference_data['sentences']):
        assert sys['sentence'] == ref['sentence'], "Sentences don't match"

        score = scorer.score(ref['analysis'], sys['analysis'])
        scores.append({
            'sentence': sys['sentence'],
            'rouge1_precision': score['rouge1'].precision,
            'rouge1_recall': score['rouge1'].recall,
            'rouge1_fmeasure': score['rouge1'].fmeasure
        })

    # Calculate average scores
    avg_precision = sum(s['rouge1_precision'] for s in scores) / len(scores)
    avg_recall = sum(s['rouge1_recall'] for s in scores) / len(scores)
    avg_fmeasure = sum(s['rouge1_fmeasure'] for s in scores) / len(scores)

    return {
        'individual_scores': scores,
        'average_scores': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_fmeasure
        }
    }


def main():
    try:
        # Initialize the analyzer
        print("Initializing analyzer...")
        analyzer = ArabicGrammarAnalyzer(
            api_key="EOQqBdGAQ866AVloO7CK3WFVtFqixZxoTBuWlyq1fgfu",
            project_id="09d6ad16-4d32-422b-9bae-86f3ead73c8d"
        )

        # Load and index the data
        print("Loading data...")
        analyzer.load_and_index_data('all_arabic_grammar_dataset.json')

        # List of test sentences
        test_sentences = [
            "ماءُ العينِ مالح",
            "نفدتِ البضاعةُ من عندِنا",
            "عُقدَ الاجتماعُ في العاصمةِ"
        ]

        # Analyze all sentences and save outputs
        system_outputs = analyzer.analyze_multiple_sentences(test_sentences)

        # Run ROUGE evaluation
        results = evaluate_rouge('system_outputs.json', 'reference.json')

        print("\nAverage ROUGE-1 Scores:")
        print(f"Precision: {results['average_scores']['precision']:.4f}")
        print(f"Recall: {results['average_scores']['recall']:.4f}")
        print(f"F1: {results['average_scores']['f1']:.4f}")

        print("\nIndividual Sentence Scores:")
        for score in results['individual_scores']:
            print(f"\nSentence: {score['sentence']}")
            print(f"Precision: {score['rouge1_precision']:.4f}")
            print(f"Recall: {score['rouge1_recall']:.4f}")
            print(f"F1: {score['rouge1_fmeasure']:.4f}")

    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()