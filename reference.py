import json

# Create the reference data
reference_data = {
    "sentences": [
        {
            "sentence": "ماءُ العينِ مالح",
            "analysis": """1. "ماءُ": مبتدأ مرفوع وعلامة رفعه الضمة الظاهرة وهو مضاف
2. "العينِ": مضاف إليه مجرور وعلامة جره الكسرة الظاهرة
3. "مالح": خبر مرفوع وعلامة رفعه الضمة الظاهرة

نوع الجملة: جملة اسمية (مبتدأ وخبر)
النمط: مبتدأ (مضاف) + مضاف إليه + خبر"""
        },
        {
            "sentence": "نفدتِ البضاعةُ من عندِنا",
            "analysis": """1. "نفدت": فعل ماضٍ مبني على الفتح
2. "البضاعةُ": فاعل مرفوع وعلامة رفعه الضمة الظاهرة
3. "من": حرف جر مبني على السكون
4. "عندِ": ظرف مكان مجرور بمن وعلامة جره الكسرة وهو مضاف
5. "نا": ضمير متصل مبني في محل جر مضاف إليه

نوع الجملة: جملة فعلية
النمط: فعل + فاعل + جار ومجرور + ظرف مكان مضاف + ضمير متصل"""
        },
        {
            "sentence": "عُقدَ الاجتماعُ في العاصمةِ",
            "analysis": """1. "عُقدَ": فعل ماضٍ مبني للمجهول مبني على الفتح
2. "الاجتماعُ": نائب فاعل مرفوع وعلامة رفعه الضمة الظاهرة
3. "في": حرف جر مبني على السكون
4. "العاصمةِ": اسم مجرور وعلامة جره الكسرة الظاهرة

نوع الجملة: جملة فعلية مبنية للمجهول
النمط: فعل مبني للمجهول + نائب فاعل + جار ومجرور"""
        }
    ]
}

# Save to JSON file with proper Arabic encoding
with open('reference.json', 'w', encoding='utf-8') as f:
    json.dump(reference_data, f, ensure_ascii=False, indent=4)

# To verify the saved data, you can read it back:
def read_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

# Test reading the file
saved_data = read_json('reference.json')
print("File saved and loaded successfully!")

# Print first sentence as a test
print("\nFirst sentence:")
print(saved_data['sentences'][0]['sentence'])
print("\nIts analysis:")
print(saved_data['sentences'][0]['analysis'])