'''from ai4bharat.transliteration import XlitEngine

e = XlitEngine("hi", beam_width=10, rescore=True)
out = e.translit_word("namasthe", topk=5)
print(out)

e = XlitEngine("ta", beam_width=10)
out = e.translit_sentence("vanakkam ulagam")
print(out)
# output: {'ta': 'வணக்கம் உலகம்'}'''

from ai4bharat.transliteration import XlitEngine

INPUT_WORD = "namaste kaise hai aap"
print("Input word:", INPUT_WORD)


xlit_engine = XlitEngine("hi")
res = xlit_engine.translit_sentence(INPUT_WORD)
print("Hindi output:", res)


e = XlitEngine("ta", beam_width=10)
out = e.translit_sentence("vanakkam ulagam enda morakadi")
print(out)