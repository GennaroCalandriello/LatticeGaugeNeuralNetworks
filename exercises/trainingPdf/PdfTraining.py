import PyPDF2
import re
from transformers import GPT2Tokenizer, GPT2Config, TFGPT2LMHeadModel
import numpy as np
import tensorflow as tf


# 1. Convert the PDF to text
def convertPdfToText(file):
    pdfFile = open(file, "rb")
    pdfReader = PyPDF2.PdfReader(pdfFile)
    text = ""
    for numPage in range(len(pdfReader.pages)):
        page_obj = pdfReader.pages[numPage]
        text += page_obj.extract_text()
    pdfFile.close()
    return text


# 2. preprocess the text
def preprocessText(text):
    # Remove special characters
    text = re.sub(r"\W", " ", text)
    # Remove single characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    # Substituting multiple spaces with single space
    text = re.sub(r"\s+", " ", text, flags=re.I)
    # Removing prefixed 'b'
    text = re.sub(r"^b\s+", "", text)
    # Converting to Lowercase
    text = text.lower()

    return text


def tonenizationAndTrain(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)

    # format data into I/O sequences
    I_sequences = []
    O_sequences = []
    sequence_length = 100

    for i in range(len(tokens) - sequence_length):
        I_sequences.append(tokens[i : i + sequence_length])
        O_sequences.append(tokens[i + sequence_length])

    I_sequences = np.array(I_sequences)
    O_sequences = np.array(O_sequences)

    # Initialize the model
    configuration = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
    model = TFGPT2LMHeadModel(configuration)

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    # train the model
    model.fit(I_sequences, O_sequences, epochs=5, batch_size=32)
    model.save("GPTPDF.h5")


if __name__ == "__main__":
    path = "prova.pdf"
    text = convertPdfToText(path)
    print("Converted PDF to txt")
    text = preprocessText(text)
    print("Preprocessed text in modo miserrimo")
    tonenizationAndTrain(text[: int(len(text) / 2)])
    print("Model Trained")

    print(preprocessText(text))
