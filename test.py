from FlagEmbedding import FlagModel
sentences_1 = ["a muddled noise of broken channel of the tv", "a person slowly clears high growth vegetation outdoors"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="Represent the audio caption:",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)

print(embeddings_1.shape)
print(type(embeddings_1))
print(embeddings_1.device)