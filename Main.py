import TextSummariser
import Server

word_embeddings = TextSummariser.extractWordEmbeddings()
Server.listen(word_embeddings)
