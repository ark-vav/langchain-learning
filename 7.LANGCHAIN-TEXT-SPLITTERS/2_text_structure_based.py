from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Artificial intelligence has rapidly evolved from a niche field of computer science into a transformative technology
used across industries. Modern AI systems are capable of understanding natural language, recognizing images, generating creative
content, and assisting with complex decision-making. Businesses leverage AI to automate repetitive tasks, improve customer support 
through chatbots, optimize supply chains, and analyze vast amounts of data for actionable insights. As models become more powerful, 
developers are increasingly focusing on building reliable, scalable, and responsible AI applications that can operate in real-world
environments while maintaining transparency, privacy, and security. """

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)


print(len(chunks))
print(chunks)

#11
# ['Artificial intelligence has rapidly evolved from a niche field of computer science into a', 
# 'transformative technology', 'used across industries. Modern AI systems are capable of understanding natural language,',
#  'recognizing images, generating creative', 'content, and assisting with complex decision-making.
#  Businesses leverage AI to automate repetitive', 'tasks, improve customer support', 'through chatbots, 
# optimize supply chains, and analyze vast amounts of data for actionable insights.', 
# 'As models become more powerful,', 'developers are increasingly focusing on building reliable, scalable, and responsible AI', 
# 'applications that can operate in real-world', 'environments while maintaining transparency, privacy, and security.']