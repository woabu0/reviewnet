from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "Machine learning makes machines learn from data. Learning is key in machine learning."

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
