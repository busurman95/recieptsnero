import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

# Веб-скрапер для извлечения рецептов с сайта
def scrape_recipes(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    recipes = []
    for recipe in soup.find_all('div', class_='recipe'):
        title = recipe.find('h2').text
        ingredients = [ingredient.text for ingredient in recipe.find_all('li', class_='ingredient')]
        recipes.append((title, ingredients))
    return recipes

# Предварительная обработка данных и обучение нейронной сети
def train_model(recipes):
    # Векторизация ингредиентов
    ingredients = [' '.join(recipe[1]) for recipe in recipes]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(ingredients)
    ingredient_sequences = tokenizer.texts_to_sequences(ingredients)

    # Создание нейронной сети
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128),
        LSTM(128),
        Dense(len(recipes), activation='softmax')
    ])

    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Обучение модели
    recipe_titles = [recipe[0] for recipe in recipes]
    model.fit(ingredient_sequences, recipe_titles, epochs=10, batch_size=32)

    return model, tokenizer

# Функция для предсказания рецептов
def predict_recipe(model, tokenizer, ingredients):
    user_ingredients = ' '.join(ingredients)
    user_ingredients_seq = tokenizer.texts_to_sequences([user_ingredients])
    predicted_recipe_index = model.predict(user_ingredients_seq).argmax()
    predicted_recipe = recipes[predicted_recipe_index][0]
    return predicted_recipe
