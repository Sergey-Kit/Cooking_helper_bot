import os
import re
import json
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI

# Load environment variables
load_dotenv('.env')

# Set proxy if needed
proxy_url = os.getenv("PROXY_URL")
os.environ["HTTP_PROXY"] = proxy_url
os.environ["HTTPS_PROXY"] = proxy_url

# OpenAI API key
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

faiss_url = 'http://127.0.0.1:8000/search/'


# Pydantic модели для структуры входных и выходных данных
class UserRequest(BaseModel):
    user_input: str

class IngredientsResponse(BaseModel):
    ingredients: List[str]
    dislikedFood: List[str]

class Recipe(BaseModel):
    title: str
    link: str
    ingredients: List[str]

class FilteredRecipesResponse(BaseModel):
    filtered_recipes: List[Recipe]

# System prompt для извлечения ингредиентов
system_prompt_ingredients = """Ты — помощник по кулинарии. Тебе нужно:

1. Проверить, относится ли запрос пользователя к теме кулинарии и тому, что можно приготовить из перечисленных ингредиентов.

2. Если запрос относится к теме, извлечь ингредиенты и запрещенные продукты из текста пользователя и вернуть их строго в формате JSON без какого-либо дополнительного текста или форматирования.

3. Если запрос не относится к теме, вернуть следующий JSON:

{
  "not_relevant": true
}

**Формат ответа для релевантного запроса:**

{
  "ingredients": [
    "<ингредиент_1>",
    "<ингредиент_2>",
    ...
  ],
  "dislikedFood": [
    "<запрещенный_продукт_1>",
    "<запрещенный_продукт_2>",
    ...
  ]
}

**Требования:**

- Ответ должен быть **только** JSON, без каких-либо пояснений, комментариев или форматирования.

- **Не включай** кодовые блоки или обратные кавычки.

- Убедись, что JSON корректно отформатирован и парсится без ошибок.
"""



# System prompt для фильтрации рецептов
system_prompt_filter = """Ты — помощник по кулинарии. Тебе нужно отфильтровать рецепты, которые соответствуют запросу пользователя, и вернуть их в строго структурированном формате JSON без какого-либо дополнительного текста.

Формат ответа:

{"filtered_recipes": [{"title": "<название_рецепта>", "link": "<ссылка_на_рецепт>", "ingredients": ["<ингредиент_1>", "<ингредиент_2>", ...]}, ...]}

Требования:

- Ответ должен быть только JSON, без каких-либо пояснений или дополнительного текста.
- Убедись, что JSON корректно отформатирован и парсится без ошибок.
"""


class RecipeBot:
    def __init__(self):
        self.api_key = LANGCHAIN_API_KEY
        self.proxy = {
            'http://': proxy_url,
            'https://': proxy_url,
        }

    async def _send_to_openai(self, system_prompt: str, user_input: str):
        """Asynchronously send a request to OpenAI using httpx."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        data = {
            "model": "gpt-4o-2024-08-06",  # or "gpt-3.5-turbo" if needed
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": 0,
        }
        async with httpx.AsyncClient(proxies=self.proxy) as client:
            response = await client.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30.0,
            )
        response.raise_for_status()
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        print("Assistant's response:")
        print(assistant_message)
        
        # Remove code block markers if present
        assistant_message_clean = assistant_message.strip()
        if assistant_message_clean.startswith("```"):
            # Remove the starting triple backticks and optional language identifier
            assistant_message_clean = re.sub(r'^```[a-z]*\n', '', assistant_message_clean)
            # Remove the ending triple backticks
            assistant_message_clean = assistant_message_clean.rstrip("`").rstrip()
        
        # Parse the assistant's response
        try:
            parsed_response = json.loads(assistant_message_clean)
        except json.JSONDecodeError as e:
            print(f"Error parsing assistant's response: {e}")
            parsed_response = {}
        return parsed_response

    async def _extract_ingredients_from_response(self, user_input: str):
        """Асинхронно извлекает ингредиенты и запрещенные продукты или определяет нерелевантность запроса."""
        response = await self._send_to_openai(system_prompt_ingredients, user_input)
        if response.get('not_relevant'):
            print('Запрос не по тематике.')
            return None, None, True  # Возвращаем флаг нерелевантности
        ingredients = response.get('ingredients', [])
        disliked_food = response.get('dislikedFood', [])
        print('Ingredients:', ingredients)
        print('Disliked food:', disliked_food)
        return ingredients, disliked_food, False

    def _mock_faiss_response(self, query):
        """Заглушка для сервиса FAISS. Возвращает фиктивные данные."""
        if "мороженое" in query or "фрукты" in query:
            return [
                {
                    "title": "Grilled peaches drizzled with honey",
                    "ingredients": [
                        "6 peaches",
                        "1 tbsp honey per peach half",
                        "1 scoop vanilla ice cream"
                    ],
                    "link": "https://cookpad.com/us/recipes/336479-grilled-peaches-drizzled-with-honey"
                },
                {
                    "title": "Cafe-style Banana Juice",
                    "ingredients": [
                        "70 grams Banana",
                        "125 ml Milk",
                        "50 ml Vanilla ice cream",
                        "1 Sugar or honey"
                    ],
                    "link": "https://cookpad.com/us/recipes/149343-cafe-style-banana-juice"
                }
            ]
        elif "foijafoiewf" in query:
            return {"detail": "No recipes found."}
        else:
            return []  # Пустой список для других запросов
    
    async def _mock_send_to_faiss(self, ingredients):
        """Временно использует заглушку вместо реального запроса к FAISS."""
        # Преобразуем список ингредиентов в строку с запятыми
        query_string = ', '.join(ingredients)
        print(f"FAISS mock query: {query_string}")
        # Используем заглушку вместо реального запроса
        response = self._mock_faiss_response(query_string)
        if isinstance(response, dict) and "detail" in response:
            print(f"FAISS mock error: {response['detail']}")
            return []
        return response
    
    async def _send_to_faiss(self, ingredients):
        """Асинхронно отправляет запрос к сервису FAISS с заданными ингредиентами."""
        # Преобразуем список ингредиентов в строку с запятыми
        query_string = ', '.join(ingredients)
        faiss_url = faiss_url  # URL вашего сервиса FAISS

        # Подготовка данных для запроса
        data = {
            'query': query_string
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    faiss_url,
                    json=data,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list):
                    # FAISS вернул список рецептов
                    return result
                elif 'detail' in result:
                    # FAISS вернул сообщение об ошибке
                    print(f"FAISS error: {result['detail']}")
                    return []
                else:
                    # Непредвиденный ответ от FAISS
                    print("FAISS вернул непредвиденный ответ.")
                    return []
            except httpx.HTTPError as e:
                print(f"Ошибка при подключении к сервису FAISS: {e}")
                return []


    async def _filter_recipes_by_user_request(self, user_input: str, recipes):
        """Асинхронно фильтрует рецепты на основе запроса пользователя."""
        ingredients, forbidden_ingredients, _ = await self._extract_ingredients_from_response(user_input)

        recipes_str = ""
        for recipe in recipes:
            ingredients_str = ', '.join(recipe['ingredients'])
            recipes_str += f"Рецепт: {recipe['title']}. Ингредиенты: {ingredients_str}. Ссылка: {recipe['link']}.\n"

        check_input = (
            f"Пользователь сказал: '{user_input}'. Проверь, пожалуйста, подходит ли каждый рецепт из списка, "
            f"учитывая, что пользователь не хочет использовать ингредиенты: {', '.join(forbidden_ingredients)}. "
            f"Вот список рецептов:\n{recipes_str}"
        )

        response = await self._send_to_openai(system_prompt_filter, check_input)
        filtered_recipes_data = response.get('filtered_recipes', [])
        filtered_recipes = [Recipe(**recipe) for recipe in filtered_recipes_data]
        return filtered_recipes

    async def handle_request(self, user_input: str):
        """Основной метод для асинхронной обработки запроса пользователя."""
        ingredients, forbidden_ingredients, not_relevant = await self._extract_ingredients_from_response(user_input)
        if not_relevant:
            # Возвращаем стандартный ответ
            return {"message": "Вопрос не по тематике."}
        recipes = await self._mock_send_to_faiss(ingredients) # todo: замени на _send_to_faiss
        if not recipes:
            # Если рецептов нет, возвращаем сообщение
            return {"message": "Не удалось найти рецепты по заданным ингредиентам."}
        filtered_recipes = await self._filter_recipes_by_user_request(user_input, recipes)
        return {"filtered_recipes": filtered_recipes}




# Создаем приложение FastAPI
app = FastAPI()

@app.post("/recipes")
async def get_recipes(request: UserRequest):
    bot = RecipeBot()
    result = await bot.handle_request(request.user_input)
    if "message" in result:
        return {"message": result["message"]}
    else:
        return FilteredRecipesResponse(filtered_recipes=result["filtered_recipes"])




