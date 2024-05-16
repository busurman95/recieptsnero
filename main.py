from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivymd.uix.behaviors import RoundedRectangularElevationBehavior
from kivymd.uix.button import MDFillRoundFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from recipe_predictor import scrape_recipes, train_model, predict_recipe

Builder.load_file('mykv.kv')



# Глобальные переменные для модели и tokenizer
model = None
tokenizer = None

@app.route('/')
def index():
    # ... ваш существующий код ...
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer

    # Если модель и tokenizer еще не обучены, обучаем их
    if model is None or tokenizer is None:
        recipe_site_url = 'https://www.russianfood.com/recipes/bytype/?fid=99'
        recipes = scrape_recipes(recipe_site_url)
        model, tokenizer = train_model(recipes)

    # Получаем ингредиенты от пользователя
    user_ingredients = request.form['ingredients'].split(',')

    # Предсказываем рецепт на основе ингредиентов
    predicted_recipe = predict_recipe(model, tokenizer, user_ingredients)

    return render_template('predict.html', predicted_recipe=predicted_recipe)


class IngredientButton(MDFillRoundFlatButton):
    k = 0

    def click(self):
        if self.md_bg_color == [240 / 255, 240 / 255, 240 / 255, 1]:
            self.md_bg_color = [250 / 255, 250 / 255, 180 / 255, 1]
            self.text_color = [1, 0, 0, 1]
            boiler.add(self.text)
            MyApp.SM.screens[1].ids.checklist.text = set_as_column(boiler)
        else:
            self.md_bg_color = [240 / 255, 240 / 255, 240 / 255, 1]
            self.text_color = [128 / 255, 128 / 255, 128 / 255, 1]
            boiler.remove(self.text)
            MyApp.SM.screens[1].ids.checklist.text = set_as_column(boiler)

    def set_data(self, ingredient_button, text):
        ingredient_button.text = text
        ingredient_button.id = str(self.k)
        Ingredients.links_to_buttons.append(ingredient_button)
        self.k += 1


class BottomCard(MDCard, RoundedRectangularElevationBehavior):
    pass


class ReceiptCard(MDCard, RoundedRectangularElevationBehavior):

    def set_data(self, title, set_of_products, string_of_receipt):
        self.children[0].text = set_as_column(set_of_products)
        Receipts.links_to_cards.append(self)
        self.children[1].text = title
        self.bind(on_release=lambda x: self.show_full_receipt(title, set_of_products, string_of_receipt))

    def click(self):
        self.md_bg_color = [240 / 255, 240 / 255, 240 / 255, 1]

    @staticmethod
    def show_full_receipt(title, set_of_products, string_of_receipt):
        MyApp.SM.screens[3].ids.img.source = title + '.jpg'
        print(MyApp.SM.screens[3].ids.img.source)
        MyApp.SM.screens[3].ids.title.text = title
        MyApp.SM.screens[3].ids.description.text = set_as_column(set_of_products)
        MyApp.SM.screens[3].ids.description.text += string_of_receipt


class MainScreen(Screen):
    pass


def set_as_column(some_set):
    ingredients_text = ''
    for ing in sorted(list(some_set)):
        ingredients_text += '• ' + ing + '\n'
    ingredients_text += '\n'
    return 'Продукты:\n' + ingredients_text


class Ingredients(Screen):
    links_to_buttons = []

    ingredients = sorted(['Сыр', 'Доширак',
                          'Молоко','Курица',
                           'Макароны'])

    def press_search(self, *args):
        if not boiler:
            self.press_search_rofl()
        else:
            Clock.schedule_once(self.press_search_time, 0.5)

    @staticmethod
    def press_search_time(*args):
        for i in receipts:
            if i[1].issubset(boiler):
                MyApp.SM.screens[2].adding(i)

    @staticmethod
    def press_search_rofl():
        i = ['Солнце', {'Солнце'}, 'Если последние деньги ушли на клубы, не отчаивайся. Представь себя солнечной '
                                   'батареей. Питайся энергией этого желтого выскочки!']
        MyApp.SM.screens[2].adding(i)

    def search_among_ingredients(self, index):
        self.ids.sf.text = ''
        self.ids.sv2.scroll_to(self.links_to_buttons[int(index)], padding=300, animate=True)
        self.links_to_buttons[int(index)].call_ripple_animation_methods(self.links_to_buttons[int(index)])

    def error(self):
        self.ids.sf.error = True

    def on_kv_post_time(self, *args):
        for ingredient in self.ingredients:
            ingredient_button = IngredientButton()
            ingredient_button.set_data(ingredient_button, ingredient)
            self.ids.gd.add_widget(ingredient_button)
        self.ids.sf.bind(on_text_validate=lambda x: self.search_among_ingredients(
            self.ingredients.index(self.convert(self.ids.sf.text))) if self.convert(
            self.ids.sf.text) in self.ingredients else self.error())
        self.ids.gd.add_widget(MDLabel())  # чтобы внизу ScrollView было пустое место

    def convert(self, ingredient):
        s = list(ingredient.casefold())
        s[0] = s[0].upper()
        if "".join(s) in self.ingredients:
            return "".join(s)
        else:
            return ingredient

    def on_kv_post(self, base_widget):
        Clock.schedule_once(self.on_kv_post_time, 0.5)


class Receipts(Screen):
    j = 0
    links_to_cards = []
    labels = []

    def adding(self, i):
        receipt_card = ReceiptCard()
        receipt_card.set_data(i[0], i[1], i[2])
        self.labels.append(i[0])
        self.ids.GL_P.add_widget(receipt_card)
        Receipts.j += 1

    def search(self, index):
        self.ids.sf1.text = ''
        card = self.links_to_cards[int(index)]
        self.ids.sv3.scroll_to(card, padding=100, animate=True)
        card.call_ripple_animation_methods(card)

    def error(self):
        self.ids.sf1.error = True

    def convert(self, label):
        s = list(label.casefold())
        s[0] = s[0].upper()
        if "".join(s) in self.labels:
            return "".join(s)
        else:
            return label

    def on_kv_post(self, base_widget):
        self.ids.sf1.bind(on_text_validate=lambda x: self.search(
            self.labels.index(self.convert(self.ids.sf1.text))) if self.convert(self.ids.sf1.text) in self.labels
            else self.error())

    def press_back(self):
        self.ids.GL_P.clear_widgets()
        Receipts.j = 0
        Receipts.links_to_cards = []
        Receipts.labels = []


# Тут будут лежать все выбранные продукты
boiler = set()

# Тут лежат рецепты
receipts = sorted([['Макароны с сыром', {'Сыр', 'Макароны'},
                    'Варим макароны, добавляем сверху сыр (если нет тёрки, просто порежьте его кубиками). А дальше — '
                    'фантазируем. Сюда же можно добавить кетчупа или любого другого соуса (в магазине их изобилие, '
                    'цена — копейки).\nВ качестве гарнира хорош свежий салат: помидоры и огурцы с растительным '
                    'маслом, петрушка и укроп, капуста и морковь.\nМакароны с сыром всегда выигрывают, '
                    'даже на вечеринке. Добавьте нарезанной ветчины, и вы получаете праздничное блюдо.\nЕщё вариант. '
                    'Сварите макарон. В отдельной кастрюле взбейте молоко (стакан) с мукой (столовая ложка), '
                    'добавив примерно 1/3 чайной ложки соли и немного сахара и перца по вкусу. Постоянно помешивайте, '
                    'доведите до кипения, затем уменьшите огонь и продолжайте помешивать в течение нескольких минут, '
                    'пока соус не загустеет. Затем добавьте измельчённый сыр и перемешивайте до расплавления, '
                    'вылейте сырную смесь на макароны. Можно добавить горох, бекон, кубики ветчины, тунца, '
                    'помидоры или лук-шалот.'],
                   ['Варёная курица', {'Курица'},
                    'Самое простое — сварить. Добавьте куриные ножки в кипящую воду (первую воду после 5-минутного '
                    'кипения можно слить и залить свежей, снова поставить кипятиться, курица получится более '
                    'диетической и не такой "пахучей"), добавить соль, перец горошком, морковку, можно луковицу '
                    'целиком, варить минут 20-30, в конце засыпать лапши, можно добавить лавровый лист, поварить 5 '
                    'минут, готово! Луковицу обычно выбрасывают.'],
                   ['Доширак', {'Доширак'},
                    'Завариваешь дошик да ешь, в чем проблема, мать?'],
                   ], key=lambda receipts: receipts[0])


class FullReceipt(Screen):
    def press_back(self):
        self.ids.img.source = ''
        # self.ids.img.reload()


class MyApp(MDApp):
    title = 'Рецепты'
    running = True

    def on_stop(self):
        self.running = False

    SM = ScreenManager()

    def build(self):
        self.theme_cls.material_style = "M3"
        sm = self.SM
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(Ingredients(name='ingredients'))
        sm.add_widget(Receipts(name='receipts'))
        sm.add_widget(FullReceipt(name='full_receipt'))
        Window.clearcolor = (1, 1, 1, 1)
        Window.bind(on_keyboard=self.on_key)
        return sm

    def on_key(self, window, key, *args):
        if key == 27:                                               # Если нажали "Назад":
            if self.SM.current_screen.name == 'main':               # если текущий экран "Главный",
                return False                                        # выходим из приложения.
            elif self.SM.current_screen.name == 'ingredients':      # Если текущий экран "Ингредиенты",
                self.SM.transition.direction = 'right'              # (анимация перехода назад)
                self.SM.current = 'main'                            # переходим на "Главный"
                return True                                         # и не выходим из приложения.
            elif self.SM.current_screen.name == 'receipts':         # Если текущий экран "Рецепты",
                self.SM.screens[2].press_back()
                self.SM.transition.direction = 'right'              # (анимация перехода назад)
                self.SM.current = 'ingredients'                     # переходим на "Ингредиенты"
                return True
            elif self.SM.current_screen.name == 'full_receipt':     # Если текущий экран "Полный рецепт",
                self.SM.screens[3].press_back()
                self.SM.transition.direction = 'up'                 # (анимация перехода назад)
                self.SM.current = 'receipts'                        # переходим на "Рецепты"
                return True                                         # и не выходим из приложения.


Window.size = (378, 819)

MyApp().run()
