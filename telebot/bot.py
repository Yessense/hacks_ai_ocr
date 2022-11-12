import telebot
import copy
import processing

bot = telebot.TeleBot('5679721050:AAE3v4OweyRgkgcZv8Plu6U6W4MWUPdnk4w')


text_to_audio = processing.TextToVoice()

texts = {0: ""}


def bot_start():
    pass


@bot.message_handler(content_types=["photo", "file"])
def got_image(message):
    # Getting image from user
    file = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file.file_path)

    # Getting bounds from docker container
    bounds = processing.ImgToText.get_bounds(downloaded_file)
    bounds = processing.SpellChecker.check_spelling(bounds)

    # Joining text from picture
    last = -2 if len(bounds) & len(bounds[0]) == 3 else -1
    text = " ".join([i[last] for i in bounds])
    text_id = list(texts.keys())[-1] + 1
    texts[text_id] = text

    # Create processed image with bounds
    img = processing.Visualize.get_visualisation(downloaded_file, bounds)

    # Creating markup
    markup = telebot.types.InlineKeyboardMarkup()
    btn_yandex = telebot.types.InlineKeyboardButton(text='Поиск в Yandex',
                                                    url='https://yandex.ru/search/?text=' + text.replace(" ", "+"))
    text_call_back = 'text|' + str(message.chat.id) + "|" + str(text_id)
    btn_text = telebot.types.InlineKeyboardButton(text='Текст', callback_data=text_call_back)
    audio_call_back = 'audio|' + str(message.chat.id) + "|" + str(text_id)
    btn_audio = telebot.types.InlineKeyboardButton(text='Аудио', callback_data=audio_call_back)
    markup.add(btn_text)
    markup.add(btn_audio)
    yndx_markup = copy.deepcopy(markup)
    yndx_markup.add(btn_yandex)

    # Sending photo to user
    try:
        bot.send_photo(message.chat.id, photo=img, reply_markup=yndx_markup)
    except:
        bot.send_photo(message.chat.id, photo=img, reply_markup=markup)


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    data = call.data.split("|")
    if len(data) > 2 and int(data[2]) in texts:
        if data[0] == 'text':
            bot.send_message(int(data[1]), texts[int(data[2])])
        if data[0] == 'audio':
            voice = text_to_audio.get_voice_rb(texts[int(data[2])])
            if voice != None:
                bot.send_audio(int(data[1]), voice)
            else:
                bot.send_message(int(data[1]), "Не удалось сформировать аудио")


bot_start()

bot.polling(none_stop=True, interval=0)
