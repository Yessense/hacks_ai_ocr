# Распознавание текст с помощью ИИ

## Telegram-бот для распознавания текста с помощью ИИ

1. Создание и отправка голосового сообщения, с распознанным текстом
2. Создание текстового документа по распознанному тексту на фотографии
3. Ссылка на поиск в Яндексе по распознанному тексту.

<p float="left">
    <figure>
        <img src="/images/text.gif" width="25%" alt="Вывод текста"/>
        <figcaption>An elephant at sunset</figcaption>

    </figure>

    <figure>
        <img src="/images/yandex_search.gif" width="25%" alt="Поиск в Яндексе"/>
        <figcaption>An elephant at sunset</figcaption>

    </figure>

    <figure>
        <img src="/images/audio.gif" width="25%" alt="Преобразование в аудио"/>
        <figcaption>An elephant at sunset</figcaption>

    </figure>
</p>

#### Для работы необходимо запустить следующие Docker контейнеры

- Telebot
- processing
- pre_processing
- post_processing

---

# Preprocessing

##### Были реализованны следующие функции:

- [X] get_grayscale
- [X] remove_noise
- [X] dilate
- [X] erode
- [X] opening
- [X] canny
- [X] deskew

---

# Postprocessing

## Spell Checker

##### Были реализованны следующие эвристики:

1. Определение на каком языке написанно слово, с учетом присутствия букв, как на русском, так и на английском
2. Определение регистра слова
3. Замена '@' на букву 'а', если это не почта
4. Убираем символы посередине слова, если, например, скобка открывается, а потом не закрывается
5. Обрабатывем педложение с помощью deeeppavlova, для исправления опечаток и неточностей
