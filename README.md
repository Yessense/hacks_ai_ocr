# Распознавание текст с помощью ИИ

## Telegram-бот для распознавания текста с помощью ИИ

1. Создание и отправка голосового сообщения, с распознанным текстом
2. Создание текстового документа по распознанному тексту на фотографии
3. Ссылка на поиск в Яндексе по распознанному тексту.

<p float="left">
        <img src="/images/audio.gif" width="25%" alt="Преобразование в аудио"/>
        <img src="/images/text.gif" width="25%" alt="Вывод текста"/>
        <img src="/images/yandex_search.gif" width="25%" alt="Поиск в Яндексе"/>
</p>

---

## Как пользоваться ботом

### Шаг 1. Запустить контейнеры 

```shell
git clone https://github.com/Yessense/hacks_ai_ocr/tree/project
docker-compose build
docker-compose up
```

### Шаг 2. Отправить фотографию боту

https://t.me/HeroesofMLandMagicBot

---

[//]: # (# Preprocessing)

[//]: # ()
[//]: # (##### Были реализованны следующие функции:)

[//]: # ()
[//]: # (- [X] get_grayscale)

[//]: # (- [X] remove_noise)

[//]: # (- [X] dilate)

[//]: # (- [X] erode)

[//]: # (- [X] opening)

[//]: # (- [X] canny)

[//]: # (- [X] deskew)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (# Postprocessing)

[//]: # ()
[//]: # (## Spell Checker)

[//]: # ()
[//]: # (##### Были реализованны следующие эвристики:)

[//]: # ()
[//]: # (1. Определение на каком языке написанно слово, с учетом присутствия букв, как на русском, так и на английском)

[//]: # (2. Определение регистра слова)

[//]: # (3. Замена '@' на букву 'а', если это не почта)

[//]: # (4. Убираем символы посередине слова, если, например, скобка открывается, а потом не закрывается)

[//]: # (5. Обрабатывем педложение с помощью deeeppavlova, для исправления опечаток и неточностей)
